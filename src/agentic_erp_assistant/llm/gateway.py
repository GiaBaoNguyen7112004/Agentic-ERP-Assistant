"""One question in, one validated answer out -- with the four checks in order.

Every piece this assembles already works on its own: prompts are built in
``prompts.py``, counted in ``tokenizer.py``, sent through a port implemented in
``adapters/``, retried by ``retry.py``, validated by ``schemas.py``, priced by
``pricing.py`` and recorded by ``telemetry.py``. What was missing was a single
place where they happen in a defensible order, so that reading :meth:`answer`
top to bottom tells a reviewer exactly what protects a request:

1. **build** the four role blocks,
2. **budget** -- estimate the request and refuse it locally if it cannot fit,
3. **call** through backoff retry,
4. **validate** the reply and record what it cost.

The order is the design. The budget check sits above the client because a
request that cannot fit must not spend money or consume a retry attempt
discovering that; the validation sits below the retry because an answer that
breaks the schema is a contract failure, not a transport hiccup, and resampling
it four times only costs four times as much.

Telemetry is written on every path that reached the provider, including the ones
that failed. A rejected reply was still generated and still billed.
"""

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

from agentic_erp_assistant.context.catalogue import DocumentCatalogue
from agentic_erp_assistant.llm.inspection import (
    ModelCallInspector,
    ModelRequestSnapshot,
    ModelResponseSnapshot,
    snapshot_request,
    snapshot_response,
)
from agentic_erp_assistant.llm.ports import (
    LargeLanguageModelClient,
    Message,
    TokenEstimating,
    ToolCallingClient,
    ToolChoice,
    Usage,
    UsageReporting,
)
from agentic_erp_assistant.llm.prompts import (
    PLANNER_CONTRACT,
    Principal,
    build_declaration_messages,
    build_messages,
    build_planner_messages,
)
from agentic_erp_assistant.llm.retry import retry_with_backoff
from agentic_erp_assistant.llm.schemas import EvidenceSnippet, GroundedAnswer
from agentic_erp_assistant.llm.streaming import AnswerStreamSink, JsonStringFieldExtractor
from agentic_erp_assistant.llm.telemetry import (
    InMemoryTelemetry,
    ModelCallRecord,
    Outcome,
    TelemetrySink,
    now,
    price,
)
from agentic_erp_assistant.llm.tokenizer import TiktokenCounter, TokenCounter
from agentic_erp_assistant.llm.tools import (
    DECLARE_REPLY_CONTRACT_TOOL,
    PLANNING_TOOLS,
    ToolCallResult,
    ToolSpec,
)
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.state.tool_outcome import ToolOutcome

__all__ = ["ContextWindowExceeded", "Evidence", "LLMGateway", "WHOLE_DOCUMENT"]

logger = logging.getLogger(__name__)


WHOLE_DOCUMENT = "full"
"""The locator used when evidence arrives as ``{source_id: text}``.

A citation needs somewhere to point, and a caller handing over whole documents
has no finer address to give. Retrieval will pass :class:`EvidenceSnippet`
objects with real chunk locators instead, and this constant will stop appearing
in traces -- which is the visible difference between "we cited a document" and
"we cited a passage".
"""

Evidence = Mapping[str, str] | Sequence[EvidenceSnippet]
"""Either whole documents keyed by source id, or already-located snippets."""


class ContextWindowExceeded(Exception):
    """The request cannot fit in the model's window, so it was never sent.

    Not an :class:`~agentic_erp_assistant.llm.ports.LLMClientError`: nothing
    about the provider failed. This is a local refusal, decided from a local
    estimate, and the fix is to retrieve less or reserve less -- never to retry.
    """


def _on_delta_kwarg(on_delta: Callable[[str], None] | None) -> dict[str, object]:
    """``{"on_delta": on_delta}``, or nothing at all.

    Not ``{"on_delta": None}`` when there is no sink: a client that never
    declared the parameter (every fake predating streaming) would raise
    ``TypeError`` on an unexpected keyword. Omitting the key entirely is
    what keeps every such fake a valid client, exactly as the port's
    docstring promises -- the gateway only asks a client to stream when a
    sink is actually bound.
    """
    return {"on_delta": on_delta} if on_delta is not None else {}


def _extra_tokens(
    client: object, *, tools: Sequence[ToolSpec] | None = None, structured: bool = False
) -> int:
    """What :class:`~agentic_erp_assistant.llm.ports.TokenEstimating` adds to
    the message count, or ``0`` for a client that does not satisfy it.

    The optional-capability pattern :class:`~agentic_erp_assistant.llm.ports.
    UsageReporting` already uses in this file: a client written before this
    protocol existed is still a valid one, merely estimated a little low
    exactly as before -- never a ``TypeError`` for lacking a method nothing
    required of it.
    """
    if isinstance(client, TokenEstimating):
        return client.estimate_extra_tokens(tools=tools, structured=structured)
    return 0


def _tool_choice_kwarg(tool_choice: ToolChoice) -> dict[str, object]:
    """``{"tool_choice": "none"}`` / ``{"tool_choice": "required"}``, or
    nothing at all.

    The same reasoning as :func:`_on_delta_kwarg`, for the same reason: a
    fake client written before ADR 0019/0021 declared no ``tool_choice``
    parameter, and passing the default value explicitly would raise
    ``TypeError`` on every one of them for a call that changes nothing about
    what they should do. Only a non-default choice is ever worth a client
    knowing about.
    """
    return {} if tool_choice == "auto" else {"tool_choice": tool_choice}


def _as_snippets(evidence: Evidence) -> list[EvidenceSnippet]:
    """Normalize whichever evidence shape the caller had."""
    if isinstance(evidence, Mapping):
        return [
            EvidenceSnippet(source_id=source_id, locator=WHOLE_DOCUMENT, text=text)
            for source_id, text in evidence.items()
        ]
    return list(evidence)


def _salvage_unresolvable_citations(text: str) -> tuple[str, int]:
    """Drop citations an empty source id or locator leaves unresolvable.

    The provider does not enforce the schema's ``min_length`` (structured
    output accepts the schema but ignores the constraint), so a composer can
    emit ``{"source_id": "doc-1", "locator": ""}`` -- observed in the wild as
    ``citations.N.locator string_too_short`` failing a turn whose other three
    citations and answer text were fine. An empty field is not a resolvable
    reference -- the citation rule's own words -- so the honest repair is to
    drop the pointer, not the turn: dropping only ever removes a citation that
    could not have been checked by a reader, never adds or rewrites one, and
    the schema's grounded-but-uncited rule still rejects what is left if no
    resolvable citation survives. Prose, malformed JSON, and shape errors
    deeper than an empty field are returned untouched -- those are the
    schema's own reports to make.

    Returns the (possibly repaired) payload and how many citations were
    dropped, so the repair is visible in the trace rather than silent.
    """
    try:
        payload = json.loads(text)
    except ValueError:
        return text, 0
    if not isinstance(payload, dict):
        return text, 0
    citations = payload.get("citations")
    if not isinstance(citations, list):
        return text, 0

    def unresolvable(citation: object) -> bool:
        # A non-dict entry is a shape error, not an empty field: the schema
        # reports it, unchanged.
        if not isinstance(citation, dict):
            return False
        return any(
            not isinstance(citation.get(field_name), str)
            or not citation.get(field_name, "").strip()
            for field_name in ("source_id", "locator")
        )

    kept = [citation for citation in citations if not unresolvable(citation)]
    dropped = len(citations) - len(kept)
    if dropped == 0:
        return text, 0
    payload["citations"] = kept
    return json.dumps(payload), dropped


@dataclass
class LLMGateway:
    """The assembled path from a question to a validated, costed answer.

    Depends on :class:`~agentic_erp_assistant.llm.ports.LargeLanguageModelClient`
    -- the port, not an adapter -- so the same gateway drives OpenAI today, a
    fake in tests, and whatever comes next without editing this file.

    Usage::

        with OpenAIChatClient() as client:
            gateway = LLMGateway(client, context_window=128_000)
            answer = gateway.answer(
                "What is the refund window?",
                {"doc-1": "Refunds close after 30 days."},
            )
    """

    client: LargeLanguageModelClient
    """Where completions come from."""

    context_window: int
    """The model's real, published context window, in tokens.

    Required, with no default, and that is the whole point of the field. A
    default would be one model's number applied to whichever model ``.env``
    happens to name -- which is precisely the bug the budget check exists to
    catch, reintroduced one layer up. Look it up for the model you chose.
    """

    output_reserve: int = 1024
    """Tokens held back for the reply.

    The window is shared between prompt and completion, so a prompt that fits
    exactly still fails the moment the model answers. This is the room left for
    it, and it is deliberately a constructor argument: a gateway asking for long
    grounded answers needs more of it than one doing classification.
    """

    telemetry: TelemetrySink = field(default_factory=InMemoryTelemetry)
    """Where the cost record goes. Swapped for the trace store when it exists."""

    inspector: ModelCallInspector | None = None
    """Where a call's live I/O goes, beside its cost record. ``None`` --
    the default -- means nobody is watching this gateway's calls live, and
    every call site in this class skips building a snapshot at all rather
    than building one nobody reads. Bound the same way ``stream`` is
    (``composition/turn.py``): per turn, to whatever is watching it."""

    inspect_io: bool = False
    """Whether a snapshot actually carries the request/reply text, when an
    :attr:`inspector` is bound at all. ``False`` -- the default -- means
    :attr:`inspector` still learns about every call (paired with ``None``,
    ``None`` for its request and response), but the prompt and reply
    themselves never leave this process. The one caller that sets this is
    ``composition/turn.py``, reading ``DEV_TRACE_MODEL_IO``."""

    counter: TokenCounter = field(default_factory=TiktokenCounter)
    """How the pre-call estimate is made."""

    max_attempts: int = 4
    """Retry budget for one :meth:`answer` call."""

    sleep: Callable[[float], object] = time.sleep
    """Passed to the retry engine; injectable so tests do not wait."""

    jitter: Callable[[], float] | None = None
    """Passed to the retry engine when set; ``None`` keeps its full-jitter default."""

    stream: AnswerStreamSink | None = None
    """Where reply text goes while it is still arriving. ``None`` -- the
    default -- means this gateway never asks the client to stream at all.

    Bound here rather than threaded through :meth:`answer` and
    :meth:`call_tools` as a parameter: the planner and the composer stay
    ignorant of streaming either way, and a second gateway built for memory
    work (:class:`~agentic_erp_assistant.memory.extractor.LLMMemoryProposer`
    and its summary sibling) is built with no sink, so a memory proposal can
    never stream into the chat. Each retried
    attempt gets ``reset()`` called on it before its first delta -- a stream
    that died partway is replayed from the top on the next attempt, and a
    sink that was not told would show the reply twice.
    """

    planner_contract: str = PLANNER_CONTRACT
    """The developer block :meth:`decide` builds its prompt with.

    Defaults to the production constant; nothing in ``composition/`` sets it
    to anything else. The one caller that does is ``eval/routing.py``'s
    comparison (ADR 0020) -- one gateway per candidate contract, so a
    three-way comparison is three constructor calls, not three copies of
    :meth:`decide`.
    """

    principal: Principal | None = None
    """Who this turn is for, appended to the system block of every prompt
    this gateway builds. ``None`` -- the default -- sends
    :data:`~agentic_erp_assistant.llm.prompts.SYSTEM_POLICY` byte-for-byte.

    Bound at construction, like :attr:`stream` and :attr:`inspector`, so
    :meth:`decide` and the composer stay ignorant of it: the planner asks
    for a decision, and who the turn is for is standing session context,
    not something a decision carries. ``call_tools`` takes messages already
    built, so it needs nothing here."""

    catalogue: DocumentCatalogue | None = None
    """Which documents this turn's actor may search (ADR 0026), appended to
    the system block of :meth:`decide`'s own prompt only. ``None`` -- the
    default -- omits the block, the same as an unset :attr:`principal` does
    for the whole system role.

    Bound at construction, exactly like :attr:`principal` -- which document
    catalogue applies is standing session context, decided once by
    ``composition/turn.py`` from the actor's own entitlements, never
    something a single decision carries. :meth:`answer`, :meth:`declare` and
    a memory-proposal gateway built without one never see this field at
    all: composing from evidence already retrieved, and declaring what a
    reply needs in the abstract, have no occasion to weigh whether a
    specific document exists."""

    def answer(
        self,
        question: str,
        evidence: Evidence,
        memories: Sequence[MemoryRecord] = (),
        history: Sequence[ConversationTurn] = (),
        observations: Sequence[ToolOutcome] = (),
        *,
        temperature: float = 0.0,
    ) -> GroundedAnswer:
        """Answer one question from the evidence given, or refuse in a typed way.

        Args:
            question: The user's words, placed verbatim in the user block.
            evidence: ``{source_id: text}``, or ``EvidenceSnippet`` objects when
                the caller knows real locators.
            memories: What recall selected for this turn. Reaches the answering
                call as well as the routing one because a stored preference is
                about how to reply -- and it cannot become a citation, since a
                memory has no locator and the caller checks every citation
                against the evidence actually retrieved.
            history: The session's recent turns, already clipped and budgeted.
                Reaches the answering call for the same reason memory does: an
                answer resolving a follow-up needs the antecedent it was
                resolved against, and it cannot become a citation either.
            observations: What this turn's own tool calls returned, in order
                (ADR 0021). Empty on a turn that never called one before
                retrieving. Cannot become a citation either, for the same
                structural reason -- no locator -- but reaches this call for a
                sharper one: a compound question redirected to retrieval
                after a tool call already succeeded needs both composed
                together, not just the passage half.
            temperature: Defaults to 0.0. A grounded answer is not a place for
                variety, and a reproducible trace is worth more here than range.

        Returns:
            A validated :class:`GroundedAnswer` -- which, by construction, either
            carries citations or says why it refused. A citation whose source
            id or locator arrives empty is dropped before validation rather
            than allowed to fail the turn -- it is not a resolvable reference,
            and the drop is recorded in the call's trace detail. A reply left
            with no resolvable citation still fails the schema below.

        Raises:
            ContextWindowExceeded: The estimate does not leave room for a reply.
                Raised before any request is made.
            ValidationError: A reply came back and did not satisfy the schema.
                Recorded, then raised: it is a contract failure, and the decision
                about what a user sees belongs to the guardrail layer above.
            TransientProviderError: Retries were exhausted.
            ProviderAuthError: A definitive rejection from the provider.
            ValueError: ``question`` is blank (from ``build_messages``).
        """
        # 1. Build.
        messages = build_messages(
            question, _as_snippets(evidence), memories, history, observations,
            principal=self.principal,
        )

        # 2. Budget, before anything is sent.
        estimated = self.counter.count_message_tokens(
            messages, model=self.client.model_name
        ) + _extra_tokens(self.client, structured=True)
        request_snapshot = self._snapshot_request("answer", messages, temperature=temperature)
        self._check_budget(estimated, request=request_snapshot)

        # 3. Call, with retry.
        attempts = 0

        def one_attempt():
            nonlocal attempts
            attempts += 1
            on_delta = None
            if self.stream is not None:
                if attempts > 1:
                    self.stream.reset()
                extractor = JsonStringFieldExtractor("answer")
                sink = self.stream

                def on_delta(fragment: str) -> None:
                    piece = extractor.feed(fragment)
                    if piece:
                        sink.delta(piece)

            return self.client.complete(
                messages, temperature=temperature, **_on_delta_kwarg(on_delta)
            )

        started = time.perf_counter()
        try:
            response = retry_with_backoff(
                one_attempt,
                max_attempts=self.max_attempts,
                sleep=self.sleep,
                **({"jitter": self.jitter} if self.jitter is not None else {}),
            )
        except Exception as error:
            # Every attempt that reached the provider was billed, whether or not
            # anything usable came back. last_usage is the only record of that
            # spend once the exception replaced the response.
            self._record(
                outcome="provider_failure",
                estimated=estimated,
                usage=self._reported_usage(),
                latency=time.perf_counter() - started,
                attempts=attempts,
                detail=f"{type(error).__name__}: {error}",
                request=request_snapshot,
            )
            raise
        latency = time.perf_counter() - started

        # 4. Validate, and record what it really cost. One repair runs first
        # (see _salvage_unresolvable_citations): a citation an empty field
        # leaves unresolvable is dropped, and the drop itself is recorded.
        salvaged, dropped = _salvage_unresolvable_citations(response["text"])
        try:
            answer = GroundedAnswer.model_validate_json(salvaged)
        except ValidationError as error:
            self._record(
                outcome="invalid_schema",
                estimated=estimated,
                usage=response["usage"],
                latency=latency,
                attempts=attempts,
                model=response["model"],
                detail=f"{error.error_count()} validation error(s)",
                request=request_snapshot,
                response=self._snapshot_response(content=response["text"]),
            )
            raise

        self._record(
            outcome="answered",
            estimated=estimated,
            usage=response["usage"],
            latency=latency,
            attempts=attempts,
            model=response["model"],
            detail=(
                f"dropped {dropped} citation(s) with an empty source id or locator"
                if dropped
                else None
            ),
            request=request_snapshot,
            response=self._snapshot_response(
                content=response["text"], stop_reason=response["stop_reason"]
            ),
        )
        return answer

    def decide(
        self,
        question: str,
        evidence: Evidence = (),
        observations: Sequence[ToolOutcome] = (),
        memories: Sequence[MemoryRecord] = (),
        history: Sequence[ConversationTurn] = (),
        *,
        tools: Sequence[ToolSpec] = PLANNING_TOOLS,
        temperature: float = 0.0,
        tool_choice: ToolChoice = "auto",
    ) -> ToolCallResult:
        """Ask the model what to do next, and report the one choice it made.

        The sibling of :meth:`answer`, through the same four steps -- build,
        budget, call with retry, record -- because a routing call costs money
        and can fail exactly like an answering one, and a decision that skipped
        the budget check would be the request that blows the window right
        before the reply that needed the room.

        What comes back is transport-shaped on purpose: a tool name and parsed
        arguments, or content. Turning that into a route is policy
        (:mod:`agentic_erp_assistant.reasoning.planner`), and policy in the
        gateway would be a second place routing is decided.

        Args:
            question: The user's words, verbatim.
            evidence: Whatever retrieval has supplied so far this turn.
            observations: What this turn's calls have returned, in order.
            memories: What recall selected for this turn, already filtered and
                budgeted -- never everything that is stored.
            history: The session's recent turns, already clipped and budgeted.
            tools: What to offer. Defaults to
                :data:`~agentic_erp_assistant.llm.tools.PLANNING_TOOLS`.
            temperature: 0.0. A routing decision is not a place for variety.
            tool_choice: ``"auto"`` (default), ``"none"`` (ADR 0019 -- forces
                content back even though ``tools`` is still offered), or
                ``"required"`` (ADR 0021 -- forces a call back).

        Returns:
            A :class:`~agentic_erp_assistant.llm.tools.ToolCallResult`.

        Raises:
            TypeError: The configured client cannot make tool calls. Raised
                before anything is sent, and raised rather than routed: a
                gateway wired to a text-only client is a deployment mistake,
                not a turn that failed.
            ContextWindowExceeded: The estimate does not leave room for a reply.
            TransientProviderError: Retries were exhausted.
            ProviderAuthError: A definitive rejection from the provider.
            ValueError: ``question`` is blank, or ``tools`` is empty.
        """
        return self.call_tools(
            build_planner_messages(
                question,
                _as_snippets(evidence),
                observations,
                memories,
                history,
                contract=self.planner_contract,
                catalogue=self.catalogue,
                principal=self.principal,
            ),
            tools=tools,
            temperature=temperature,
            tool_choice=tool_choice,
        )

    def declare(
        self, question: str, history: Sequence[ConversationTurn] = ()
    ) -> ToolCallResult:
        """Ask what a complete reply to ``question`` must rest on (ADR 0021).

        The sibling of :meth:`decide`, offering exactly one function
        (:data:`~agentic_erp_assistant.llm.tools.DECLARE_REPLY_CONTRACT_TOOL`)
        with ``tool_choice="required"``, so the result is always a call, never
        the prose a declaration call exists to prevent. Goes through
        :meth:`call_tools` like every other function-calling path in this
        class, so it is budgeted, retried and recorded exactly like a routing
        decision -- a declaration call costs real tokens and can fail exactly
        like one.

        Args:
            question: The user's words, verbatim.
            history: The session's recent turns, already clipped and
                budgeted -- so a reference like "that milestone" can be
                resolved the same way a routing decision resolves it.

        Returns:
            A :class:`~agentic_erp_assistant.llm.tools.ToolCallResult`
            naming ``declare_reply_contract`` with parsed arguments.
            :meth:`~agentic_erp_assistant.reasoning.planner.Planner.declare`
            is what turns this into a
            :class:`~agentic_erp_assistant.state.reply_contract.ReplyContract`,
            never-fail.

        Raises:
            ContextWindowExceeded: The estimate does not leave room for a
                reply.
            TransientProviderError: Retries were exhausted.
            ProviderAuthError: A definitive rejection from the provider.
            ValueError: ``question`` is blank.
        """
        return self.call_tools(
            build_declaration_messages(question, history, principal=self.principal),
            tools=(DECLARE_REPLY_CONTRACT_TOOL,),
            temperature=0.0,
            tool_choice="required",
        )

    def call_tools(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = PLANNING_TOOLS,
        temperature: float = 0.0,
        tool_choice: ToolChoice = "auto",
    ) -> ToolCallResult:
        """Offer ``tools`` against an already-built prompt and return the choice.

        The body :meth:`decide` used to be, with the message building lifted
        out. It exists because routing is not the only thing this project asks a
        model to answer with a function call: the memory layer asks what a
        finished turn is worth remembering, in a different prompt, and the
        alternative was for it to reach the client directly.

        That alternative is the one worth arguing against. It would skip the
        budget check, the retry engine and the cost record -- so a proposal made
        after every turn would spend money nothing counted, retry nothing, and
        blow the context window on the largest conversation rather than being
        refused locally. The four steps are the point of this class, and a
        second caller is a reason to share them, not to route around them.

        What stays with the caller is the prompt. Building one is where the
        roles are decided -- which block is instruction and which is data -- and
        that is a decision each caller has to make in the open rather than
        inherit from a shared default.

        Args:
            messages: The role blocks, already built. See
                :mod:`agentic_erp_assistant.llm.prompts`.
            tools: What to offer.
            temperature: 0.0. Neither a routing decision nor a memory proposal
                is a place for variety.
            tool_choice: ``"auto"`` (default), ``"none"`` (ADR 0019), or
                ``"required"`` (ADR 0021). A non-default choice is recorded in
                the telemetry row's ``detail`` so a trace shows a call was
                forced, not merely answered.

        Returns:
            A :class:`~agentic_erp_assistant.llm.tools.ToolCallResult`.

        Raises:
            TypeError: The client cannot make tool calls.
            ContextWindowExceeded: The estimate leaves no room for a reply.
            TransientProviderError: Retries were exhausted.
            ProviderAuthError: A definitive rejection from the provider.
            ValueError: ``tools`` is empty.
        """
        client = self.client
        if not isinstance(client, ToolCallingClient):
            raise TypeError(
                f"{type(client).__name__} cannot make tool calls, so this "
                f"gateway cannot route; wire a client satisfying "
                f"ToolCallingClient to use decide()"
            )

        estimated = self.counter.count_message_tokens(
            messages, model=client.model_name
        ) + _extra_tokens(client, tools=tools)
        request_snapshot = self._snapshot_request(
            "tools",
            messages,
            tools=[tool.name for tool in tools],
            tool_choice=tool_choice,
            temperature=temperature,
        )
        self._check_budget(estimated, request=request_snapshot)

        attempts = 0

        def one_attempt() -> ToolCallResult:
            nonlocal attempts
            attempts += 1
            on_delta = None
            if self.stream is not None:
                if attempts > 1:
                    self.stream.reset()
                on_delta = self.stream.delta

            return client.call_with_tools(
                messages,
                tools=tools,
                temperature=temperature,
                **_on_delta_kwarg(on_delta),
                **_tool_choice_kwarg(tool_choice),
            )

        started = time.perf_counter()
        try:
            decision = retry_with_backoff(
                one_attempt,
                max_attempts=self.max_attempts,
                sleep=self.sleep,
                **({"jitter": self.jitter} if self.jitter is not None else {}),
            )
        except Exception as error:
            self._record(
                outcome="provider_failure",
                estimated=estimated,
                usage=self._reported_usage(),
                latency=time.perf_counter() - started,
                attempts=attempts,
                detail=f"{type(error).__name__}: {error}",
                request=request_snapshot,
            )
            raise
        latency = time.perf_counter() - started

        # No response object comes back from a decision -- the port returns the
        # choice, not the envelope -- so the provider's own usage is the only
        # record of what it cost. Zeros when the client cannot report it, which
        # is visible in the trace rather than silently absent.
        self._record(
            outcome="routed",
            estimated=estimated,
            usage=self._reported_usage(),
            latency=latency,
            attempts=attempts,
            detail=(
                f"chose {decision.tool_name}"
                if decision.tool_name
                else "answered without a tool"
            )
            + ("" if tool_choice == "auto" else f" (tool_choice={tool_choice})"),
            request=request_snapshot,
            response=self._snapshot_response(
                tool_name=decision.tool_name,
                arguments=decision.arguments,
                content=decision.content,
            ),
        )
        return decision

    def _check_budget(
        self, estimated: int, *, request: ModelRequestSnapshot | None = None
    ) -> None:
        """Refuse a request that cannot fit, before it costs anything.

        The comparison is strictly greater than: a request that fills the window
        exactly, reserve included, still fits. Refusing at equality would be a
        second, invisible margin on top of ``output_reserve``.
        """
        needed = estimated + self.output_reserve
        if needed <= self.context_window:
            return

        detail = (
            f"{estimated} estimated prompt tokens + {self.output_reserve} "
            f"reserved for the reply exceeds the {self.context_window}-token "
            f"window of {self.client.model_name} by "
            f"{needed - self.context_window}"
        )
        self._record(
            outcome="budget_exceeded",
            estimated=estimated,
            usage={"input_tokens": 0, "output_tokens": 0},
            latency=0.0,
            attempts=0,
            detail=detail,
            request=request,
        )
        logger.warning("refusing request locally: %s", detail)
        raise ContextWindowExceeded(detail)

    # -- the live inspector, off by default -----------------------------

    def _snapshot_enabled(self) -> bool:
        """Whether building a snapshot at all is worth the caller's while --
        `inspector` bound and `inspect_io` on. Checked at every call site
        before building one, so a gateway nobody is watching, or one only
        counting calls without their text, never pays for a snapshot
        nothing will read."""
        return self.inspector is not None and self.inspect_io

    def _snapshot_request(
        self,
        kind: Literal["answer", "tools"],
        messages: Sequence[Message],
        *,
        tools: Sequence[str] = (),
        tool_choice: str | None = None,
        temperature: float = 0.0,
    ) -> ModelRequestSnapshot | None:
        if not self._snapshot_enabled():
            return None
        return snapshot_request(
            kind, messages, tools=tools, tool_choice=tool_choice, temperature=temperature
        )

    def _snapshot_response(
        self,
        *,
        content: str | None = None,
        tool_name: str | None = None,
        arguments: Mapping[str, object] | None = None,
        stop_reason: str | None = None,
    ) -> ModelResponseSnapshot | None:
        if not self._snapshot_enabled():
            return None
        return snapshot_response(
            content=content, tool_name=tool_name, arguments=arguments, stop_reason=stop_reason
        )

    def _reported_usage(self) -> Usage:
        """What the provider last said it billed, when no response survived.

        Reads the optional
        :class:`~agentic_erp_assistant.llm.ports.UsageReporting` protocol rather
        than an attribute that may not exist, and falls back to zeros for a
        client that cannot report -- zeros meaning "not measured" here, which is
        why the record's ``detail`` always carries the failure alongside them.
        """
        if not isinstance(self.client, UsageReporting):
            return {"input_tokens": 0, "output_tokens": 0}

        reported = self.client.last_usage or {}
        prompt_tokens = reported.get("prompt_tokens")
        completion_tokens = reported.get("completion_tokens")
        return {
            "input_tokens": prompt_tokens if isinstance(prompt_tokens, int) else 0,
            "output_tokens": (
                completion_tokens if isinstance(completion_tokens, int) else 0
            ),
        }

    def _record(
        self,
        *,
        outcome: Outcome,
        estimated: int,
        usage: Usage,
        latency: float,
        attempts: int,
        model: str | None = None,
        detail: str | None = None,
        request: ModelRequestSnapshot | None = None,
        response: ModelResponseSnapshot | None = None,
    ) -> None:
        """Price one call, hand it to the cost sink, and -- if anyone is
        watching this gateway live -- hand the same record to the inspector
        alongside whatever request/response snapshot the call site built.

        Prices from the provider's reported counts, never from the estimate --
        the estimate is what decided whether to send, and the invoice is what
        the provider says it billed. Both end up in the record so the gap is
        visible.
        """
        served_model = model or self.client.model_name
        cost, price_detail = price(
            served_model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )

        record = ModelCallRecord(
            model=served_model,
            outcome=outcome,
            estimated_input_tokens=estimated,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_usd=cost,
            latency_seconds=latency,
            attempts=attempts,
            occurred_at=now(),
            detail="; ".join(part for part in (detail, price_detail) if part) or None,
        )
        self.telemetry.record(record)

        if self.inspector is not None:
            try:
                self.inspector.model_call(record, request, response)
            except Exception:  # noqa: BLE001 - telemetry never fails a request
                logger.warning(
                    "model call inspector raised; dropping this call's live "
                    "snapshot",
                    exc_info=True,
                )
