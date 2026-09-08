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

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from agentic_erp_assistant.llm.ports import (
    LargeLanguageModelClient,
    Message,
    ToolCallingClient,
    Usage,
    UsageReporting,
)
from agentic_erp_assistant.llm.prompts import build_messages, build_planner_messages
from agentic_erp_assistant.llm.retry import retry_with_backoff
from agentic_erp_assistant.llm.schemas import EvidenceSnippet, GroundedAnswer
from agentic_erp_assistant.llm.telemetry import (
    InMemoryTelemetry,
    ModelCallRecord,
    Outcome,
    TelemetrySink,
    now,
    price,
)
from agentic_erp_assistant.llm.tokenizer import TiktokenCounter, TokenCounter
from agentic_erp_assistant.llm.tools import PLANNING_TOOLS, ToolCallResult, ToolSpec
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


def _as_snippets(evidence: Evidence) -> list[EvidenceSnippet]:
    """Normalize whichever evidence shape the caller had."""
    if isinstance(evidence, Mapping):
        return [
            EvidenceSnippet(source_id=source_id, locator=WHOLE_DOCUMENT, text=text)
            for source_id, text in evidence.items()
        ]
    return list(evidence)


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

    counter: TokenCounter = field(default_factory=TiktokenCounter)
    """How the pre-call estimate is made."""

    max_attempts: int = 4
    """Retry budget for one :meth:`answer` call."""

    sleep: Callable[[float], object] = time.sleep
    """Passed to the retry engine; injectable so tests do not wait."""

    jitter: Callable[[], float] | None = None
    """Passed to the retry engine when set; ``None`` keeps its full-jitter default."""

    def answer(
        self,
        question: str,
        evidence: Evidence,
        *,
        temperature: float = 0.0,
    ) -> GroundedAnswer:
        """Answer one question from the evidence given, or refuse in a typed way.

        Args:
            question: The user's words, placed verbatim in the user block.
            evidence: ``{source_id: text}``, or ``EvidenceSnippet`` objects when
                the caller knows real locators.
            temperature: Defaults to 0.0. A grounded answer is not a place for
                variety, and a reproducible trace is worth more here than range.

        Returns:
            A validated :class:`GroundedAnswer` -- which, by construction, either
            carries citations or says why it refused.

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
        messages = build_messages(question, _as_snippets(evidence))

        # 2. Budget, before anything is sent.
        estimated = self.counter.count_message_tokens(
            messages, model=self.client.model_name
        )
        self._check_budget(estimated)

        # 3. Call, with retry.
        attempts = 0

        def one_attempt():
            nonlocal attempts
            attempts += 1
            return self.client.complete(messages, temperature=temperature)

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
            )
            raise
        latency = time.perf_counter() - started

        # 4. Validate, and record what it really cost.
        try:
            answer = GroundedAnswer.model_validate_json(response["text"])
        except ValidationError as error:
            self._record(
                outcome="invalid_schema",
                estimated=estimated,
                usage=response["usage"],
                latency=latency,
                attempts=attempts,
                model=response["model"],
                detail=f"{error.error_count()} validation error(s)",
            )
            raise

        self._record(
            outcome="answered",
            estimated=estimated,
            usage=response["usage"],
            latency=latency,
            attempts=attempts,
            model=response["model"],
        )
        return answer

    def decide(
        self,
        question: str,
        evidence: Evidence = (),
        observations: Sequence[ToolOutcome] = (),
        *,
        tools: Sequence[ToolSpec] = PLANNING_TOOLS,
        temperature: float = 0.0,
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
            tools: What to offer. Defaults to
                :data:`~agentic_erp_assistant.llm.tools.PLANNING_TOOLS`.
            temperature: 0.0. A routing decision is not a place for variety.

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
            build_planner_messages(question, _as_snippets(evidence), observations),
            tools=tools,
            temperature=temperature,
        )

    def call_tools(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = PLANNING_TOOLS,
        temperature: float = 0.0,
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
        )
        self._check_budget(estimated)

        attempts = 0

        def one_attempt() -> ToolCallResult:
            nonlocal attempts
            attempts += 1
            return client.call_with_tools(
                messages, tools=tools, temperature=temperature
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
            ),
        )
        return decision

    def _check_budget(self, estimated: int) -> None:
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
        )
        logger.warning("refusing request locally: %s", detail)
        raise ContextWindowExceeded(detail)

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
    ) -> None:
        """Price one call and hand it to the sink.

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

        self.telemetry.record(
            ModelCallRecord(
                model=served_model,
                outcome=outcome,
                estimated_input_tokens=estimated,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cost_usd=cost,
                latency_seconds=latency,
                attempts=attempts,
                occurred_at=now(),
                detail="; ".join(part for part in (detail, price_detail) if part)
                or None,
            )
        )
