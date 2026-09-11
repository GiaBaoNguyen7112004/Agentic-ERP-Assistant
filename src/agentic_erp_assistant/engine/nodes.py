"""The three things a turn can be doing, as functions from state to state.

One node per execution shape, and the shapes are the routes: thinking, searching
documents, running a tool. A node takes an
:class:`~agentic_erp_assistant.state.agent_state.AgentState` and returns the next
one, having moved through
:func:`~agentic_erp_assistant.engine.transitions.advance` -- so a node cannot
invent a move the table does not declare, and cannot end a turn without leaving
a response or a failure behind.

What a node is not allowed to do
--------------------------------

Loop. A node performs one action and returns; the runtime decides whether
another node runs. That is what keeps the step budget meaningful -- a node that
retried internally would spend time and money outside the counter that is
supposed to bound it. The one place this shows is the rate-limit path in
:meth:`GraphNodes.execute_tool`: it returns a state on the *same* route, with
``retry_count`` raised, and the engine calls it again. A retry is a repeated node
execution, never a graph edge, which is why ``call_tool -> call_tool`` is not in
the transition table and should not be.

Where the reason-act cycle actually closes
------------------------------------------

A successful tool call routes back to ``think`` rather than straight to an
answer. That costs a second model call and buys the thing the loop exists for:
the model sees what its own action returned and decides whether that is enough,
whether another call is needed, or whether the result changes the question. An
edge from ``call_tool`` to ``answer`` would make the graph a single shot with a
planner attached, and the step budget would be guarding a cycle that could not
happen.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from agentic_erp_assistant.reasoning.decision import DecisionRoute
from agentic_erp_assistant.engine.ports import (
    AnswerComposerPort,
    DocumentRetrieverPort,
    PlannerPort,
    ToolGatewayPort,
)
from agentic_erp_assistant.engine.transitions import advance
from agentic_erp_assistant.state.agent_state import (
    AgentState,
    ERROR_DETAIL_MAX_CHARS,
)
from agentic_erp_assistant.state.events import (
    EVENT_DETAIL_MAX_CHARS,
    EventKind,
    TraceEvent,
)
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.state.tool_request import ToolRequest

__all__ = [
    "EVIDENCE_LIMIT",
    "GraphNodes",
    "NO_EVIDENCE_REPLY",
    "Node",
    "NodeTable",
    "RETRIEVAL_TOOL",
    "SOURCES_PREFIX",
]

logger = logging.getLogger(__name__)


Node = Callable[[AgentState], AgentState]
"""One step of the graph: a state in, the next state out."""

NodeTable = Mapping[DecisionRoute | None, Node]
"""Which node runs for a state sitting on which route.

``None`` is a real key: an unrouted state is where every turn starts, and a
table that began at the second step would leave the first one undispatched.
"""


EVIDENCE_LIMIT = 4
"""How many passages one retrieval pulls.

A runtime decision rather than a retriever default, for the reason
:class:`~agentic_erp_assistant.engine.ports.DocumentRetrieverPort` gives: it
sets the size of every prompt, and a number living inside a backend would change
that from a place no reviewer looks.
"""


RETRIEVAL_TOOL = "search_project_documents"
"""The offered function that means "search", carried on the state as the tool
this turn chose.

Retrieval is chosen exactly like a tool and is not run like one, so the choice
is recorded in ``tool_name``/``tool_arguments`` while the execution happens
against the retriever port. A reviewer reading a trace sees the same shape for
every decision the model made, which is the point.
"""


NO_EVIDENCE_REPLY = (
    "I could not find anything in the project documents that answers that, so "
    "I am not going to guess."
)
"""What an empty retrieval says. A refusal, not an apology with a hedge in it."""


SOURCES_PREFIX = "Sources: "
"""How a reply lists what backs it. One prefix, so a UI can find the line."""


def _clip(text: str, limit: int) -> str:
    """Cut free text to a field's cap, marking that it was cut.

    Traces are capped for a reason (see
    :data:`~agentic_erp_assistant.state.events.EVENT_DETAIL_MAX_CHARS`), and a
    node that let a provider's error message overflow one would turn a bad turn
    into a failed construction.
    """
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _event(node: str, kind: EventKind, detail: str = "") -> TraceEvent:
    return TraceEvent(node=node, kind=kind, detail=_clip(detail, EVENT_DETAIL_MAX_CHARS))


def _with_sources(body: str, sources: Sequence[str]) -> str:
    """Append the identifiers a reply rests on, if there are any.

    Deterministic, and built from what was actually retrieved or returned rather
    than from anything a model wrote. A citation the model composed is a claim;
    this is a record.
    """
    if not sources:
        return body
    return f"{body}\n\n{SOURCES_PREFIX}{', '.join(sources)}"


@dataclass(frozen=True)
class GraphNodes:
    """The nodes, bound to the four ports they run against.

    Frozen, and every collaborator is a protocol. A test builds this with four
    fakes and gets the real routing logic; nothing here knows what a retriever
    indexes, what a gateway executes, or who serves the model.
    """

    retriever: DocumentRetrieverPort
    tools: ToolGatewayPort
    planner: PlannerPort
    composer: AnswerComposerPort

    evidence_limit: int = EVIDENCE_LIMIT
    """How many passages to pull per search."""

    tool_retry_budget: int = 1
    """How many times one turn may wait out a rate limit and try again.

    A budget, not a policy of persistence: the limiter says how long to wait and
    this says how many times that is worth doing. Zero is a legitimate setting
    and means "report the throttle, do not wait".
    """

    sleep: Callable[[float], object] = lambda seconds: None
    """How to wait between rate-limited attempts.

    Defaults to not waiting at all, which is the right default for a component
    whose tests must not sleep. A deployment passes ``time.sleep``; the runtime
    passes whatever it was given.
    """

    def table(self) -> NodeTable:
        """Which node handles which route.

        A table, not a chain of ``if``. A route with no entry is a routing bug
        the engine reports by name, rather than a state that silently does
        nothing and looks like a hang.
        """
        return {
            None: self.think,
            "think": self.think,
            "retrieve_project_documents": self.retrieve_and_answer,
            "call_tool": self.execute_tool,
        }

    # -- reason ------------------------------------------------------------

    def think(self, state: AgentState) -> AgentState:
        """Ask the planner what to do next, and move to whatever it said.

        The rationale goes into the trace as the ``route_selected`` detail --
        capped, and never read by a branch. What the graph acts on is the typed
        route beside it.

        A planner that raises is a provider that failed, which is a turn that
        ends rather than an exception the caller has to catch: the run is
        already half-recorded, and the trace has to say why it stopped.
        """
        try:
            decision = self.planner.plan(state)
        except Exception as error:  # noqa: BLE001 - deliberately broad; see above
            logger.warning("planner failed on %s: %s", state.trace_id, error)
            return advance(
                state,
                "fail",
                mutating=state.tool_mutating if state.route == "call_tool" else None,
                failure="provider_failure",
                error_detail=_clip(
                    f"{type(error).__name__}: {error}", ERROR_DETAIL_MAX_CHARS
                ),
                events=state.events
                + (
                    _event("think", "failed", f"planner raised {type(error).__name__}"),
                ),
            )

        events = state.events + (
            _event(
                "think",
                "route_selected",
                f"{decision.route}: {decision.rationale}",
            ),
        )

        if decision.route == "retrieve_project_documents":
            return advance(
                state,
                decision.route,
                tool_name=RETRIEVAL_TOOL,
                tool_arguments={"query": decision.search_query},
                tool_mutating=False,
                events=events,
            )

        if decision.route == "request_approval":
            # ADR 0016: a write is put to a human only after the checks that
            # would refuse it anyway have already passed. The transition
            # guard still forbids an unapproved write from reaching
            # call_tool -- this only decides whether the human is asked at
            # all.
            outcome = self.tools.preflight(
                ToolRequest(
                    trace_id=state.trace_id,
                    tool_name=decision.required_tool,
                    arguments=decision.tool_arguments or {},
                    actor=state.actor,
                    project_code=state.project_code,
                    scopes=state.scopes,
                    # Deliberately not state.approval: that field describes
                    # the *previous* tool call this turn made, if any -- an
                    # approved write earlier in the same turn must not make a
                    # second, unrelated write look pre-approved here. Nothing
                    # has been decided yet about *this* call, so the honest
                    # value is the request's own default.
                    approval="not_required",
                )
            )
            if outcome.status != "approval_required":
                return advance(
                    state,
                    "fail",
                    tool_name=decision.required_tool,
                    tool_arguments=decision.tool_arguments,
                    tool_mutating=decision.mutating,
                    observations=state.observations + (outcome,),
                    failure="tool_failure",
                    error_detail=_clip(
                        f"{outcome.tool_name} -> {outcome.status}: "
                        f"{outcome.error or ''}",
                        ERROR_DETAIL_MAX_CHARS,
                    ),
                    events=events
                    + (
                        _event(
                            "think",
                            "failed",
                            f"{outcome.tool_name} refused before approval: "
                            f"{outcome.status}",
                        ),
                    ),
                )
            return advance(
                state,
                decision.route,
                tool_name=decision.required_tool,
                tool_arguments=decision.tool_arguments,
                tool_mutating=decision.mutating,
                approval="pending",
                observations=state.observations + (outcome,),
                events=events
                + (
                    _event(
                        "think",
                        "approval_requested",
                        f"{decision.required_tool} needs a human",
                    ),
                ),
            )

        if decision.route == "call_tool":
            return advance(
                state,
                decision.route,
                mutating=decision.mutating,
                tool_name=decision.required_tool,
                tool_arguments=decision.tool_arguments,
                tool_mutating=decision.mutating,
                events=events,
            )

        if decision.route == "answer":
            return advance(
                state,
                "answer",
                response=_with_sources(
                    decision.message or "",
                    _observed_sources(state.observations),
                ),
                events=events,
            )

        if decision.route in ("clarify", "refuse"):
            return advance(state, decision.route, response=decision.message, events=events)

        return advance(
            state,
            "fail",
            failure="provider_failure",
            error_detail=_clip(decision.rationale, ERROR_DETAIL_MAX_CHARS)
            or "the planner could not produce an actionable decision",
            events=events + (_event("think", "failed", decision.rationale),),
        )

    # -- act: documents ----------------------------------------------------

    def retrieve_and_answer(self, state: AgentState) -> AgentState:
        """Search, then answer from what came back -- or refuse, citing nothing.

        Retrieval and composition are one node because they are one obligation:
        an answer to a document question exists only if passages back it. Split
        across two nodes, there would be a state in between where the graph held
        evidence and had not yet decided whether it was enough, and something
        would eventually answer from it.
        """
        query = self._query(state)
        try:
            snippets = tuple(self.retriever.search(query, limit=self.evidence_limit))
        except Exception as error:  # noqa: BLE001 - a failed turn, not a crash
            logger.warning("retriever failed on %s: %s", state.trace_id, error)
            return advance(
                state,
                "fail",
                failure="provider_failure",
                error_detail=_clip(
                    f"{type(error).__name__}: {error}", ERROR_DETAIL_MAX_CHARS
                ),
                events=state.events
                + (
                    _event(
                        "retrieve", "failed", f"retriever raised {type(error).__name__}"
                    ),
                ),
            )
        events = state.events + (
            _event(
                "retrieve",
                "evidence_retrieved",
                f"{len(snippets)} passage(s) for {query!r}",
            ),
        )

        if not snippets:
            return advance(
                state,
                "refuse",
                evidence=(),
                response=NO_EVIDENCE_REPLY,
                failure="insufficient_evidence",
                events=events + (_event("retrieve", "failed", "no passages"),),
            )

        try:
            answer = self.composer.answer(
                state.request, snippets, state.memories, state.history
            )
        except Exception as error:  # noqa: BLE001 - a failed turn, not a crash
            logger.warning("composer failed on %s: %s", state.trace_id, error)
            return advance(
                state,
                "fail",
                evidence=snippets,
                failure="provider_failure",
                error_detail=_clip(
                    f"{type(error).__name__}: {error}", ERROR_DETAIL_MAX_CHARS
                ),
                events=events
                + (
                    _event(
                        "retrieve", "failed", f"composer raised {type(error).__name__}"
                    ),
                ),
            )

        problem = _ungrounded(answer, snippets)
        if problem is not None:
            return advance(
                state,
                "refuse",
                evidence=snippets,
                response=answer.refusal_reason or NO_EVIDENCE_REPLY,
                failure="insufficient_evidence",
                error_detail=_clip(problem, ERROR_DETAIL_MAX_CHARS),
                events=events + (_event("retrieve", "failed", problem),),
            )

        return advance(
            state,
            "answer",
            evidence=snippets,
            response=_with_sources(
                answer.answer,
                [f"[{cite.source_id}#{cite.locator}]" for cite in answer.citations],
            ),
            events=events,
        )

    def _query(self, state: AgentState) -> str:
        """What to search for: the planner's query, or the request behind it.

        The planner's query is preferred because it is the part of the question
        that has to be looked up. Falling back to the raw request rather than
        failing keeps a hand-built state -- a replay, a test, a resumed run --
        runnable without a planner having been involved.
        """
        arguments = state.tool_arguments or {}
        query = arguments.get("query")
        return query if isinstance(query, str) and query.strip() else state.request

    # -- act: tools --------------------------------------------------------

    def execute_tool(self, state: AgentState) -> AgentState:
        """Run the chosen call through the gateway and route on what came back.

        The gateway is the authority on whether the call may happen: this node
        hands it the actor, their scopes and the recorded approval and reads the
        status it returns. It never decides that a call is permitted, which is
        why an ``approval_required`` coming back is routed to a human rather
        than treated as a contradiction -- policy can escalate a read, and only
        the registry knows that it did.
        """
        outcome = self.tools.execute(
            ToolRequest(
                trace_id=state.trace_id,
                tool_name=state.tool_name or "",
                arguments=state.tool_arguments or {},
                actor=state.actor,
                project_code=state.project_code,
                scopes=state.scopes,
                approval=state.approval,
            )
        )
        observations = state.observations + (outcome,)
        events = state.events + (
            _event(
                "execute_tool",
                "tool_called" if outcome.status == "ok" else "failed",
                f"{outcome.tool_name} -> {outcome.status}",
            ),
        )

        if outcome.status == "ok":
            # Back to the planner, not straight to an answer: the model has to
            # see what the call returned before deciding the turn is done.
            return advance(
                state,
                "think",
                mutating=state.tool_mutating,
                observations=observations,
                events=events,
            )

        if outcome.status == "rate_limited":
            return self._throttled(state, outcome, observations, events)

        if outcome.status == "approval_required":
            return advance(
                state,
                "request_approval",
                mutating=state.tool_mutating,
                approval="pending",
                observations=observations,
                events=events
                + (
                    _event(
                        "execute_tool",
                        "approval_requested",
                        f"{outcome.tool_name} was escalated by policy",
                    ),
                ),
            )

        return advance(
            state,
            "fail",
            mutating=state.tool_mutating,
            observations=observations,
            failure="tool_failure",
            error_detail=_clip(
                f"{outcome.tool_name} -> {outcome.status}: {outcome.error or ''}",
                ERROR_DETAIL_MAX_CHARS,
            ),
            events=events,
        )

    def _throttled(
        self,
        state: AgentState,
        outcome: ToolOutcome,
        observations: tuple[ToolOutcome, ...],
        events: tuple[TraceEvent, ...],
    ) -> AgentState:
        """Wait out a spent budget once, or report it and stop.

        No route change on the retry path. The state comes back sitting on
        ``call_tool`` with one more retry spent, and the engine runs this node
        again -- so the attempt is inside the step budget, appears in the trace
        as its own step, and needs no edge that would also permit a loop nobody
        intended.
        """
        wait = outcome.retry_after_seconds or 0.0
        events = events + (
            _event(
                "execute_tool",
                "rate_limited",
                f"{outcome.tool_name} throttled, {wait:g}s to wait",
            ),
        )

        if state.retry_count >= self.tool_retry_budget:
            return advance(
                state,
                "fail",
                mutating=state.tool_mutating,
                observations=observations,
                failure="tool_failure",
                error_detail=_clip(
                    f"{outcome.tool_name} stayed rate limited after "
                    f"{state.retry_count} retr"
                    f"{'y' if state.retry_count == 1 else 'ies'}",
                    ERROR_DETAIL_MAX_CHARS,
                ),
                events=events,
            )

        self.sleep(wait)
        return state.evolve(
            observations=observations,
            retry_count=state.retry_count + 1,
            events=events
            + (
                _event(
                    "execute_tool",
                    "retry_scheduled",
                    f"attempt {state.retry_count + 2} of "
                    f"{self.tool_retry_budget + 1}",
                ),
            ),
        )


def _observed_sources(observations: Sequence[ToolOutcome]) -> list[str]:
    """Every identifier this turn's calls actually touched, in order, deduped."""
    seen: list[str] = []
    for outcome in observations:
        for source_id in outcome.source_ids:
            if source_id not in seen:
                seen.append(source_id)
    return seen


def _ungrounded(answer, snippets: Sequence[EvidenceSnippet]) -> str | None:
    """Say why this answer must not be delivered as fact, or ``None``.

    Three checks, and the third is the one that matters. A composer is trusted
    to write and is not trusted to have cited something real, so every citation
    is matched against the passages actually retrieved. A model that cites a
    plausible source id it never saw produces an answer that looks grounded and
    is not, and no downstream reader can tell.
    """
    if not answer.grounded:
        return f"the composer refused: {answer.refusal_reason or 'no reason given'}"
    if not answer.citations:
        return "the answer claimed grounding and cited nothing"

    retrieved = {snippet.source_id for snippet in snippets}
    invented = sorted(
        {cite.source_id for cite in answer.citations} - retrieved
    )
    if invented:
        return f"cited sources that were never retrieved: {', '.join(invented)}"
    return None
