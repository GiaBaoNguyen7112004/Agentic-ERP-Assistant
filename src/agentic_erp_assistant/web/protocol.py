"""The typed event vocabulary the server and the browser agree on.

Nine event types, each a frozen, ``extra="forbid"`` pydantic model with its
own ``type`` literal, discriminated by that field into :data:`ServerEvent`.
This is the one contract both sides of the wire are built against:
``ui/src/protocol.ts`` mirrors it field for field, and
``tests/web/test_protocol_drift.py`` (Phase G6) fails the Python suite the
day the two disagree, rather than a browser console discovering it.

Nothing here decides *when* an event fires -- that is
:mod:`agentic_erp_assistant.web.stream` and
:mod:`agentic_erp_assistant.web.service`. This module only says what an
event may contain.
"""

import re
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from agentic_erp_assistant.engine.nodes import SOURCES_PREFIX
from agentic_erp_assistant.llm.telemetry import ModelCallRecord
from agentic_erp_assistant.memory.audit import MemoryAuditRow
from agentic_erp_assistant.memory.models import MemoryDecisionKind, RejectionReason
from agentic_erp_assistant.persistence.postgres_queries import ModelCallTotals
from agentic_erp_assistant.reasoning.decision import DecisionRoute, FailureMode
from agentic_erp_assistant.state.agent_state import AgentState, ApprovalDecision
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.memory import MemoryKind
from agentic_erp_assistant.state.reply_contract import ReplyNeed
from agentic_erp_assistant.state.tool_outcome import ToolStatus
from agentic_erp_assistant.tools.models import AuditRow

__all__ = [
    "AnswerEvent",
    "AnswerOut",
    "ApprovalRequiredEvent",
    "CitationOut",
    "clip_text",
    "ContextEvent",
    "ContractOut",
    "encode_sse",
    "ErrorEvent",
    "EVENT_TYPES",
    "EvidenceOut",
    "HistoryTurnOut",
    "memory_audit_out",
    "MemoryAuditOut",
    "MemoryOut",
    "MessageOut",
    "model_call_totals_out",
    "ModelCallOut",
    "ModelCallTotalsOut",
    "ModelRequestOut",
    "ModelResponseOut",
    "ObservationOut",
    "parse_citations",
    "ResetEvent",
    "RetrievalHitOut",
    "RetrievalOut",
    "RunOut",
    "RunReportOut",
    "ServerEvent",
    "StepEvent",
    "TEXT_MAX_CHARS",
    "TextOut",
    "TokenEvent",
    "ToolOutcomeOut",
    "TraceRow",
    "TurnFinishedEvent",
    "TurnStartedEvent",
]


class _Event(BaseModel):
    """Frozen, closed configuration shared by every event this module declares.

    ``extra="forbid"`` for the reason every wire model in this project uses
    it: a field smuggled onto one event is a field the browser's typed union
    never saw coming.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


TEXT_MAX_CHARS = 4_000
"""How much free text one wire payload block may carry.

A developer-only inspector still needs a bound: without one, an evidence
passage, a tool's summary, or (once the model I/O toggle exists) a whole
prompt block travels to the browser at its full, unbounded size on every
turn that touches it. :class:`TextOut` and :func:`clip_text` are the one
place that bound is applied and the one place a client learns it was
applied, rather than every payload growing its own truncation rule.
"""


class TextOut(_Event):
    """One block of free text, bounded for the wire.

    ``truncated``/``chars`` say plainly what was cut, rather than silently
    handing back a shorter string a reader might mistake for the whole
    thing.
    """

    text: str
    truncated: bool
    chars: int
    """The original length, in characters -- what ``truncated`` refers to,
    kept even when nothing was cut so a reader never has to infer it."""


def clip_text(text: str, limit: int = TEXT_MAX_CHARS) -> TextOut:
    """Bound one piece of text to ``limit`` characters, wire-ready."""
    if len(text) <= limit:
        return TextOut(text=text, truncated=False, chars=len(text))
    return TextOut(text=text[:limit], truncated=True, chars=len(text))


class HistoryTurnOut(_Event):
    """One of the session's recent turns, as this turn was shown it --
    :class:`~agentic_erp_assistant.state.conversation.ConversationTurn`,
    already clipped and tag-stripped by the time it reaches here."""

    trace_id: str
    request: str
    response: str | None
    route: DecisionRoute | None
    failure: FailureMode
    tool_name: str | None
    approval: ApprovalDecision
    started_at: datetime
    finished_at: datetime


class MemoryOut(_Event):
    """One durable memory recall selected for this turn --
    :class:`~agentic_erp_assistant.state.memory.MemoryRecord`, minus nothing
    it carries."""

    memory_id: str
    kind: MemoryKind
    key: str
    statement: str
    confidence: float
    recorded_in_run: str
    recorded_at: datetime
    supersedes: tuple[str, ...]
    links: tuple[str, ...]
    required_scope: str
    actor: str
    project_code: str
    session_id: str


class ContractOut(_Event):
    """What the planner declared this turn's reply must rest on (ADR 0021)."""

    needs: tuple[ReplyNeed, ...]
    document_query: str | None


class EvidenceOut(_Event):
    """One retrieved passage -- :class:`~agentic_erp_assistant.state.evidence.
    EvidenceSnippet`, with its text bounded for the wire."""

    source_id: str
    locator: str
    tag: str
    """The exact citation tag, e.g. ``"[doc-12#3.2]"``."""
    text: TextOut


class ToolOutcomeOut(_Event):
    """What one tool call produced -- :class:`~agentic_erp_assistant.state.
    tool_outcome.ToolOutcome`, with its summary bounded for the wire."""

    tool_name: str
    arguments_summary: str
    status: ToolStatus
    summary: TextOut
    source_ids: tuple[str, ...]
    error: str | None
    attempts: int
    retry_after_seconds: float | None


class MemoryAuditOut(_Event):
    """One decision consolidation made about a piece of would-be memory --
    :class:`~agentic_erp_assistant.memory.audit.MemoryAuditRow`, the columns
    a screen (not a database join) needs."""

    occurred_at: datetime
    memory_id: str
    kind: MemoryKind
    decision: MemoryDecisionKind
    rejection: RejectionReason | None
    reason: str
    statement_summary: str


class MessageOut(_Event):
    """One role block of a model request -- the port's own
    :class:`~agentic_erp_assistant.llm.ports.Role`, before an adapter folds
    it onto the wire's narrower vocabulary."""

    role: str
    content: TextOut


class ModelRequestOut(_Event):
    """What one model call sent -- live-only, and present only when
    ``DEV_TRACE_MODEL_IO`` is on; see :class:`ModelCallOut`."""

    kind: Literal["answer", "tools"]
    """``"answer"`` for a structured-output call (no tools offered);
    ``"tools"`` for a function-calling call (routing, a declaration, a
    memory proposal)."""

    messages: tuple[MessageOut, ...]
    tools: tuple[str, ...]
    """The offered tools' names, empty for ``kind="answer"``."""

    tool_choice: str | None
    """The wire's own ``tool_choice`` value, or ``None`` for ``kind="answer"``,
    which has none."""

    temperature: float


class ModelResponseOut(_Event):
    """What one model call received back -- content, or a tool call, never
    both. Live-only, present only when ``DEV_TRACE_MODEL_IO`` is on."""

    content: TextOut | None
    tool_name: str | None
    arguments: dict[str, object] | None
    stop_reason: str | None


class RetrievalHitOut(_Event):
    """One chunk a search returned, with the ranks and scores that put it
    there -- :class:`~agentic_erp_assistant.rag.fusion.FusedHit`, live-only:
    the port ``engine/`` depends on deliberately returns none of this (see
    that port's own docstring)."""

    chunk_id: str
    document_id: str
    locator: str
    title: str
    score: float
    """The fused (reciprocal-rank) score. Comparable within this search
    only -- see :attr:`~agentic_erp_assistant.rag.fusion.FusedHit.score`."""
    ranks: dict[str, int]
    scores: dict[str, float]


class RetrievalOut(_Event):
    """One search's diagnostics -- :class:`~agentic_erp_assistant.rag.
    retriever.RetrievalOutcome`, live-only, from a composition-level
    wrapper around the retriever the port itself carries none of."""

    query: str
    limit: int
    hits: tuple[RetrievalHitOut, ...]
    best_similarity: float | None
    """The highest cosine the dense half saw before the floor was applied,
    or ``None`` only when it returned nothing at all."""
    minimum_similarity: float
    dense_candidates: int
    lexical_candidates: int
    gated: bool
    """Whether the similarity floor is what made this search come back
    empty."""


class ModelCallOut(_Event):
    """One model call's cost, latency and outcome --
    :class:`~agentic_erp_assistant.llm.telemetry.ModelCallRecord`, as filed.
    """

    model: str
    outcome: str
    estimated_input_tokens: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_seconds: float
    attempts: int
    occurred_at: datetime
    detail: str | None
    request: ModelRequestOut | None = None
    """The prompt this call sent -- ``None`` unless ``DEV_TRACE_MODEL_IO``
    was on for this turn, live-only, and never what
    :meth:`~agentic_erp_assistant.persistence.postgres_queries.
    EvidenceQueries.model_calls` returns (nothing this rich is ever filed;
    see the toggle's own doc in ``composition/settings.py``)."""
    response: ModelResponseOut | None = None
    """The reply this call received -- the same rule as :attr:`request`."""


class TurnStartedEvent(_Event):
    """Sent once, before the engine runs (or resumes)."""

    type: Literal["turn_started"] = "turn_started"
    trace_id: str
    session_id: str | None
    actor: str
    resumed: bool
    """``True`` on an approval decision's stream, ``False`` on a fresh chat."""


class ContextEvent(_Event):
    """Sent once, right after :class:`TurnStartedEvent` and before the first
    node runs -- what the orchestrator attached to the state before handing
    it to the engine (or, on a resume, what the paused state already
    carried): the session's recent turns, recalled memory, and the declared
    reply contract. Nothing here is re-derived from the trace rows a client
    already has; ``history_recalled``/``memory_recalled``/
    ``contract_declared`` still stream as :class:`TraceRow` rows exactly as
    before, folded into the first node's own batch (see ``engine/
    orchestrator.py``) -- this event exists because those rows say *that*
    something was recalled, never *what*.
    """

    type: Literal["context"] = "context"
    request: str
    history: tuple[HistoryTurnOut, ...]
    memories: tuple[MemoryOut, ...]
    contract: ContractOut | None
    """``None`` means this turn was never checked (ADR 0021) -- a replay, a
    hand-built state, or a declarer that raised outright."""
    model_calls: tuple[ModelCallOut, ...]
    """Model calls made before the first node ran -- the reply contract's
    own declaration call. Ordering, not the engine's: the declaration
    happens before this event is even built (see ``engine/orchestrator.py``
    -- recall, then declare, then the engine), so a call made here would
    otherwise have nowhere on the wire to be attributed to at all."""
    state: AgentState
    """The state the engine is about to run from -- history, memories and
    contract already attached, no node executed yet (or, on a resume, the
    paused state exactly as the pause store returned it). The baseline
    every later :class:`StepEvent.state` is read against on the client, and
    the same object :class:`~agentic_erp_assistant.web.stream.TurnStream`
    seeds ``_last`` with (see ``docs/agent-state-inspector-plan.md``)."""


class TraceRow(_Event):
    """One line of the trace, engine or gateway.

    ``seq`` is the row's position in ``AgentState.events`` for an engine
    row -- stable, so the UI can de-duplicate a reconnect -- and ``None``
    for a gateway-internal row (a retry, an approval record), which is
    never stored on the state at all; see
    :attr:`~agentic_erp_assistant.tools.gateway.ToolGateway.on_event`.
    """

    type: Literal["trace"] = "trace"
    seq: int | None
    node: str
    kind: str
    detail: str
    source: Literal["engine", "tool_gateway"]
    step: int | None
    """Which node execution was under way when this row was produced --
    ``last_observed.step_count + 1``, so a live gateway row (pushed to the
    stream the instant it happens, mid-node, ahead of that node's own rows
    -- see ``web/stream.py::TurnStream.trace_event``) still names the span
    it belongs to instead of leaving a client to infer it from arrival
    order. ``None`` for an engine row: its own position in ``events`` under
    ``seq`` already places it inside the ``node_entered``/``node_exited``
    span a client is grouping rows into."""


class StepEvent(_Event):
    """Sent after every node execution -- the engine's own observer hook.

    Named ``step`` rather than ``state`` on the wire for historical reasons
    (the earlier, delta-only shape this event started as); it now carries
    both: the delta fields below, kept because the existing blocks read them
    cheaply, and :attr:`state` itself -- the whole object the engine's
    observer was handed for this node, so a reader never has to sum earlier
    deltas to know what a node actually saw or produced (``docs/
    agent-state-inspector-plan.md`` D1).
    """

    type: Literal["step"] = "step"
    route: DecisionRoute | None
    tool_name: str | None
    tool_arguments: dict[str, object] | None
    tool_mutating: bool | None
    approval: ApprovalDecision
    step_count: int
    terminal: bool
    node: str | None
    """Which node execution this step closes -- the ``node_entered`` row's
    own ``node`` among the rows just streamed with it, or ``None`` for an
    engine-level state change with no node span of its own (a decision
    recorded on resume, a denial's refusal, the loop guard)."""
    elapsed_ms: float
    """Wall clock since the previous observer call (or since
    :class:`ContextEvent` for the first one) -- an approximation, not a
    node's true execution time: it also carries whatever the engine's own
    bookkeeping cost between calls. ``docs/trace-inspector-plan.md`` §13
    names this explicitly rather than promising precision it cannot keep."""
    evidence: tuple[EvidenceOut, ...] | None
    """The passages retrieval supplied, sent once -- on the step that set
    them -- and ``None`` on every other step. Evidence is assigned exactly
    once per turn (``engine/nodes.py::retrieve_and_answer``), so a client
    never has to reconcile two different evidence sets for one turn."""
    observations: tuple[ToolOutcomeOut, ...]
    """What this step's own node added to ``AgentState.observations``,
    never the turn's full accumulated list -- a reason-act cycle can call
    several tools, and resending every earlier outcome on each new step
    would repeat payloads a client already has."""
    response: str | None
    failure: FailureMode
    error_detail: str | None
    draft: str | None
    redirected_needs: tuple[ReplyNeed, ...]
    retry_count: int
    retrieval: RetrievalOut | None
    """This step's own search diagnostics, from the composition-level
    retriever wrapper -- ``None`` on every step but the one that searched."""
    model_calls: tuple[ModelCallOut, ...]
    """Model calls made during this node's own execution -- a planner
    decision, a composer call. Empty on a step that made none."""
    state: AgentState
    """The whole state this node returned -- the exact object the engine's
    observer was handed, events included. The delta fields above are
    projections of this one for the blocks that already read them; this is
    the record itself, sent whole on every step (``docs/
    agent-state-inspector-plan.md`` D1/D2)."""


class TokenEvent(_Event):
    """One fragment of reply text, in order. A preview -- see
    :class:`AnswerEvent` for the authoritative reading (ADR 0015 / D4)."""

    type: Literal["token"] = "token"
    text: str


class ResetEvent(_Event):
    """The gateway retried the streamed call; the client must discard
    whatever :class:`TokenEvent` text it has shown for this attempt."""

    type: Literal["reset"] = "reset"


class ApprovalRequiredEvent(_Event):
    """The run paused. Everything an approver's card needs to decide."""

    type: Literal["approval_required"] = "approval_required"
    trace_id: str
    tool_name: str
    arguments: dict[str, object]
    summary: str
    actor: str
    """Who asked -- the requester, not the approver about to decide."""


class ObservationOut(_Event):
    """One tool call this turn made, as a summary line for the trace panel."""

    tool: str
    status: ToolStatus
    attempts: int


class CitationOut(_Event):
    """One resolvable reference out of the answer's ``Sources:`` trailer.

    ``kind`` distinguishes what the chip does on click: a ``"document"``
    opens ``/api/documents/{source_id}``; an ``"erp"`` chip is a plain
    label -- the mock ERP has no document to open.
    """

    source_id: str
    locator: str | None
    tag: str
    """The trailer item verbatim, e.g. ``"[m2-status.md#p.2]"`` or ``"risk-r-3"``."""
    kind: Literal["document", "erp"]


class AnswerEvent(_Event):
    """The run ended. Authoritative: replaces whatever :class:`TokenEvent`
    text the client accumulated (D4)."""

    type: Literal["answer"] = "answer"
    text: str
    route: DecisionRoute | None
    failure: FailureMode
    error_detail: str | None
    citations: tuple[CitationOut, ...]


class ModelCallTotalsOut(_Event):
    count: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    unpriced: int
    records: tuple[ModelCallOut, ...]
    """Every model call this run made, in the order they were recorded --
    the same rows :meth:`~agentic_erp_assistant.persistence.
    postgres_queries.EvidenceQueries.model_calls` returns, already filed by
    the time :class:`TurnFinishedEvent` is sent."""


class TurnFinishedEvent(_Event):
    """After the run was filed -- the one event a client can wait for to
    know the trace is durable, whatever the outcome."""

    type: Literal["turn_finished"] = "turn_finished"
    outcome: Literal["paused", "terminal"]
    route: DecisionRoute | None
    failure: FailureMode
    step_count: int
    evidence: tuple[str, ...]
    """Document ids actually retrieved this turn, deduplicated."""
    memories_recalled: int
    history_shown: int
    observations: tuple[ObservationOut, ...]
    model_calls: ModelCallTotalsOut
    memory_audit: tuple[MemoryAuditOut, ...]
    """Every memory decision consolidation made while filing this run,
    rejections included -- the same rows
    :meth:`~agentic_erp_assistant.persistence.postgres_queries.
    EvidenceQueries.memory_audit` returns."""
    started_at: datetime
    finished_at: datetime


class ErrorEvent(_Event):
    """Any exception that escaped the service. Also logged server-side."""

    type: Literal["error"] = "error"
    message: str


ServerEvent = Annotated[
    Union[
        TurnStartedEvent,
        ContextEvent,
        TraceRow,
        StepEvent,
        TokenEvent,
        ResetEvent,
        ApprovalRequiredEvent,
        AnswerEvent,
        TurnFinishedEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]

EVENT_TYPES: tuple[str, ...] = (
    "turn_started",
    "context",
    "trace",
    "step",
    "token",
    "reset",
    "approval_required",
    "answer",
    "turn_finished",
    "error",
)
"""Every member of :data:`ServerEvent`'s discriminator, in declaration order.

Exported so ``tests/web/test_protocol_drift.py`` can assert
``ui/src/protocol.ts`` names exactly these and no others -- a renamed event
fails in ``pytest``, not in a browser console.
"""


def encode_sse(event: BaseModel) -> bytes:
    """One event, wire-ready: ``event: <type>\\ndata: <json>\\n\\n``."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n".encode(  # type: ignore[attr-defined]
        "utf-8"
    )


_CITATION_ITEM = re.compile(r"^\[([^\[\]#]+)#([^\[\]]+)\]$")
"""A document citation tag as engine/nodes.py renders it -- see
``[f"[{cite.source_id}#{cite.locator}]" for cite in answer.citations]`` in
``retrieve_and_answer``. Not the same pattern
``context/history_injection._CITATION_TAG`` strips: this one *parses* rather
than discards, and is anchored end to end since it is matched against one
already-split trailer item, never against a whole reply."""


def parse_citations(
    response: str | None, state: AgentState
) -> tuple[str | None, tuple[CitationOut, ...]]:
    """Split the ``Sources:`` trailer :func:`engine.nodes._with_sources`
    wrote off the reply text, and parse it into typed citations.

    Locators may contain spaces (``"row R-2"``, ``"§3.2 part 1"``), so the
    trailer's items are split on ``", "`` rather than on whitespace -- the
    same reason the trailer is built with that exact separator.

    Args:
        response: ``AgentState.response`` -- may be ``None`` (a paused
            turn), or carry no trailer at all (``clarify``, ``refuse``,
            an escalated-read answer with nothing retrieved).
        state: The finished state, for ``state.evidence`` -- a document
            citation is only trusted as one when its id is among the
            passages this turn actually retrieved; anything else (or a
            bare identifier with no ``#``) is an ERP record.

    Returns:
        The reply text with the trailer removed (or ``response`` unchanged
        if there was none), and the parsed citations, in trailer order.
    """
    if not response:
        return response, ()

    marker = f"\n\n{SOURCES_PREFIX}"
    if marker not in response:
        return response, ()

    body, _, trailer = response.partition(marker)
    evidence_ids = {snippet.source_id for snippet in state.evidence}

    citations: list[CitationOut] = []
    for raw in trailer.split(", "):
        item = raw.strip()
        if not item:
            continue
        match = _CITATION_ITEM.match(item)
        if match and match.group(1) in evidence_ids:
            citations.append(
                CitationOut(
                    source_id=match.group(1),
                    locator=match.group(2),
                    tag=item,
                    kind="document",
                )
            )
        else:
            citations.append(
                CitationOut(source_id=item, locator=None, tag=item, kind="erp")
            )

    return body, tuple(citations)


# --------------------------------------------------------------------------
# GET /api/runs/{trace_id} -- a typed read model over a filed run, for
# Phase 4's hydration (docs/trace-inspector-plan.md): the same evidence a
# live turn already streamed, reconstructed for a client that never saw it
# live (a session loaded from history, an approval decided from the queue).
# Not part of ServerEvent/EVENT_TYPES -- this is a REST response body, never
# an SSE frame.
# --------------------------------------------------------------------------


def model_call_totals_out(
    records: Sequence[ModelCallRecord], totals: ModelCallTotals
) -> ModelCallTotalsOut:
    """The one place ``ModelCallRecord`` rows become the wire's
    ``ModelCallTotalsOut`` -- shared by the live tail
    (``web/service.py::_emit_tail``) and this module's own
    :class:`RunReportOut`, so a run's totals read the same way whether they
    arrived live or were read back afterward."""
    return ModelCallTotalsOut(
        count=totals.count,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        cost_usd=totals.cost_usd,
        unpriced=totals.unpriced,
        records=tuple(
            ModelCallOut(
                model=record.model,
                outcome=record.outcome,
                estimated_input_tokens=record.estimated_input_tokens,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cost_usd=record.cost_usd,
                latency_seconds=record.latency_seconds,
                attempts=record.attempts,
                occurred_at=record.occurred_at,
                detail=record.detail,
            )
            for record in records
        ),
    )


def memory_audit_out(rows: Sequence[MemoryAuditRow]) -> tuple[MemoryAuditOut, ...]:
    """The same conversion :attr:`model_call_totals_out` is -- shared by the
    live tail and :class:`RunReportOut`."""
    return tuple(
        MemoryAuditOut(
            occurred_at=row.occurred_at,
            memory_id=row.memory_id,
            kind=row.kind,
            decision=row.decision,
            rejection=row.rejection,
            reason=row.reason,
            statement_summary=row.statement_summary,
        )
        for row in rows
    )


class RunOut(_Event):
    """A run's own columns, plus its revalidated state --
    :class:`~agentic_erp_assistant.persistence.postgres_queries.RunRow`, as
    filed."""

    trace_id: str
    actor: str
    project_code: str | None
    outcome: str
    started_at: datetime
    finished_at: datetime
    state: AgentState


class AnswerOut(_Event):
    """The reply this run ended with, and its parsed citations -- the same
    reading :func:`parse_citations` gives a live turn's :class:`AnswerEvent`,
    so a hydrated turn's citation chips work identically to a live one's."""

    text: str
    citations: tuple[CitationOut, ...]


class RunReportOut(_Event):
    """Everything ``GET /api/runs/{trace_id}`` returns -- the same evidence
    :class:`~agentic_erp_assistant.persistence.postgres_queries.
    EvidenceQueries` reads, typed once so ``ui/src/protocol.ts`` has a fixed
    shape to hydrate a filed run's execution tree from, rather than the
    untyped dict this endpoint returned before Phase 4."""

    run: RunOut
    events: tuple[TraceEvent, ...]
    audit_rows: tuple[AuditRow, ...]
    model_calls: ModelCallTotalsOut
    memory_audit: tuple[MemoryAuditOut, ...]
    answer: AnswerOut
