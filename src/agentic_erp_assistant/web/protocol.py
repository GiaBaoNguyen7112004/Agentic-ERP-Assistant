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
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from agentic_erp_assistant.engine.nodes import SOURCES_PREFIX
from agentic_erp_assistant.reasoning.decision import DecisionRoute, FailureMode
from agentic_erp_assistant.state.agent_state import AgentState, ApprovalDecision
from agentic_erp_assistant.state.tool_outcome import ToolStatus

__all__ = [
    "AnswerEvent",
    "ApprovalRequiredEvent",
    "CitationOut",
    "encode_sse",
    "ErrorEvent",
    "EVENT_TYPES",
    "ModelCallTotalsOut",
    "ObservationOut",
    "parse_citations",
    "ResetEvent",
    "ServerEvent",
    "StepEvent",
    "TokenEvent",
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


class TurnStartedEvent(_Event):
    """Sent once, before the engine runs (or resumes)."""

    type: Literal["turn_started"] = "turn_started"
    trace_id: str
    session_id: str | None
    actor: str
    resumed: bool
    """``True`` on an approval decision's stream, ``False`` on a fresh chat."""


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


class StepEvent(_Event):
    """Sent after every node execution -- the engine's own observer hook.

    Named ``step`` rather than ``state`` on the wire: it is a projection of
    the state a screen needs, not the state itself, which is neither small
    nor stable across engine changes.
    """

    type: Literal["step"] = "step"
    route: DecisionRoute | None
    tool_name: str | None
    tool_arguments: dict[str, object] | None
    tool_mutating: bool | None
    approval: ApprovalDecision
    step_count: int
    terminal: bool


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


class ErrorEvent(_Event):
    """Any exception that escaped the service. Also logged server-side."""

    type: Literal["error"] = "error"
    message: str


ServerEvent = Annotated[
    Union[
        TurnStartedEvent,
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
