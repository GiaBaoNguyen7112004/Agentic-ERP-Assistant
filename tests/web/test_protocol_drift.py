"""The Python event vocabulary and ui/src/protocol.ts must never drift apart.

A renamed or added event type is a wire-format change two independently
typed files have to agree on; nothing at the type-checker level enforces
that agreement across a language boundary. This test is the enforcement:
it reads the TypeScript source as text and checks its `type: "..."`
literals against web/protocol.py's own EVENT_TYPES, so a mismatch fails
`pytest`, not a browser console three files away.
"""

import re
from pathlib import Path

import pytest

from agentic_erp_assistant.tools.models import AuditRow
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.state.reply_contract import ReplyContract
from agentic_erp_assistant.web.protocol import (
    AnswerEvent,
    AnswerOut,
    ApprovalRequiredEvent,
    CitationOut,
    ContextEvent,
    ContractOut,
    ErrorEvent,
    EVENT_TYPES,
    EvidenceOut,
    HistoryTurnOut,
    MemoryAuditOut,
    MemoryOut,
    ModelCallOut,
    ModelCallTotalsOut,
    ObservationOut,
    RunOut,
    RunReportOut,
    StepEvent,
    TextOut,
    ToolOutcomeOut,
    TraceRow,
    TurnFinishedEvent,
    TurnStartedEvent,
)

PROTOCOL_TS = Path(__file__).resolve().parents[2] / "ui" / "src" / "protocol.ts"


def _ts_event_types() -> set[str]:
    """Every `type: '...'` literal declared in protocol.ts's event interfaces.

    Matches ``type: 'turn_started'`` (single-quoted, TypeScript's own
    convention in this file) rather than parsing the file as TypeScript --
    this is a drift guard, not a compiler, and a regex over one narrow,
    consistently-formatted pattern is the whole job.
    """
    source = PROTOCOL_TS.read_text(encoding="utf-8")
    return set(re.findall(r"type:\s*'([a-z_]+)'", source))


def test_protocol_ts_exists() -> None:
    assert PROTOCOL_TS.exists(), (
        f"{PROTOCOL_TS} is missing -- has ui/src/protocol.ts moved or been renamed?"
    )


def test_every_python_event_type_is_named_in_protocol_ts() -> None:
    ts_types = _ts_event_types()
    missing = set(EVENT_TYPES) - ts_types
    assert not missing, f"protocol.ts is missing event type(s): {sorted(missing)}"


def test_protocol_ts_names_no_event_type_python_does_not_have() -> None:
    ts_types = _ts_event_types()
    extra = ts_types - set(EVENT_TYPES)
    assert not extra, f"protocol.ts declares event type(s) EVENT_TYPES does not: {sorted(extra)}"


def test_event_types_export_matches_the_ts_array_too() -> None:
    """EVENT_TYPES itself (the const array ui/src/protocol.ts also exports)
    must list the same members -- catches a type-only interface added
    without also being added to the runtime array a test might iterate."""
    source = PROTOCOL_TS.read_text(encoding="utf-8")
    match = re.search(r"EVENT_TYPES:.*?=\s*\[(.*?)\]\s*as const", source, re.DOTALL)
    assert match, "protocol.ts must export EVENT_TYPES as a const array"
    declared = set(re.findall(r"'([a-z_]+)'", match.group(1)))
    assert declared == set(EVENT_TYPES)


# --------------------------------------------------------------------------
# Field-level drift: a widened model must widen its TS mirror too, not just
# the event types the two checks above already cover.
# --------------------------------------------------------------------------


def _ts_interface_fields(source: str, name: str) -> set[str]:
    """Every field name declared in ``export interface {name} { ... }``.

    A regex over one field-per-line block, the same kind of narrow,
    consistently-formatted match the event-type checks above make -- this
    is a drift guard, not a TypeScript parser.
    """
    match = re.search(rf"export interface {re.escape(name)} \{{(.*?)\n\}}", source, re.DOTALL)
    assert match, f"protocol.ts has no `export interface {name}` block"
    return set(re.findall(r"^\s*([a-zA-Z_]+)\??:", match.group(1), re.MULTILINE))


# TS interface name -> the pydantic model it mirrors. Named identically on
# both sides except where the TS side dropped the "Out" suffix already (a
# pre-existing choice this test does not relitigate).
_MIRRORED_MODELS = {
    "TurnStartedEvent": TurnStartedEvent,
    "ContextEvent": ContextEvent,
    "TraceRow": TraceRow,
    "StepEvent": StepEvent,
    "ApprovalRequiredEvent": ApprovalRequiredEvent,
    "AnswerEvent": AnswerEvent,
    "TurnFinishedEvent": TurnFinishedEvent,
    "ErrorEvent": ErrorEvent,
    "Citation": CitationOut,
    "Observation": ObservationOut,
    "ModelCallTotals": ModelCallTotalsOut,
    "TextOut": TextOut,
    "HistoryTurnOut": HistoryTurnOut,
    "MemoryOut": MemoryOut,
    "ContractOut": ContractOut,
    "EvidenceOut": EvidenceOut,
    "ToolOutcomeOut": ToolOutcomeOut,
    "MemoryAuditOut": MemoryAuditOut,
    "ModelCallOut": ModelCallOut,
    # The filed-run read model (Phase 4): the state models it embeds are
    # mirrored too, so hydrating one cannot read a field the run never filed.
    "RunReport": RunReportOut,
    "RunOut": RunOut,
    "RunAnswerOut": AnswerOut,
    "AgentStateSnapshot": AgentState,
    "AuditRow": AuditRow,
    "TraceEvent": TraceEvent,
    "EvidenceSnippet": EvidenceSnippet,
    "MemoryRecord": MemoryRecord,
    "ConversationTurn": ConversationTurn,
    "ReplyContract": ReplyContract,
}


@pytest.mark.parametrize("ts_name,model", sorted(_MIRRORED_MODELS.items()))
def test_ts_interface_fields_match_the_pydantic_model(ts_name, model) -> None:
    source = PROTOCOL_TS.read_text(encoding="utf-8")
    ts_fields = _ts_interface_fields(source, ts_name)
    py_fields = set(model.model_fields)
    assert ts_fields == py_fields, (
        f"{ts_name} (TS) vs {model.__name__} (Python) disagree: "
        f"TS-only={sorted(ts_fields - py_fields)}, "
        f"Python-only={sorted(py_fields - ts_fields)}"
    )
