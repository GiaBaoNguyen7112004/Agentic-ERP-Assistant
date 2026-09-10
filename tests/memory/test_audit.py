"""The rows that matter most are the refusals, so a row that stored nothing has
to be as complete as one that stored something."""

import pytest
from pydantic import ValidationError

from tests.memory.builders import RECORDED

from agentic_erp_assistant.memory.audit import (
    STATEMENT_SUMMARY_MAX_CHARS,
    InMemoryMemoryAudit,
    MemoryAuditRow,
    MemoryAuditSink,
    summarize_statement,
)


def row(**overrides: object) -> MemoryAuditRow:
    fields: dict[str, object] = {
        "occurred_at": RECORDED,
        "trace_id": "run-1",
        "session_id": "sess-1",
        "project_code": "atlas",
        "actor": "priya",
        "memory_id": "mem-1",
        "kind": "preference",
        "decision": "write",
        "statement_summary": "Prefers replies written in Vietnamese.",
    }
    fields.update(overrides)
    return MemoryAuditRow(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The expected path
# --------------------------------------------------------------------------


def test_a_write_is_recorded_with_the_five_identifiers() -> None:
    """Without any one of them the row is a decision nobody can place."""
    recorded = row()

    assert recorded.trace_id == "run-1"
    assert recorded.session_id == "sess-1"
    assert recorded.project_code == "atlas"
    assert recorded.memory_id == "mem-1"
    assert recorded.actor == "priya"


def test_a_rejection_names_the_rule_that_refused_it() -> None:
    recorded = row(
        decision="reject",
        rejection="instruction_like",
        reason="reads as standing instruction, not a fact",
    )

    assert recorded.decision == "reject"
    assert recorded.rejection == "instruction_like"


def test_a_rejection_still_names_a_memory() -> None:
    """The whole reason memory_id is derived from content rather than generated
    at write time: a refusal row carrying a null there would be a decision about
    nothing."""
    recorded = row(decision="reject", rejection="duplicate")

    assert recorded.memory_id == "mem-1"


def test_a_forget_is_recorded_against_the_record_it_retired() -> None:
    """All four decisions occur here: forget is what an update does to what it
    replaced."""
    recorded = row(decision="forget", memory_id="mem-old")

    assert recorded.decision == "forget"


def test_a_row_is_frozen() -> None:
    with pytest.raises(ValidationError):
        row().decision = "reject"  # type: ignore[misc]


def test_an_unmodelled_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        row(severity="high")


# --------------------------------------------------------------------------
# A row that contradicts itself has no instances
# --------------------------------------------------------------------------


def test_a_refusal_must_carry_a_rejection_reason() -> None:
    with pytest.raises(ValidationError, match="rejection"):
        row(decision="reject")


def test_only_a_refusal_may_carry_one() -> None:
    with pytest.raises(ValidationError, match="rejection"):
        row(decision="write", rejection="duplicate")


@pytest.mark.parametrize(
    "field",
    ["trace_id", "session_id", "project_code", "actor", "memory_id", "statement_summary"],
)
def test_no_identifier_may_be_blank(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        row(**{field: "   "})


def test_a_quoted_statement_longer_than_the_cap_is_refused() -> None:
    """A rejection is often a rejection because the text carried something that
    should not be stored, and this table is exported as evidence."""
    with pytest.raises(ValidationError, match="statement_summary"):
        row(statement_summary="x" * (STATEMENT_SUMMARY_MAX_CHARS + 1))


# --------------------------------------------------------------------------
# Summarizing what is quoted
# --------------------------------------------------------------------------


def test_a_summary_collapses_whitespace() -> None:
    """A statement containing a newline could otherwise appear as two rows in a
    report and manufacture a decision that was never made."""
    assert summarize_statement("Prefers replies\n  in Vietnamese.") == (
        "Prefers replies in Vietnamese."
    )


def test_a_long_statement_is_clipped_to_the_cap() -> None:
    summary = summarize_statement("x" * 500)

    assert len(summary) == STATEMENT_SUMMARY_MAX_CHARS
    assert summary.endswith("...")


def test_a_short_statement_is_left_alone() -> None:
    assert summarize_statement("Prefers Vietnamese.") == "Prefers Vietnamese."


# --------------------------------------------------------------------------
# The sink
# --------------------------------------------------------------------------


def test_the_in_memory_sink_satisfies_the_port_structurally() -> None:
    assert isinstance(InMemoryMemoryAudit(), MemoryAuditSink)


def test_rows_are_kept_in_the_order_they_were_decided() -> None:
    sink = InMemoryMemoryAudit()

    sink.record(row(memory_id="mem-1"))
    sink.record(row(memory_id="mem-2", decision="reject", rejection="duplicate"))

    assert [r.memory_id for r in sink.rows] == ["mem-1", "mem-2"]
