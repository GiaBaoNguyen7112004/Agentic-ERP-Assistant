"""The trace is the evidence, so a log entry has to be as typed as the state."""

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.state.events import EVENT_DETAIL_MAX_CHARS, TraceEvent


def test_an_event_names_the_node_and_what_happened() -> None:
    event = TraceEvent(node="retrieve", kind="evidence_retrieved", detail="4 chunks")

    assert (event.node, event.kind) == ("retrieve", "evidence_retrieved")


def test_the_detail_is_optional() -> None:
    """Most transitions say everything in their kind."""
    assert TraceEvent(node="route", kind="node_entered").detail == ""


def test_a_kind_nobody_reports_on_is_rejected() -> None:
    """Closed set: a free string would let each node invent its own vocabulary,
    and a report grouping by kind would be counting spellings."""
    with pytest.raises(ValidationError, match="kind"):
        TraceEvent(node="route", kind="something_happened")


def test_an_event_must_say_which_node_emitted_it() -> None:
    with pytest.raises(ValidationError, match="node"):
        TraceEvent(node="", kind="node_entered")


def test_a_detail_at_the_cap_is_accepted() -> None:
    event = TraceEvent(
        node="answer", kind="failed", detail="x" * EVENT_DETAIL_MAX_CHARS
    )

    assert len(event.detail) == EVENT_DETAIL_MAX_CHARS


def test_a_detail_over_the_cap_is_rejected() -> None:
    """The cap is what stops a prompt or a full payload being dumped into the
    log and then read out of an export as reviewed reasoning."""
    with pytest.raises(ValidationError, match="detail"):
        TraceEvent(
            node="answer", kind="failed", detail="x" * (EVENT_DETAIL_MAX_CHARS + 1)
        )


def test_an_event_cannot_be_rewritten_after_the_fact() -> None:
    """A record that can be edited is not evidence."""
    event = TraceEvent(node="approve", kind="approval_recorded", detail="approved")

    with pytest.raises(ValidationError):
        event.detail = "denied"  # type: ignore[misc]


def test_an_event_carries_no_clock_of_its_own() -> None:
    """Order comes from the log's position, and wall-clock time belongs to the
    trace store that writes the run down -- so an expected event stays
    deterministic in a test."""
    assert "timestamp" not in TraceEvent.model_fields
    assert set(TraceEvent.model_fields) == {"node", "kind", "detail"}


def test_the_loop_guard_has_a_kind_of_its_own() -> None:
    """A run nobody ended must be countable apart from a node that ended one."""
    event = TraceEvent(node="engine", kind="run_failed", detail="max_steps_exceeded")

    assert event.kind == "run_failed"
