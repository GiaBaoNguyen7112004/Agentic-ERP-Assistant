"""A record that can disagree with the run it files is not evidence."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.trace.records import RunRecord

STARTED = datetime(2026, 9, 5, 9, 30, tzinfo=UTC)
FINISHED = datetime(2026, 9, 5, 9, 30, 2, tzinfo=UTC)


def finished_state(**changes: object) -> AgentState:
    """A state a run could plausibly end on, with every field overridable."""
    return AgentState(
        request="How is M2 tracking?",
        actor="bao",
        trace_id="run-1",
        route="answer",
        terminal=True,
        response="M2 is at risk, two days late.",
        **changes,  # type: ignore[arg-type]
    )


def paused_state() -> AgentState:
    """The state a mutating call leaves a turn in while a human decides."""
    return AgentState(
        request="Record the vendor risk.",
        actor="bao",
        trace_id="run-1",
        route="request_approval",
        tool_name="create_risk",
        tool_arguments={"project_id": "atlas", "title": "x", "severity": "low"},
        tool_mutating=True,
        approval="pending",
    )


def record(state: AgentState, **overrides: object) -> RunRecord:
    fields: dict[str, object] = {
        "trace_id": state.trace_id,
        "outcome": "paused" if state.approval == "pending" else "terminal",
        "started_at": STARTED,
        "finished_at": FINISHED,
        "state": state,
    }
    fields.update(overrides)
    return RunRecord(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# What a well-formed record carries
# --------------------------------------------------------------------------


def test_a_record_files_the_state_it_sums_up() -> None:
    filed = record(finished_state())

    assert filed.trace_id == "run-1"
    assert filed.outcome == "terminal"
    assert filed.state.terminal


# --------------------------------------------------------------------------
# The record cannot lie about the run it files
# --------------------------------------------------------------------------


def test_a_record_cannot_carry_another_runs_id() -> None:
    with pytest.raises(ValidationError, match="trace_id"):
        record(finished_state(), trace_id="run-2")


def test_an_outcome_cannot_contradict_the_state() -> None:
    """A record that files a paused state as terminal -- or the reverse --
    reads as evidence for a run that did not happen."""
    with pytest.raises(ValidationError, match="outcome"):
        record(finished_state(), outcome="paused")


def test_a_paused_state_files_as_paused() -> None:
    filed = record(paused_state())

    assert filed.outcome == "paused"


def test_a_run_cannot_finish_before_it_started() -> None:
    with pytest.raises(ValidationError, match="finished_at"):
        record(finished_state(), finished_at=STARTED, started_at=FINISHED)


def test_no_field_may_be_smuggled_in() -> None:
    """extra='forbid': a field invented at one call site would be a column no
    report knows how to read."""
    with pytest.raises(ValidationError):
        record(finished_state(), reviewer="nobody")