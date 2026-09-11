"""A turn is only fit to become another turn's history if it says who it was
for, what was asked, what was answered, and whether it is even settled yet."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn

STARTED = datetime(2026, 9, 8, 9, 0, tzinfo=UTC)
FINISHED = datetime(2026, 9, 8, 9, 0, 5, tzinfo=UTC)


def turn(**overrides: object) -> ConversationTurn:
    fields: dict[str, object] = {
        "trace_id": "run-1",
        "session_id": "sess-1",
        "actor": "bao",
        "request": "How is M2 tracking?",
        "response": "On track.",
        "route": "answer",
        "started_at": STARTED,
        "finished_at": FINISHED,
    }
    fields.update(overrides)
    return ConversationTurn(**fields)  # type: ignore[arg-type]


def answered_state(**overrides: object) -> AgentState:
    fields: dict[str, object] = {
        "request": "How is M2 tracking?",
        "actor": "bao",
        "trace_id": "run-1",
        "session_id": "sess-1",
        "route": "answer",
        "response": "On track.",
        "terminal": True,
    }
    fields.update(overrides)
    return AgentState(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# from_state
# --------------------------------------------------------------------------


def test_from_state_copies_the_fields_the_prompt_needs() -> None:
    state = answered_state()

    projected = ConversationTurn.from_state(
        state, started_at=STARTED, finished_at=FINISHED
    )

    assert projected.trace_id == "run-1"
    assert projected.session_id == "sess-1"
    assert projected.actor == "bao"
    assert projected.request == "How is M2 tracking?"
    assert projected.response == "On track."
    assert projected.route == "answer"
    assert projected.failure == "none"
    assert projected.approval == "not_required"
    assert projected.started_at == STARTED
    assert projected.finished_at == FINISHED


def test_from_state_refuses_a_turn_with_no_session() -> None:
    state = answered_state(session_id=None)

    with pytest.raises(ValueError, match="session_id"):
        ConversationTurn.from_state(state, started_at=STARTED, finished_at=FINISHED)


def test_from_state_carries_a_failure_with_no_reply() -> None:
    state = answered_state(
        route="refuse", response=None, failure="insufficient_evidence"
    )

    projected = ConversationTurn.from_state(
        state, started_at=STARTED, finished_at=FINISHED
    )

    assert projected.response is None
    assert projected.failure == "insufficient_evidence"


def test_from_state_carries_a_paused_turn() -> None:
    state = answered_state(
        route="request_approval",
        response=None,
        terminal=False,
        tool_name="create_risk",
        tool_arguments={"severity": "high"},
        approval="pending",
    )

    projected = ConversationTurn.from_state(
        state, started_at=STARTED, finished_at=FINISHED
    )

    assert projected.paused is True
    assert projected.tool_name == "create_risk"
    assert projected.approval == "pending"


# --------------------------------------------------------------------------
# paused / settled
# --------------------------------------------------------------------------


def test_a_paused_turn_knows_it_is_paused() -> None:
    waiting = turn(
        route="request_approval",
        response=None,
        approval="pending",
        tool_name="create_risk",
    )

    assert waiting.paused is True
    assert waiting.settled is False


def test_a_settled_turn_knows_it_is_settled() -> None:
    assert turn().paused is False
    assert turn().settled is True


def test_a_paused_turn_must_name_its_tool() -> None:
    with pytest.raises(ValidationError, match="tool_name"):
        turn(route="request_approval", response=None, approval="pending")


# --------------------------------------------------------------------------
# invariants and serialization
# --------------------------------------------------------------------------


def test_finished_at_cannot_precede_started_at() -> None:
    with pytest.raises(ValidationError, match="finished_at"):
        turn(started_at=FINISHED, finished_at=STARTED)


def test_round_trips_through_json() -> None:
    original = turn()

    assert ConversationTurn.model_validate(original.model_dump(mode="json")) == original


def test_an_unmodelled_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        turn(unexpected="value")
