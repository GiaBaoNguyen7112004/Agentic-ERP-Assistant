"""The short-term window: what one session's turns look like in storage, and
what a turn is shown of its own recent past."""

from datetime import UTC, datetime, timedelta

from tests.memory.builders import RECORDED, make_turn

from agentic_erp_assistant.memory.conversation import (
    ConversationMemory,
    InMemoryConversationStore,
)
from agentic_erp_assistant.state.agent_state import AgentState

NOW = datetime(2026, 9, 20, 9, 0, tzinfo=UTC)


def state(**overrides: object) -> AgentState:
    fields: dict[str, object] = {
        "request": "How is M2 tracking?",
        "actor": "priya",
        "project_code": "atlas",
        "trace_id": "run-1",
        "session_id": "sess-1",
    }
    fields.update(overrides)
    return AgentState(**fields)  # type: ignore[arg-type]


def answered_state(**overrides: object) -> AgentState:
    fields: dict[str, object] = {
        "route": "answer",
        "response": "On track.",
        "terminal": True,
    }
    fields.update(overrides)
    return state(**fields)


def bound(**overrides: object) -> ConversationMemory:
    fields: dict[str, object] = {
        "store": InMemoryConversationStore(),
        "model": "gpt-4o",
        "now": lambda: NOW,
    }
    fields.update(overrides)
    return ConversationMemory(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# InMemoryConversationStore
# --------------------------------------------------------------------------


def test_append_is_idempotent_and_keeps_the_first_started_at() -> None:
    store = InMemoryConversationStore()
    earlier = RECORDED
    later = RECORDED + timedelta(minutes=5)

    store.append(make_turn(trace_id="run-1", started_at=earlier, finished_at=earlier))
    store.append(make_turn(trace_id="run-1", started_at=later, finished_at=later))

    kept = store.recent("sess-1", actor="priya", limit=10)
    assert len(kept) == 1
    assert kept[0].started_at == earlier


def test_append_replaces_the_reply_when_a_paused_turn_resumes() -> None:
    store = InMemoryConversationStore()
    store.append(
        make_turn(
            trace_id="run-1",
            route="request_approval",
            response=None,
            approval="pending",
            tool_name="create_risk",
        )
    )
    store.append(
        make_turn(trace_id="run-1", route="answer", response="Recorded.", approval="approved")
    )

    kept = store.recent("sess-1", actor="priya", limit=10)
    assert kept[0].response == "Recorded."
    assert kept[0].approval == "approved"


def test_recent_is_oldest_first_and_bounded() -> None:
    store = InMemoryConversationStore()
    for i in range(5):
        store.append(
            make_turn(
                trace_id=f"run-{i}",
                started_at=RECORDED + timedelta(minutes=i),
                finished_at=RECORDED + timedelta(minutes=i),
            )
        )

    kept = store.recent("sess-1", actor="priya", limit=3)

    assert [t.trace_id for t in kept] == ["run-2", "run-3", "run-4"]


def test_another_actor_in_the_same_session_sees_nothing() -> None:
    store = InMemoryConversationStore()
    store.append(make_turn(trace_id="run-1", actor="priya"))

    assert store.recent("sess-1", actor="bao", limit=10) == ()


def test_evicted_is_what_lies_beyond_keep_and_is_not_promoted() -> None:
    store = InMemoryConversationStore()
    for i in range(5):
        store.append(
            make_turn(
                trace_id=f"run-{i}",
                started_at=RECORDED + timedelta(minutes=i),
                finished_at=RECORDED + timedelta(minutes=i),
            )
        )

    evicted = store.evicted("sess-1", actor="priya", keep=2, limit=10)

    assert [t.trace_id for t in evicted] == ["run-0", "run-1", "run-2"]


def test_a_paused_turn_is_never_evicted() -> None:
    store = InMemoryConversationStore()
    store.append(
        make_turn(
            trace_id="run-0",
            started_at=RECORDED,
            finished_at=RECORDED,
            route="request_approval",
            response=None,
            approval="pending",
            tool_name="create_risk",
        )
    )
    store.append(
        make_turn(
            trace_id="run-1",
            started_at=RECORDED + timedelta(minutes=1),
            finished_at=RECORDED + timedelta(minutes=1),
        )
    )

    evicted = store.evicted("sess-1", actor="priya", keep=0, limit=10)

    assert [t.trace_id for t in evicted] == ["run-1"]


def test_mark_promoted_hides_turns_from_evicted_but_not_from_recent() -> None:
    store = InMemoryConversationStore()
    store.append(make_turn(trace_id="run-0", started_at=RECORDED, finished_at=RECORDED))

    store.mark_promoted(["run-0"], run="run-5")

    assert store.evicted("sess-1", actor="priya", keep=0, limit=10) == ()
    assert [t.trace_id for t in store.recent("sess-1", actor="priya", limit=10)] == ["run-0"]


def test_evicted_is_capped_by_limit_oldest_first() -> None:
    store = InMemoryConversationStore()
    for i in range(5):
        store.append(
            make_turn(
                trace_id=f"run-{i}",
                started_at=RECORDED + timedelta(minutes=i),
                finished_at=RECORDED + timedelta(minutes=i),
            )
        )

    evicted = store.evicted("sess-1", actor="priya", keep=0, limit=2)

    assert [t.trace_id for t in evicted] == ["run-0", "run-1"]


def test_mark_promoted_ignores_an_unknown_trace_id() -> None:
    store = InMemoryConversationStore()

    store.mark_promoted(["run-does-not-exist"], run="run-5")  # must not raise


# --------------------------------------------------------------------------
# ConversationMemory
# --------------------------------------------------------------------------


def test_recall_is_the_selected_window() -> None:
    memory = bound()
    memory.store.append(make_turn(trace_id="run-0", started_at=RECORDED, finished_at=RECORDED))

    recalled = memory.recall(state())

    assert [t.trace_id for t in recalled] == ["run-0"]


def test_no_session_means_nothing_read_and_nothing_written() -> None:
    memory = bound()
    turn_state = state(session_id=None)

    assert memory.recall(turn_state) == ()

    memory.record(answered_state(session_id=None), started_at=RECORDED)
    assert memory.store.recent("sess-1", actor="priya", limit=10) == ()


def test_record_files_the_turn_with_the_clocks_given() -> None:
    memory = bound()

    memory.record(answered_state(), started_at=RECORDED)

    kept = memory.store.recent("sess-1", actor="priya", limit=10)
    assert len(kept) == 1
    assert kept[0].started_at == RECORDED
    assert kept[0].finished_at == NOW
    assert kept[0].response == "On track."


def test_evicted_keeps_turn_limit_minus_one() -> None:
    memory = bound(turn_limit=3)
    for i in range(5):
        memory.store.append(
            make_turn(
                trace_id=f"run-{i}",
                started_at=RECORDED + timedelta(minutes=i),
                finished_at=RECORDED + timedelta(minutes=i),
            )
        )

    evicted = memory.evicted(state())

    # turn_limit=3, so keep=2: the two newest of the five stay, the rest go.
    assert [t.trace_id for t in evicted] == ["run-0", "run-1", "run-2"]


def test_no_session_means_nothing_evicted() -> None:
    memory = bound()

    assert memory.evicted(state(session_id=None)) == ()


def test_promoted_marks_exactly_the_turns_given() -> None:
    memory = bound()
    memory.store.append(make_turn(trace_id="run-0", started_at=RECORDED, finished_at=RECORDED))
    memory.store.append(
        make_turn(
            trace_id="run-1",
            started_at=RECORDED + timedelta(minutes=1),
            finished_at=RECORDED + timedelta(minutes=1),
        )
    )

    memory.promoted((make_turn(trace_id="run-0"),), run="run-5")

    remaining = memory.store.evicted("sess-1", actor="priya", keep=0, limit=10)
    assert [t.trace_id for t in remaining] == ["run-1"]


def test_promoted_is_a_no_op_on_nothing() -> None:
    memory = bound()

    memory.promoted((), run="run-5")  # must not raise


def test_the_service_is_what_the_orchestrator_expects() -> None:
    """ConversationMemory satisfies the orchestrator's port structurally --
    neither side imports the other, the arrangement every port here uses."""
    from agentic_erp_assistant.engine.orchestrator import SessionHistoryPort

    assert isinstance(bound(), SessionHistoryPort)
