"""The conversation store, against the real Postgres it exists for.

The same eight cases ``tests/memory/test_conversation.py`` proves against a
dict, proven again one layer down -- because that is the deal the port makes:
the fake is the contract in a test, the adapter is the contract in a
database, and only one of them can be wrong without the other noticing.

    docker compose up -d postgres
    uv run python scripts/init_postgres.py
    uv run pytest -m postgres
"""

from datetime import UTC, datetime, timedelta

import pytest

from tests.memory.builders import RECORDED, make_turn

from agentic_erp_assistant.memory.conversation import ConversationStorePort
from agentic_erp_assistant.persistence import PostgresConversationStore

pytestmark = pytest.mark.postgres


@pytest.fixture
def store_connection(database):
    """A database with no turns in it, so a test's rows are its own."""
    database.execute("TRUNCATE session_turns CASCADE")
    return database


@pytest.fixture
def store(store_connection) -> PostgresConversationStore:
    return PostgresConversationStore(store_connection)


def test_the_adapter_satisfies_its_port_without_inheriting_from_it(store) -> None:
    assert isinstance(store, ConversationStorePort)


def test_append_is_idempotent_and_keeps_the_first_started_at(store) -> None:
    earlier = RECORDED
    later = RECORDED + timedelta(minutes=5)

    store.append(make_turn(trace_id="run-1", started_at=earlier, finished_at=earlier))
    store.append(make_turn(trace_id="run-1", started_at=later, finished_at=later))

    kept = store.recent("sess-1", actor="priya", limit=10)
    assert len(kept) == 1
    assert kept[0].started_at == earlier


def test_append_replaces_the_reply_when_a_paused_turn_resumes(store) -> None:
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


def test_recent_is_oldest_first_and_bounded(store) -> None:
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


def test_another_actor_in_the_same_session_sees_nothing(store) -> None:
    store.append(make_turn(trace_id="run-1", actor="priya"))

    assert store.recent("sess-1", actor="bao", limit=10) == ()


def test_evicted_is_what_lies_beyond_keep_and_is_not_promoted(store) -> None:
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


def test_a_paused_turn_is_never_evicted(store) -> None:
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


def test_mark_promoted_hides_turns_from_evicted_but_not_from_recent(store) -> None:
    store.append(make_turn(trace_id="run-0", started_at=RECORDED, finished_at=RECORDED))

    store.mark_promoted(["run-0"], run="run-5")

    assert store.evicted("sess-1", actor="priya", keep=0, limit=10) == ()
    assert [t.trace_id for t in store.recent("sess-1", actor="priya", limit=10)] == ["run-0"]


def test_evicted_is_capped_by_limit_oldest_first(store) -> None:
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


def test_mark_promoted_ignores_an_unknown_trace_id(store) -> None:
    store.mark_promoted(["run-does-not-exist"], run="run-5")  # must not raise
