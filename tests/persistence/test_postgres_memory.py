"""The memory adapters, against the real Postgres they exist for.

Everything ``tests/memory/test_store.py`` proves against a dict, these prove
again one layer down -- because that is the deal the ports make: the fake is the
contract in a test, the adapter is the contract in a database, and only one of
them can be wrong without the other noticing.

Two things are proven here that the fake cannot prove at all. The
one-open-intent rule is a *partial unique index*, so it holds against two
processes rather than against one dict; and the audit's
``rejection matches decision`` rule is a CHECK constraint, so a row that
contradicts itself is refused by the database even if it never went through the
model.

    docker compose up -d postgres
    uv run python scripts/init_postgres.py
    uv run pytest -m postgres
"""

from datetime import UTC, datetime

import pytest

from agentic_erp_assistant.memory.audit import MemoryAuditRow
from agentic_erp_assistant.memory.intent import IntentState
from agentic_erp_assistant.memory.models import MemoryScope
from agentic_erp_assistant.memory.store import IntentStorePort, MemoryStorePort
from agentic_erp_assistant.persistence import (
    PostgresMemoryAudit,
    PostgresMemoryStore,
    apply_schema,
)
from agentic_erp_assistant.state.memory import MemoryRecord

pytestmark = pytest.mark.postgres

RECORDED = datetime(2026, 9, 8, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 9, 9, 0, tzinfo=UTC)


@pytest.fixture
def store_connection(database):
    """A database with no memory in it, so a test's rows are its own."""
    database.execute("TRUNCATE memories, intents, memory_audit CASCADE")
    return database


@pytest.fixture
def store(store_connection) -> PostgresMemoryStore:
    return PostgresMemoryStore(store_connection)


@pytest.fixture
def audit(store_connection) -> PostgresMemoryAudit:
    return PostgresMemoryAudit(store_connection)


def scope(**overrides: object) -> MemoryScope:
    fields: dict[str, object] = {
        "project_code": "atlas",
        "session_id": "sess-1",
        "scopes": frozenset({"project.docs.read"}),
    }
    fields.update(overrides)
    actor = fields.pop("actor", "priya")
    return MemoryScope.for_actor(actor, **fields)  # type: ignore[arg-type]


def record(**overrides: object) -> MemoryRecord:
    fields: dict[str, object] = {
        "memory_id": "mem-1",
        "kind": "preference",
        "key": "reply_language",
        "statement": "Prefers replies written in Vietnamese.",
        "project_code": "atlas",
        "required_scope": "project.docs.read",
        "actor": "priya",
        "session_id": "sess-1",
        "recorded_in_run": "run-1",
        "recorded_at": RECORDED,
        "confidence": 0.9,
    }
    fields.update(overrides)
    return MemoryRecord(**fields)  # type: ignore[arg-type]


def intent(**overrides: object) -> IntentState:
    fields: dict[str, object] = {
        "intent_id": "int-1",
        "goal": "Draft the Q4 risk register",
        "project_code": "atlas",
        "required_scope": "project.docs.read",
        "actor": "priya",
        "session_id": "sess-1",
        "unresolved_slots": ("quarter",),
        "opened_at": RECORDED,
        "updated_at": RECORDED,
    }
    fields.update(overrides)
    return IntentState(**fields)  # type: ignore[arg-type]


def audit_row(**overrides: object) -> MemoryAuditRow:
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
# The adapters are the ports
# --------------------------------------------------------------------------


def test_the_store_satisfies_both_ports_structurally(store) -> None:
    assert isinstance(store, MemoryStorePort)
    assert isinstance(store, IntentStorePort)


# --------------------------------------------------------------------------
# Records: round trip, supersession, scope
# --------------------------------------------------------------------------


def test_a_written_record_round_trips_through_the_database(store) -> None:
    """Revalidated on the way back, so a row written by an older shape of the
    model fails loudly instead of returning a silently wrong object."""
    store.write(record(supersedes=("mem-0",), links=("RISK-3",)))

    assert store.live(scope()) == (record(supersedes=("mem-0",), links=("RISK-3",)),)


def test_writing_the_same_id_twice_leaves_one_row(store) -> None:
    store.write(record())
    store.write(record(statement="Prefers replies written in English."))

    kept = store.live(scope())
    assert len(kept) == 1
    assert kept[0].statement == "Prefers replies written in English."


def test_retiring_returns_the_rows_it_retired(store) -> None:
    store.write(record())

    retired = store.supersede(["mem-1"], at=LATER)

    assert [r.memory_id for r in retired] == ["mem-1"]
    assert retired[0].superseded_at == LATER


def test_a_retired_row_is_kept_and_stops_being_live(store) -> None:
    store.write(record())
    store.supersede(["mem-1"], at=LATER)

    assert store.live(scope()) == ()
    assert store.by_id(["mem-1"], scope()) == ()


def test_retiring_twice_does_not_move_the_date(store) -> None:
    """A consolidator resuming after a crash finishes the job rather than
    rewriting when the assistant stopped believing something."""
    store.write(record())
    store.supersede(["mem-1"], at=RECORDED)

    assert store.supersede(["mem-1"], at=LATER) == ()


def test_retiring_an_unknown_id_is_not_an_error(store) -> None:
    assert store.supersede(["mem-nope"], at=LATER) == ()


def test_another_projects_memory_is_never_returned(store) -> None:
    store.write(record(memory_id="mem-b", project_code="borealis"))

    assert store.live(scope()) == ()


def test_another_actors_preference_is_never_returned(store) -> None:
    """The scope clause is built from bounds(), so the SQL and the in-memory
    store cannot disagree about whose memory a record is."""
    store.write(record(memory_id="mem-m", actor="marco"))

    assert store.live(scope()) == ()


def test_a_project_decision_crosses_sessions_and_actors(store) -> None:
    store.write(
        record(
            memory_id="mem-d",
            kind="decision",
            key="deployment_window",
            statement="The cutover happens on a Thursday evening.",
            actor="marco",
            session_id="sess-other",
        )
    )

    assert [r.memory_id for r in store.live(scope())] == ["mem-d"]


def test_another_sessions_intent_projection_is_never_returned(store) -> None:
    store.write(
        record(
            memory_id="mem-o", kind="intent", key="intent:x", session_id="sess-9"
        )
    )

    assert store.live(scope()) == ()


def test_kinds_narrows_the_listing(store) -> None:
    store.write(record())
    store.write(record(memory_id="mem-d", kind="decision", key="deployment_window"))

    assert [r.memory_id for r in store.live(scope(), kinds=("decision",))] == ["mem-d"]


def test_records_come_back_newest_first(store) -> None:
    store.write(record(memory_id="mem-old", key="a", recorded_at=RECORDED))
    store.write(record(memory_id="mem-new", key="b", recorded_at=LATER))

    assert [r.memory_id for r in store.live(scope())] == ["mem-new", "mem-old"]


def test_hydration_preserves_the_order_the_index_gave(store) -> None:
    """The vector index ranked them; the store must not reorder that."""
    store.write(record(memory_id="mem-a", key="a"))
    store.write(record(memory_id="mem-b", key="b"))

    found = store.by_id(["mem-b", "mem-a"], scope())

    assert [r.memory_id for r in found] == ["mem-b", "mem-a"]


def test_hydration_drops_an_id_this_scope_may_not_read(store) -> None:
    """What makes the index an optimization rather than a second source of
    truth."""
    store.write(record(memory_id="mem-m", actor="marco"))

    assert store.by_id(["mem-m"], scope()) == ()


def test_hydrating_nothing_asks_the_database_nothing(store) -> None:
    assert store.by_id([], scope()) == ()


# --------------------------------------------------------------------------
# The intent, and the index that makes one-per-session a database fact
# --------------------------------------------------------------------------


def test_an_intent_round_trips_with_its_slots(store) -> None:
    store.save_intent(intent().continue_with({"quarter": "Q4"}, at=LATER))

    found = store.open_intent(scope())

    assert found is not None
    assert found.confirmed_slots == {"quarter": "Q4"}
    assert found.unresolved_slots == ()


def test_a_second_open_task_is_refused_by_the_database(store) -> None:
    """The partial unique index, which holds against two processes -- the
    in-memory store can only hold against one."""
    store.save_intent(intent())

    with pytest.raises(ValueError, match="already has a task open"):
        store.save_intent(intent(intent_id="int-2", goal="Check the M2 budget"))


def test_switching_is_allowed_because_it_closes_the_old_task_first(store) -> None:
    store.save_intent(intent())
    previous, fresh = intent().switch_to(
        "Check the M2 budget", intent_id="int-2", at=LATER
    )

    store.save_intent(previous)
    store.save_intent(fresh)

    found = store.open_intent(scope())
    assert found is not None
    assert found.intent_id == "int-2"
    assert found.confirmed_slots == {}


def test_updating_the_same_task_is_not_a_clash(store) -> None:
    store.save_intent(intent())

    store.save_intent(intent().continue_with({"quarter": "Q4"}, at=LATER))

    found = store.open_intent(scope())
    assert found is not None
    assert found.confirmed_slots["quarter"] == "Q4"


def test_a_closed_task_is_not_returned(store) -> None:
    store.save_intent(intent().close(at=LATER))

    assert store.open_intent(scope()) is None


def test_another_sessions_task_is_not_this_sessions_task(store) -> None:
    store.save_intent(intent(intent_id="int-9", session_id="sess-9"))

    assert store.open_intent(scope()) is None


# --------------------------------------------------------------------------
# The audit: every decision, including the ones that stored nothing
# --------------------------------------------------------------------------


def rows(connection) -> list[tuple]:
    return connection.execute(
        "SELECT decision, rejection, memory_id FROM memory_audit ORDER BY id"
    ).fetchall()


def test_a_write_is_recorded(audit, store_connection) -> None:
    audit.record(audit_row())

    assert rows(store_connection) == [("write", None, "mem-1")]


def test_a_rejection_is_recorded_with_the_rule_that_refused_it(
    audit, store_connection
) -> None:
    """The half of the record that shows the policy working rather than merely
    existing."""
    audit.record(
        audit_row(decision="reject", rejection="instruction_like", memory_id="mem-2")
    )

    assert rows(store_connection) == [("reject", "instruction_like", "mem-2")]


def test_all_four_decisions_land_in_one_column(audit, store_connection) -> None:
    audit.record(audit_row(decision="write"))
    audit.record(audit_row(decision="update", memory_id="mem-2"))
    audit.record(audit_row(decision="forget", memory_id="mem-old"))
    audit.record(
        audit_row(decision="reject", rejection="duplicate", memory_id="mem-3")
    )

    assert [row[0] for row in rows(store_connection)] == [
        "write",
        "update",
        "forget",
        "reject",
    ]


def test_the_database_refuses_a_row_that_contradicts_itself(store_connection) -> None:
    """The CHECK constraint, reached by inserting directly -- the model would
    have refused this, and the point is that the database does too."""
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        store_connection.execute(
            """
            INSERT INTO memory_audit
                (occurred_at, trace_id, session_id, project_code, actor,
                 memory_id, kind, decision, rejection, reason, statement_summary)
            VALUES (%s, 'run-1', 'sess-1', 'atlas', 'priya', 'mem-1',
                    'preference', 'write', 'duplicate', '', 'x')
            """,
            (RECORDED,),
        )


def test_the_database_refuses_a_kind_nobody_declared(store_connection) -> None:
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        store_connection.execute(
            """
            INSERT INTO memories
                (memory_id, kind, key, statement, project_code, required_scope,
                 actor, session_id, recorded_in_run, recorded_at, confidence)
            VALUES ('mem-x', 'observation', 'k', 's', 'atlas',
                    'project.docs.read', 'priya', 'sess-1', 'run-1', %s, 0.5)
            """,
            (RECORDED,),
        )


def test_a_failed_audit_insert_does_not_raise(audit, store_connection) -> None:
    """The turn has already answered the user: an exception here would turn
    "we could not write down that we declined to remember something" into a
    failed request."""
    store_connection.execute("DROP TABLE memory_audit")
    try:
        audit.record(audit_row())
    finally:
        with store_connection.cursor() as cursor, store_connection.transaction():
            apply_schema(cursor)
