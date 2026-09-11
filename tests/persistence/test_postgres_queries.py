"""EvidenceQueries against the real database: the read model a screen needs.

Populated through the real write-side adapters wherever one exists (a run
through PostgresTraceStore, an audit row through PostgresAuditLog, a memory
through PostgresMemoryStore) so what is proved is "the read model agrees with
what the write path actually wrote", not "two hand-written SQL statements
agree with each other".

    docker compose up -d postgres
    uv run python scripts/init_postgres.py
    uv run pytest -m postgres
"""

from datetime import UTC, datetime

import pytest

from agentic_erp_assistant.llm.telemetry import ModelCallRecord
from agentic_erp_assistant.memory.audit import MemoryAuditRow
from agentic_erp_assistant.persistence.connection import (
    StoreConnectionError,
    connect,
)
from agentic_erp_assistant.persistence.postgres_audit import PostgresAuditLog
from agentic_erp_assistant.persistence.postgres_conversation import (
    PostgresConversationStore,
)
from agentic_erp_assistant.persistence.postgres_memory import (
    PostgresMemoryAudit,
    PostgresMemoryStore,
)
from agentic_erp_assistant.persistence.postgres_pause import PostgresPauseStore
from agentic_erp_assistant.persistence.postgres_queries import EvidenceQueries
from agentic_erp_assistant.persistence.postgres_trace import PostgresTraceStore
from agentic_erp_assistant.persistence.schema import apply_schema
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.tools.models import AuditRow
from agentic_erp_assistant.trace.records import RunRecord

pytestmark = pytest.mark.postgres

WHEN = datetime(2026, 9, 10, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 10, 9, 0, 5, tzinfo=UTC)

SCOPES = frozenset({"project.status.read", "project.risk.write"})


@pytest.fixture(scope="module")
def database():
    try:
        connection = connect()
    except StoreConnectionError as error:
        pytest.skip(f"no Postgres to test against: {error}")
    with connection.cursor() as cursor:
        with connection.transaction():
            apply_schema(cursor)
    yield connection
    connection.close()


@pytest.fixture
def store_connection(database):
    database.execute(
        "TRUNCATE runs, trace_events, audit_rows, model_calls, pauses, "
        "memories, memory_audit, session_turns CASCADE"
    )
    return database


def finished_state(trace_id: str = "run-1", **changes) -> AgentState:
    base = AgentState(
        request="How is M2 tracking?",
        actor="priya",
        project_code="atlas",
        trace_id=trace_id,
        route="answer",
        terminal=True,
        response="M2 is at risk, two days late.",
        events=(
            TraceEvent(node="start", kind="node_entered"),
            TraceEvent(node="think", kind="route_selected", detail="answer"),
        ),
    )
    return base.evolve(**changes) if changes else base


def a_run(trace_id: str = "run-1", **changes) -> RunRecord:
    state = finished_state(trace_id, **changes)
    return RunRecord(
        trace_id=trace_id,
        outcome="terminal",
        started_at=WHEN,
        finished_at=LATER,
        state=state,
    )


# --------------------------------------------------------------------------
# run() and events()
# --------------------------------------------------------------------------


def test_run_returns_the_revalidated_state(store_connection) -> None:
    PostgresTraceStore(store_connection).save_run(a_run())

    row = EvidenceQueries(store_connection).run("run-1")

    assert row is not None
    assert row.trace_id == "run-1"
    assert row.actor == "priya"
    assert row.project_code == "atlas"
    assert row.outcome == "terminal"
    assert row.state == finished_state("run-1")


def test_run_of_an_unknown_trace_is_none(store_connection) -> None:
    assert EvidenceQueries(store_connection).run("no-such-run") is None


def test_events_come_back_in_seq_order(store_connection) -> None:
    PostgresTraceStore(store_connection).save_run(a_run())

    events = EvidenceQueries(store_connection).events("run-1")

    assert [event.kind for event in events] == ["node_entered", "route_selected"]


# --------------------------------------------------------------------------
# audit_rows()
# --------------------------------------------------------------------------


def test_audit_rows_come_back_in_the_order_they_were_recorded(store_connection) -> None:
    PostgresTraceStore(store_connection).save_run(a_run())
    audit = PostgresAuditLog(store_connection)
    audit.record(
        AuditRow(
            trace_id="run-1",
            occurred_at=WHEN,
            actor="priya",
            tool_name="create_risk",
            arguments_summary="create_risk(...)",
            approval="not_required",
            status="approval_required",
            source_ids=(),
        )
    )
    audit.record(
        AuditRow(
            trace_id="run-1",
            occurred_at=LATER,
            actor="priya",
            tool_name="create_risk",
            arguments_summary="create_risk(...)",
            approval="approved",
            status="ok",
            source_ids=("risk-r-3",),
        )
    )

    rows = EvidenceQueries(store_connection).audit_rows("run-1")

    assert [row.status for row in rows] == ["approval_required", "ok"]
    assert rows[1].source_ids == ("risk-r-3",)


# --------------------------------------------------------------------------
# model_calls() and its totals
# --------------------------------------------------------------------------


def test_model_calls_totals_sum_tokens_and_cost_when_every_call_is_priced(
    store_connection,
) -> None:
    PostgresTraceStore(store_connection).save_run(a_run())
    traces = PostgresTraceStore(store_connection)
    traces.record_model_call(
        "run-1",
        ModelCallRecord(
            model="gpt-4o", outcome="answered", estimated_input_tokens=100,
            input_tokens=90, output_tokens=10, cost_usd=0.01,
            latency_seconds=0.5, attempts=1, occurred_at=WHEN,
        ),
    )
    traces.record_model_call(
        "run-1",
        ModelCallRecord(
            model="gpt-4o", outcome="routed", estimated_input_tokens=50,
            input_tokens=45, output_tokens=5, cost_usd=0.005,
            latency_seconds=0.2, attempts=1, occurred_at=LATER,
        ),
    )

    records, totals = EvidenceQueries(store_connection).model_calls("run-1")

    assert len(records) == 2
    assert totals.count == 2
    assert totals.input_tokens == 135
    assert totals.output_tokens == 15
    assert totals.cost_usd == pytest.approx(0.015)
    assert totals.unpriced == 0


def test_model_calls_totals_report_cost_as_none_when_any_call_is_unpriced(
    store_connection,
) -> None:
    PostgresTraceStore(store_connection).save_run(a_run())
    traces = PostgresTraceStore(store_connection)
    traces.record_model_call(
        "run-1",
        ModelCallRecord(
            model="gpt-4o", outcome="answered", estimated_input_tokens=100,
            input_tokens=90, output_tokens=10, cost_usd=0.01,
            latency_seconds=0.5, attempts=1, occurred_at=WHEN,
        ),
    )
    traces.record_model_call(
        "run-1",
        ModelCallRecord(
            model="some-unpriced-model", outcome="answered",
            estimated_input_tokens=10, input_tokens=10, output_tokens=1,
            cost_usd=None, latency_seconds=0.1, attempts=1, occurred_at=LATER,
        ),
    )

    _, totals = EvidenceQueries(store_connection).model_calls("run-1")

    assert totals.cost_usd is None
    assert totals.unpriced == 1


# --------------------------------------------------------------------------
# pending_approvals()
# --------------------------------------------------------------------------


def paused_state(trace_id: str = "run-2") -> AgentState:
    return AgentState(
        request="Record the vendor risk.",
        actor="priya",
        project_code="atlas",
        trace_id=trace_id,
        session_id="sess-1",
        route="request_approval",
        tool_name="create_risk",
        tool_arguments={"project_id": "atlas", "title": "x", "severity": "low"},
        tool_mutating=True,
        approval="pending",
    )


def test_pending_approvals_lists_only_pending_pauses(store_connection) -> None:
    pauses = PostgresPauseStore(store_connection)
    pauses.save(paused_state("run-2"))
    pauses.save(paused_state("run-3"))
    pauses.claim("run-3", approved=True)  # settled, must not appear

    listed = EvidenceQueries(store_connection).pending_approvals()

    assert [row.trace_id for row in listed] == ["run-2"]
    assert listed[0].tool_name == "create_risk"


def test_pending_approvals_filters_by_actor(store_connection) -> None:
    pauses = PostgresPauseStore(store_connection)
    pauses.save(paused_state("run-2").evolve(actor="priya"))
    pauses.save(paused_state("run-3").evolve(actor="wei"))

    listed = EvidenceQueries(store_connection).pending_approvals(actor="wei")

    assert [row.trace_id for row in listed] == ["run-3"]


def test_pending_approvals_carries_the_session_id_from_session_turns(
    store_connection,
) -> None:
    pauses = PostgresPauseStore(store_connection)
    pauses.save(paused_state("run-2"))
    PostgresConversationStore(store_connection).append(
        ConversationTurn(
            trace_id="run-2", session_id="sess-1", actor="priya",
            request="Record the vendor risk.", response=None,
            route="request_approval", tool_name="create_risk",
            approval="pending", started_at=WHEN, finished_at=WHEN,
        )
    )

    listed = EvidenceQueries(store_connection).pending_approvals()

    assert listed[0].session_id == "sess-1"


# --------------------------------------------------------------------------
# memory_audit() and memories()
# --------------------------------------------------------------------------


def test_memory_audit_lists_decisions_for_the_run(store_connection) -> None:
    PostgresTraceStore(store_connection).save_run(a_run())
    PostgresMemoryAudit(store_connection).record(
        MemoryAuditRow(
            occurred_at=WHEN, trace_id="run-1", session_id="sess-1",
            project_code="atlas", actor="priya", memory_id="mem-1",
            kind="preference", decision="write", statement_summary="replies in Vietnamese",
        )
    )

    rows = EvidenceQueries(store_connection).memory_audit("run-1")

    assert len(rows) == 1
    assert rows[0].decision == "write"


def test_memories_lists_only_live_records_in_scope(store_connection) -> None:
    store = PostgresMemoryStore(store_connection)
    live = MemoryRecord(
        memory_id="mem-1", kind="preference", key="reply_language",
        statement="Prefers Vietnamese.", project_code="atlas",
        required_scope="project.docs.read", actor="priya", session_id="sess-1",
        recorded_in_run="run-1", recorded_at=WHEN, confidence=0.9,
    )
    other_actor = MemoryRecord(
        memory_id="mem-2", kind="preference", key="reply_language",
        statement="Prefers English.", project_code="atlas",
        required_scope="project.docs.read", actor="wei", session_id="sess-2",
        recorded_in_run="run-2", recorded_at=WHEN, confidence=0.9,
    )
    store.write(live)
    store.write(other_actor)

    memories = EvidenceQueries(store_connection).memories(
        actor="priya", project_code="atlas", session_id="sess-1"
    )

    assert [m.memory_id for m in memories] == ["mem-1"]


# --------------------------------------------------------------------------
# sessions() and session_turns()
# --------------------------------------------------------------------------


def a_turn(trace_id: str, session_id: str, *, started_at: datetime, request: str) -> ConversationTurn:
    return ConversationTurn(
        trace_id=trace_id, session_id=session_id, actor="priya",
        request=request, response="ok", route="answer",
        started_at=started_at, finished_at=started_at,
    )


def test_sessions_lists_the_actors_sessions_most_recent_first(
    store_connection,
) -> None:
    store = PostgresConversationStore(store_connection)
    store.append(a_turn("run-1", "sess-1", started_at=WHEN, request="First question."))
    store.append(a_turn("run-2", "sess-2", started_at=LATER, request="Second session."))

    sessions = EvidenceQueries(store_connection).sessions("priya")

    assert [s.session_id for s in sessions] == ["sess-2", "sess-1"]
    assert sessions[1].first_request == "First question."


def test_session_turns_are_ordered_oldest_first(store_connection) -> None:
    store = PostgresConversationStore(store_connection)
    store.append(a_turn("run-2", "sess-1", started_at=LATER, request="Second."))
    store.append(a_turn("run-1", "sess-1", started_at=WHEN, request="First."))

    turns = EvidenceQueries(store_connection).session_turns("sess-1", "priya")

    assert [t.request for t in turns] == ["First.", "Second."]
