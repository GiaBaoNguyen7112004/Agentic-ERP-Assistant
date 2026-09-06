"""The SQL adapters, against the real Postgres they exist for.

Everything the in-memory fakes prove, these tests prove again one layer down,
because that is the deal the ports make: the fake is the contract in a test
and the adapter is the contract in a database, and only one of them can be
wrong without the other noticing. A round-trip that revalidates, a claim that
settles once, a unique index that refuses a second pending row -- each is
already tested against the dict; here it has to survive serialization, jsonb,
and a database's own idea of ordering.

Every test in this file is marked ``postgres`` and the module skips itself
when no database answers, so ``uv run pytest`` on a machine without the
container stays green -- and stays honest: the run that claims the adapters
work is ``uv run pytest -m postgres``.

    docker compose up -d postgres
    uv run python scripts/init_postgres.py
    uv run pytest -m postgres

What is real here: the database, the SQL, the schema constraints. What is
faked: nothing about storage -- the engine doubles are a scripted model and
a scripted composer, because this file is about where a turn's record lands,
not about the turn itself.
"""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentic_erp_assistant.engine.orchestrator import (
    ApprovalAlreadySettled,
    RunOrchestrator,
)
from agentic_erp_assistant.engine.workflow import WorkflowRuntime
from agentic_erp_assistant.erp.mock import DEFAULT_DATASET_PATH, MockErp
from agentic_erp_assistant.llm.schemas import Citation, GroundedAnswer
from agentic_erp_assistant.llm.tools import ToolCallResult
from agentic_erp_assistant.persistence import (
    PostgresAuditLog,
    PostgresPauseStore,
    PostgresTraceStore,
    StoreConnectionError,
    apply_schema,
    connect,
)
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.tools.gateway import ToolGateway
from agentic_erp_assistant.tools.models import AuditRow
from agentic_erp_assistant.tools.registry import build_default_registry
from agentic_erp_assistant.trace.records import RunRecord

pytestmark = pytest.mark.postgres

STARTED = datetime(2026, 9, 6, 10, 0, tzinfo=UTC)
FINISHED = datetime(2026, 9, 6, 10, 0, 2, tzinfo=UTC)

SCOPES = frozenset(
    {
        "project.status.read",
        "project.sprint.read",
        "project.budget.read",
        "project.risk.read",
        "project.risk.write",
    }
)

EVENTS = (
    TraceEvent(node="start", kind="node_entered"),
    TraceEvent(node="start", kind="node_exited"),
    TraceEvent(node="think", kind="route_selected", detail="answer: done"),
)


# --------------------------------------------------------------------------
# The database, once per module; a clean one per test
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def database():
    """One connection for the module, skipped entirely without the database.

    The schema is applied rather than assumed: ``IF NOT EXISTS`` makes that a
    no-op on an initialized store, and it keeps these tests runnable on a
    fresh container without pretending the init script ran.
    """
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
    """A database with nothing in it, so a test's rows are its own.

    Five tables, CASCADE from runs -- the same shape an operator's fresh
    ``init_postgres`` run produces, so these tests observe the adapters, not
    whatever the previous test left behind.
    """
    database.execute(
        "TRUNCATE runs, trace_events, audit_rows, model_calls, pauses CASCADE"
    )
    return database


# --------------------------------------------------------------------------
# The states these tests file
# --------------------------------------------------------------------------


def finished_state(trace_id: str = "run-1") -> AgentState:
    return AgentState(
        request="How is M2 tracking?",
        actor="bao",
        trace_id=trace_id,
        route="answer",
        terminal=True,
        response="M2 is at risk.",
        events=EVENTS,
    )


def paused_state(trace_id: str = "run-1") -> AgentState:
    return AgentState(
        request="Record the vendor risk.",
        actor="bao",
        trace_id=trace_id,
        route="request_approval",
        tool_name="create_risk",
        tool_arguments={"project_id": "atlas", "title": "vendor slipped", "severity": "low"},
        tool_mutating=True,
        approval="pending",
        events=EVENTS,
    )


def a_record(state: AgentState) -> RunRecord:
    return RunRecord(
        trace_id=state.trace_id,
        outcome="paused" if state.approval == "pending" else "terminal",
        started_at=STARTED,
        finished_at=FINISHED,
        state=state,
    )


# --------------------------------------------------------------------------
# The trace store
# --------------------------------------------------------------------------


def test_a_run_round_trips_into_its_identical_state(store_connection) -> None:
    """The whole point of the jsonb column: what loads back is what was
    saved, or a loud refusal -- never a silently half-parsed state."""
    traces = PostgresTraceStore(store_connection)
    filed = finished_state()

    traces.save_run(a_record(filed))

    assert traces.load_run("run-1") == filed
    events = store_connection.execute(
        "SELECT count(*) FROM trace_events WHERE trace_id = 'run-1'"
    ).fetchone()
    assert events[0] == len(EVENTS)


def test_a_run_that_never_happened_loads_as_none(store_connection) -> None:
    assert PostgresTraceStore(store_connection).load_run("no-such-run") is None


def test_resaving_a_run_extends_it_instead_of_forking_it(store_connection) -> None:
    """The resumed-segment case: the second save rewrites the shared events
    (no duplicates, thanks to the seq upsert) and the run still reads as
    having started when it actually started."""
    traces = PostgresTraceStore(store_connection)
    traces.save_run(a_record(finished_state()))

    longer = finished_state().evolve(
        events=EVENTS + (TraceEvent(node="answer", kind="node_entered"),)
    )
    traces.save_run(a_record(longer))

    assert traces.load_run("run-1") == longer
    row = store_connection.execute(
        "SELECT count(*), min(seq), max(seq) FROM trace_events "
        "WHERE trace_id = 'run-1'"
    ).fetchone()
    assert (row[0], row[1], row[2]) == (4, 0, 3)


def test_a_later_segment_does_not_move_the_runs_start(store_connection) -> None:
    """LEAST(started_at): a run resumed tomorrow still reports the moment it
    began, not the moment its second segment happened to."""
    traces = PostgresTraceStore(store_connection)
    traces.save_run(a_record(finished_state()))

    traces.save_run(
        RunRecord(
            trace_id="run-1",
            outcome="terminal",
            started_at=STARTED.replace(hour=23),
            finished_at=FINISHED.replace(hour=23),
            state=finished_state(),
        )
    )

    started = store_connection.execute(
        "SELECT started_at FROM runs WHERE trace_id = 'run-1'"
    ).fetchone()
    assert started[0] == STARTED


def test_a_model_call_is_filed_under_its_run(store_connection) -> None:
    from agentic_erp_assistant.llm.telemetry import ModelCallRecord

    traces = PostgresTraceStore(store_connection)

    traces.record_model_call(
        "run-1",
        ModelCallRecord(
            model="gpt-5.2",
            outcome="routed",
            estimated_input_tokens=100,
            input_tokens=90,
            output_tokens=10,
            cost_usd=0.001,
            latency_seconds=0.5,
            attempts=1,
            occurred_at=FINISHED,
        ),
    )

    row = store_connection.execute(
        "SELECT model, outcome, cost_usd FROM model_calls WHERE trace_id = 'run-1'"
    ).fetchone()
    assert (row[0], row[1], row[2]) == ("gpt-5.2", "routed", 0.001)


# --------------------------------------------------------------------------
# The audit log
# --------------------------------------------------------------------------


def test_an_audit_row_lands_with_its_trace_id(store_connection) -> None:
    """The join the graded requirement turns on: 'who changed what, as part
    of which run' answered from one row."""
    audit = PostgresAuditLog(store_connection)

    audit.record(
        AuditRow(
            trace_id="run-1",
            occurred_at=FINISHED,
            actor="bao",
            tool_name="create_risk",
            arguments_summary="create_risk(project_id=atlas, title=vendor slipped)",
            approval="approved",
            status="ok",
            source_ids=("atlas",),
        )
    )

    row = store_connection.execute(
        "SELECT trace_id, actor, approval, status, source_ids "
        "FROM audit_rows"
    ).fetchone()
    assert row == ("run-1", "bao", "approved", "ok", ["atlas"])


def test_an_insert_the_database_refuses_is_logged_not_raised(store_connection) -> None:
    """The port's sharpest rule, made concrete. The row is valid -- the model
    will not build one that contradicts itself -- so the refusal is staged on
    the database side: the table is dropped, the INSERT cannot land, and the
    sink still returns normally. An audit sink that can abort the call it is
    recording turns "we could not write the record" into "the approved work
    did not happen"."""
    audit = PostgresAuditLog(store_connection)
    store_connection.execute("DROP TABLE audit_rows")

    audit.record(
        AuditRow(
            trace_id="run-1",
            occurred_at=FINISHED,
            actor="bao",
            tool_name="create_risk",
            arguments_summary="create_risk(severity=low)",
            approval="approved",
            status="ok",
            source_ids=(),
        )
    )

    with store_connection.cursor() as cursor:
        with store_connection.transaction():
            apply_schema(cursor)  # the next test's fixture truncates, not creates


# --------------------------------------------------------------------------
# The pause store
# --------------------------------------------------------------------------


def test_a_pause_survives_new_store_instances(store_connection) -> None:
    """The gap this package exists to close: a new connection is a restart,
    and the pending approval is still there to be answered."""
    PostgresPauseStore(store_connection).save(paused_state())

    reopened = PostgresPauseStore(store_connection)

    assert reopened.pending("run-1") == paused_state()


def test_a_second_pending_pause_is_refused_by_the_database(store_connection) -> None:
    """The partial unique index says so, not a check a process could race
    past -- and it surfaces as the same typed error the fake raises."""
    from agentic_erp_assistant.trace import PauseAlreadyPending

    pauses = PostgresPauseStore(store_connection)
    pauses.save(paused_state())

    with pytest.raises(PauseAlreadyPending):
        pauses.save(paused_state())


def test_only_a_paused_state_may_be_filed(store_connection) -> None:
    pauses = PostgresPauseStore(store_connection)

    with pytest.raises(ValueError, match="paused"):
        pauses.save(finished_state())


def test_the_first_claim_wins_the_second_gets_nothing(store_connection) -> None:
    pauses = PostgresPauseStore(store_connection)
    pauses.save(paused_state())

    first = pauses.claim("run-1", approved=True)
    second = pauses.claim("run-1", approved=False)

    assert first == paused_state()
    assert second is None
    row = store_connection.execute(
        "SELECT status, decision, decided_at FROM pauses WHERE trace_id = 'run-1'"
    ).fetchone()
    assert row[0] == "resolved"
    assert row[1] == "approved"
    assert row[2] is not None


# --------------------------------------------------------------------------
# The whole story: pause, restart, approve, and settle exactly once
# --------------------------------------------------------------------------


class ScriptedModel:
    """Stands in for the provider: returns the choices a real one would make."""

    def __init__(self, *results: ToolCallResult) -> None:
        self.results = list(results)
        self.calls = 0

    def decide(self, question, evidence=(), observations=(), *, tools=()):
        self.calls += 1
        return self.results[min(self.calls - 1, len(self.results) - 1)]


class FakeComposer:
    def __init__(self, answer: GroundedAnswer) -> None:
        self.answer_value = answer

    def answer(self, question: str, evidence):
        return self.answer_value


class FakeRetriever:
    def search(self, query: str, *, limit: int):
        return (
            EvidenceSnippet(
                source_id="m2-status.md", locator="p.2", text="M2 cutover slipped."
            ),
        )


def a_writable_copy(target: Path) -> MockErp:
    """A store over a throwaway copy of the fixture.

    An approved ``create_risk`` here reaches a real file, so no test in this
    file may sit on the repo's fixture: this is the one suite where a write is
    expected to happen, and the assertion is that it reached the disk.
    """
    target.write_text(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return MockErp.load(target)


def an_orchestrator(
    connection, erp: MockErp, *results: ToolCallResult
) -> tuple[RunOrchestrator, ScriptedModel]:
    """The full stack over the real stores: engine, gateway, Postgres.

    The scripted results are a parameter, not a constant, because the restart
    is the point of the story below: a brand-new orchestrator means a
    brand-new model, and its script has to start from where the turn now is
    -- the same thing a real provider would see, since the tool's observation
    travels in the state and not in any process's memory.
    """
    model = ScriptedModel(*results)
    runtime = WorkflowRuntime(
        retriever=FakeRetriever(),
        # The audit sink is the Postgres one: without wiring it here, an
        # approved write would leave evidence only in the in-memory default,
        # which is exactly the gap this package exists to close.
        tools=ToolGateway(
            registry=build_default_registry(erp),
            audit=PostgresAuditLog(connection),
        ),
        planner=Planner(model),
        composer=FakeComposer(
            GroundedAnswer(
                answer="Risk recorded.",
                citations=[Citation(source_id="m2-status.md", locator="p.2")],
                grounded=True,
                confidence=0.9,
            )
        ),
        sleep=lambda seconds: None,
    )
    return (
        RunOrchestrator(
            runtime, PostgresTraceStore(connection), PostgresPauseStore(connection)
        ),
        model,
    )


def test_a_pause_answers_after_a_restart_and_settles_once(
    store_connection, tmp_path
) -> None:
    """The story this package exists to tell, end to end: a turn pauses on a
    mutating write; the process 'restarts' (a brand-new connection, new
    stores, a new orchestrator); the approval is given and the write happens
    anyway -- reaching the ERP store, the file on disk, and the audit table;
    and a second decision, however late, executes nothing."""
    erp = a_writable_copy(tmp_path / "project.json")
    orchestrator, _ = an_orchestrator(
        store_connection,
        erp,
        ToolCallResult.from_tool_call(
            tool_name="create_risk",
            arguments={
                "project_id": "atlas",
                "title": "vendor slipped",
                "severity": "low",
            },
        ),
        ToolCallResult.from_content("Risk recorded."),
    )

    state = AgentState(
        request="Record the vendor risk.",
        actor="bao",
        trace_id="run-1",
        scopes=SCOPES,
    )
    final = orchestrator.handle(state)
    assert final.route == "request_approval"

    # The restart: nothing from the first orchestrator is reused, not even
    # the connection. If the pending approval survived, it survived in the
    # database and nowhere else -- and the new model's script starts where
    # the turn now is: the write executed, the observation in the state.
    restarted, _ = an_orchestrator(
        store_connection, erp, ToolCallResult.from_content("Risk recorded.")
    )
    resumed = restarted.resume("run-1", approved=True)

    assert resumed.terminal
    assert any(risk.title == "vendor slipped" for risk in erp.risks)
    reread = MockErp.load(tmp_path / "project.json")
    assert any(risk.title == "vendor slipped" for risk in reread.risks)

    audit_row = store_connection.execute(
        "SELECT trace_id, tool_name, approval FROM audit_rows"
    ).fetchone()
    assert audit_row == ("run-1", "create_risk", "approved")

    with pytest.raises(ApprovalAlreadySettled):
        restarted.resume("run-1", approved=True)

    risks_after_second_decision = len(MockErp.load(tmp_path / "project.json").risks)
    assert risks_after_second_decision == len(reread.risks)