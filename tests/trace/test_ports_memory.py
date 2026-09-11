"""The in-memory stores keep the contracts the Postgres adapters inherit.

Most of what these tests prove is the pause rule: settling is once, and only
the caller that wins the claim resumes. The dict proves it cheaply; the SQL
adapter proves the same thing against a real database, and a test here that
fails says the Postgres one is about to mislead.
"""

from datetime import UTC, datetime

import pytest

from agentic_erp_assistant.llm.telemetry import ModelCallRecord
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.trace import (
    InMemoryPauseStore,
    InMemoryTraceStore,
    PauseAlreadyPending,
    RunTelemetry,
)
from agentic_erp_assistant.trace.records import RunRecord

STARTED = datetime(2026, 9, 5, 9, 30, tzinfo=UTC)
FINISHED = datetime(2026, 9, 5, 9, 30, 2, tzinfo=UTC)


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


def a_run(trace_id: str = "run-1") -> RunRecord:
    state = AgentState(
        request="How is M2 tracking?",
        actor="bao",
        trace_id=trace_id,
        route="answer",
        terminal=True,
        response="M2 is at risk.",
    )
    return RunRecord(
        trace_id=trace_id,
        outcome="terminal",
        started_at=STARTED,
        finished_at=FINISHED,
        state=state,
    )


# --------------------------------------------------------------------------
# The run store
# --------------------------------------------------------------------------


def test_a_saved_run_loads_back_as_its_state() -> None:
    store = InMemoryTraceStore()
    run = a_run()

    store.save_run(run)

    assert store.load_run("run-1") == run.state


def test_an_unknown_run_loads_as_none_not_an_error() -> None:
    """'Never happened' is a reportable answer, not an exception."""
    assert InMemoryTraceStore().load_run("no-such-run") is None


def test_resaving_a_run_extends_it_rather_than_forking_it() -> None:
    """A resumed run is saved twice; the second write must replace the first,
    because one trace id is one run, not two."""
    store = InMemoryTraceStore()
    first = a_run()
    second = a_run()

    store.save_run(first)
    store.save_run(second)

    assert store.load_run("run-1") == second.state


def test_a_model_call_is_filed_under_its_run() -> None:
    store = InMemoryTraceStore()
    call = ModelCallRecord(
        model="gpt-5.2",
        outcome="answered",
        estimated_input_tokens=100,
        input_tokens=90,
        output_tokens=10,
        cost_usd=0.001,
        latency_seconds=0.5,
        attempts=1,
        occurred_at=FINISHED,
    )

    store.record_model_call("run-1", call)

    assert store.model_calls["run-1"] == [call]


# --------------------------------------------------------------------------
# The pause store
# --------------------------------------------------------------------------


def test_a_paused_state_is_saved_and_comes_back() -> None:
    store = InMemoryPauseStore()
    paused = paused_state()

    store.save(paused)

    assert store.pending("run-1") == paused


def test_only_a_paused_state_may_be_filed() -> None:
    """Queuing a turn nobody was asked to approve would be auditing an
    approval that was never requested."""
    settled = AgentState(
        request="Done.",
        actor="bao",
        trace_id="run-1",
        route="refuse",
        terminal=True,
        response="Not approved.",
    )
    store = InMemoryPauseStore()

    with pytest.raises(ValueError, match="paused"):
        store.save(settled)


def test_a_run_cannot_wait_twice() -> None:
    store = InMemoryPauseStore()

    store.save(paused_state())
    with pytest.raises(PauseAlreadyPending):
        store.save(paused_state())


def test_the_first_claim_wins_and_the_second_gets_nothing() -> None:
    """The property the whole approval story rests on: a decision about one
    call is recorded once, and only the winner resumes."""
    store = InMemoryPauseStore()
    store.save(paused_state())

    first = store.claim("run-1", approved=True)
    second = store.claim("run-1", approved=False)

    assert first is not None
    assert second is None


def test_a_denied_claim_removes_the_pause_too() -> None:
    """Both answers settle the queue; 'no' is not a retry prompt."""
    store = InMemoryPauseStore()
    store.save(paused_state())

    assert store.claim("run-1", approved=False) is not None
    assert store.pending("run-1") is None


def test_claiming_a_run_nobody_paused_gets_nothing() -> None:
    assert InMemoryPauseStore().claim("no-such-run", approved=True) is None


def test_claim_keeps_decided_by_beside_the_decision() -> None:
    store = InMemoryPauseStore()
    store.save(paused_state())

    store.claim("run-1", approved=True, decided_by="priya")

    assert store.decided_by("run-1") == "priya"


def test_decided_by_defaults_to_none_when_nobody_is_named() -> None:
    store = InMemoryPauseStore()
    store.save(paused_state())

    store.claim("run-1", approved=True)

    assert store.decided_by("run-1") is None


# --------------------------------------------------------------------------
# The run-scoped telemetry adapter
# --------------------------------------------------------------------------


def test_run_telemetry_stamps_the_run_onto_each_record() -> None:
    store = InMemoryTraceStore()
    sink = RunTelemetry(trace_id="run-1", store=store)
    call = ModelCallRecord(
        model="gpt-5.2",
        outcome="answered",
        estimated_input_tokens=1,
        input_tokens=1,
        output_tokens=1,
        cost_usd=None,
        latency_seconds=0.1,
        attempts=1,
        occurred_at=FINISHED,
    )

    sink.record(call)

    assert store.model_calls["run-1"] == [call]
    assert "run-2" not in store.model_calls