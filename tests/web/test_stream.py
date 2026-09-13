"""TurnStream: the thread -> asyncio bridge, exercised with a real thread.

Every producing method (step, trace_event, delta, reset) is called from a
genuine background thread -- the shape the real request handler uses
(loop.run_in_executor runs the turn on a worker thread; the route coroutine
owns the loop and awaits events()) -- while the loop consumes on the test's
own thread. Nothing here mocks call_soon_threadsafe; if the bridge were
broken, these tests would hang or lose events, not just fail an assertion.
"""

import asyncio
import threading
from collections.abc import Callable

from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.reply_contract import ReplyContract
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.web.protocol import (
    ContextEvent,
    ResetEvent,
    ServerEvent,
    StepEvent,
    TokenEvent,
    TraceRow,
    TurnStartedEvent,
)
from agentic_erp_assistant.web.stream import TurnStream

from tests.memory.builders import make_record, make_turn

SCOPES = frozenset({"project.status.read"})


def state(**changes) -> AgentState:
    base = AgentState(
        request="How is M2 tracking?", actor="priya", project_code="atlas",
        trace_id="run-1", scopes=SCOPES,
    )
    return base.evolve(**changes) if changes else base


def run_and_collect(
    produce: Callable[[TurnStream], None], *, start_seq: int = 0
) -> list[ServerEvent]:
    """Build a TurnStream, run ``produce`` on a real background thread, and
    collect everything it put onto the stream, in order, until close()."""
    loop = asyncio.new_event_loop()
    stream = TurnStream(loop, start_seq=start_seq)
    collected: list[ServerEvent] = []

    def worker() -> None:
        produce(stream)
        stream.close()

    thread = threading.Thread(target=worker)

    async def consume() -> None:
        thread.start()
        async for event in stream.events():
            collected.append(event)

    try:
        loop.run_until_complete(consume())
    finally:
        thread.join(timeout=5)
        loop.close()
    return collected


# --------------------------------------------------------------------------
# context(): the turn's prelude, and the baseline for the first step's delta
# --------------------------------------------------------------------------


def test_context_emits_the_turns_history_memory_and_contract() -> None:
    declared = state(
        history=(make_turn(),),
        memories=(make_record(),),
        contract=ReplyContract(needs=frozenset({"erp_field"})),
    )

    (event,) = run_and_collect(lambda s: s.context(declared))

    assert isinstance(event, ContextEvent)
    assert event.request == declared.request
    assert len(event.history) == 1
    assert event.history[0].trace_id == "run-1"
    assert len(event.memories) == 1
    assert event.memories[0].memory_id == "mem-1"
    assert event.contract is not None
    assert event.contract.needs == ("erp_field",)


def test_context_with_no_declared_contract_is_none() -> None:
    (event,) = run_and_collect(lambda s: s.context(state()))

    assert event.history == ()
    assert event.memories == ()
    assert event.contract is None


def test_context_seeds_the_delta_baseline_for_the_first_step() -> None:
    """A resumed turn's context is the paused state itself, so the first
    step after it reports only what changed since the pause -- not since
    the turn began."""
    declared = state()  # no evidence, no observations yet
    stepped = declared.evolve(
        events=(TraceEvent(node="start", kind="node_entered"),),
        step_count=1,
    )

    def produce(s: TurnStream) -> None:
        s.context(declared)
        s.step(stepped)

    step_event = next(e for e in run_and_collect(produce) if isinstance(e, StepEvent))
    assert step_event.evidence is None
    assert step_event.observations == ()


# --------------------------------------------------------------------------
# step(): trace rows for new events, then a StepEvent
# --------------------------------------------------------------------------


def test_step_emits_new_trace_rows_then_a_step_event() -> None:
    first = state(
        events=(
            TraceEvent(node="start", kind="node_entered"),
            TraceEvent(node="think", kind="route_selected", detail="answer"),
        ),
        route="answer",
        step_count=1,
    )

    events = run_and_collect(lambda s: s.step(first))

    assert [type(e).__name__ for e in events] == ["TraceRow", "TraceRow", "StepEvent"]
    assert [e.seq for e in events[:2]] == [0, 1]
    assert events[0].detail == ""
    assert events[1].detail == "answer"
    assert events[2].route == "answer"
    assert events[2].step_count == 1


def test_step_only_emits_rows_new_since_the_last_call() -> None:
    first = state(events=(TraceEvent(node="start", kind="node_entered"),), step_count=1)
    second = first.evolve(
        events=first.events + (TraceEvent(node="think", kind="route_selected", detail="answer"),),
        route="answer",
        step_count=2,
    )

    def produce(s: TurnStream) -> None:
        s.step(first)
        s.step(second)

    events = run_and_collect(produce)

    trace_rows = [e for e in events if isinstance(e, TraceRow)]
    assert len(trace_rows) == 2  # not 3 -- the first row is not repeated
    assert trace_rows[0].seq == 0
    assert trace_rows[1].seq == 1


def test_start_seq_skips_events_the_client_already_saw() -> None:
    """A resumed turn's state already carries the pre-pause events; only
    what is newer than start_seq should stream."""
    resumed = state(
        events=(
            TraceEvent(node="start", kind="node_entered"),
            TraceEvent(node="think", kind="approval_requested", detail="create_risk needs a human"),
            TraceEvent(node="approval", kind="approval_recorded", detail="create_risk approved by priya"),
        ),
        step_count=1,
    )

    events = run_and_collect(lambda s: s.step(resumed), start_seq=2)

    trace_rows = [e for e in events if isinstance(e, TraceRow)]
    assert len(trace_rows) == 1
    assert trace_rows[0].seq == 2
    assert trace_rows[0].detail == "create_risk approved by priya"


def test_step_reports_the_node_that_just_ran() -> None:
    with_span = state(
        events=(
            TraceEvent(node="call_tool", kind="node_entered"),
            TraceEvent(node="execute_tool", kind="tool_called", detail="get_project_status -> ok"),
            TraceEvent(node="call_tool", kind="node_exited"),
        ),
        step_count=1,
    )

    step_event = next(e for e in run_and_collect(lambda s: s.step(with_span)) if isinstance(e, StepEvent))

    assert step_event.node == "call_tool"


def test_step_reports_no_node_for_a_bare_engine_level_change() -> None:
    bare = state(
        events=(
            TraceEvent(node="approval", kind="approval_recorded", detail="create_risk approved by priya"),
        ),
        step_count=1,
    )

    step_event = next(e for e in run_and_collect(lambda s: s.step(bare)) if isinstance(e, StepEvent))

    assert step_event.node is None


def test_step_sends_evidence_only_the_first_time_it_appears() -> None:
    snippet = EvidenceSnippet(source_id="doc-1", locator="p.1", text="Some passage text.")
    first = state(
        evidence=(snippet,),
        events=(TraceEvent(node="retrieve_project_documents", kind="node_entered"),),
        step_count=1,
    )
    second = first.evolve(
        events=first.events + (TraceEvent(node="retrieve_project_documents", kind="node_exited"),),
        route="answer",
        response="An answer.",
        terminal=True,
        step_count=1,
    )

    def produce(s: TurnStream) -> None:
        s.step(first)  # no context() call -- _last starts None, so this sends it
        s.step(second)  # unchanged since `first` -- must not resend the passage

    steps = [e for e in run_and_collect(produce) if isinstance(e, StepEvent)]

    assert steps[0].evidence is not None
    assert steps[0].evidence[0].source_id == "doc-1"
    assert steps[0].evidence[0].tag == "[doc-1#p.1]"
    assert steps[1].evidence is None


def test_step_sends_only_the_observations_new_since_the_previous_step() -> None:
    read = ToolOutcome(
        tool_name="list_risks", status="ok", summary="2 open risks.", source_ids=("project-atlas",)
    )
    write = ToolOutcome(
        tool_name="create_risk", status="ok", summary="Recorded R-3.", source_ids=("risk-r-3",)
    )
    first = state(observations=(read,), step_count=1)
    second = first.evolve(observations=(read, write), step_count=2)

    def produce(s: TurnStream) -> None:
        s.step(first)
        s.step(second)

    steps = [e for e in run_and_collect(produce) if isinstance(e, StepEvent)]

    assert [o.tool_name for o in steps[0].observations] == ["list_risks"]
    assert [o.tool_name for o in steps[1].observations] == ["create_risk"]


def test_step_reflects_the_terminal_response_and_failure() -> None:
    finished = state(route="answer", response="All good.", failure="none", terminal=True, step_count=1)

    step_event = next(e for e in run_and_collect(lambda s: s.step(finished)) if isinstance(e, StepEvent))

    assert step_event.response == "All good."
    assert step_event.failure == "none"


def test_step_elapsed_ms_is_never_negative() -> None:
    only = state(events=(TraceEvent(node="start", kind="node_entered"),), step_count=1)

    step_event = next(e for e in run_and_collect(lambda s: s.step(only)) if isinstance(e, StepEvent))

    assert step_event.elapsed_ms >= 0


# --------------------------------------------------------------------------
# trace_event(): gateway-internal, seq is always None
# --------------------------------------------------------------------------


def test_trace_event_has_no_seq_and_is_tagged_tool_gateway() -> None:
    event = TraceEvent(node="gateway", kind="retry_scheduled", detail="attempt 2")

    events = run_and_collect(lambda s: s.trace_event(event))

    (row,) = events
    assert isinstance(row, TraceRow)
    assert row.seq is None
    assert row.source == "tool_gateway"
    assert row.detail == "attempt 2"


def test_trace_event_is_stamped_with_the_next_step_count() -> None:
    seeded = state(step_count=2)
    gateway_event = TraceEvent(node="tool_gateway", kind="retry_scheduled", detail="attempt 2")

    def produce(s: TurnStream) -> None:
        s.step(seeded)  # _last.step_count becomes 2
        s.trace_event(gateway_event)

    rows = [e for e in run_and_collect(produce) if isinstance(e, TraceRow) and e.seq is None]

    assert rows[0].step == 3


def test_trace_event_before_any_step_is_stamped_as_the_first_node() -> None:
    gateway_event = TraceEvent(node="tool_gateway", kind="tool_called", detail="get_project_status -> ok")

    (row,) = run_and_collect(lambda s: s.trace_event(gateway_event))

    assert row.step == 1


# --------------------------------------------------------------------------
# delta() and reset()
# --------------------------------------------------------------------------


def test_delta_and_reset_emit_in_order() -> None:
    def produce(s: TurnStream) -> None:
        s.delta("Hel")
        s.delta("lo")
        s.reset()
        s.delta("Hi")

    events = run_and_collect(produce)

    assert events == [
        TokenEvent(text="Hel"),
        TokenEvent(text="lo"),
        ResetEvent(),
        TokenEvent(text="Hi"),
    ]


# --------------------------------------------------------------------------
# emit(): service-level events pass through untouched
# --------------------------------------------------------------------------


def test_emit_passes_a_prebuilt_event_straight_through() -> None:
    started = TurnStartedEvent(trace_id="run-1", session_id="s1", actor="priya", resumed=False)

    events = run_and_collect(lambda s: s.emit(started))

    assert events == [started]


# --------------------------------------------------------------------------
# flush_events(): catches trace rows the observer never saw
# --------------------------------------------------------------------------


def test_flush_events_catches_up_on_events_added_after_the_run() -> None:
    observed = state(events=(TraceEvent(node="start", kind="node_entered"),), step_count=1)
    filed = observed.evolve(
        events=observed.events
        + (TraceEvent(node="memory", kind="memory_written", detail="write (new fact)"),)
    )

    def produce(s: TurnStream) -> None:
        s.step(observed)  # the observer sees this one
        s.flush_events(filed)  # consolidation appended one more, after the run

    events = run_and_collect(produce)

    trace_rows = [e for e in events if isinstance(e, TraceRow)]
    assert len(trace_rows) == 2
    assert trace_rows[1].kind == "memory_written"
    assert trace_rows[1].seq == 1


def test_flush_events_emits_nothing_new_when_there_is_nothing_new() -> None:
    only = state(events=(TraceEvent(node="start", kind="node_entered"),), step_count=1)

    def produce(s: TurnStream) -> None:
        s.step(only)
        s.flush_events(only)

    events = run_and_collect(produce)

    assert len([e for e in events if isinstance(e, TraceRow)]) == 1


# --------------------------------------------------------------------------
# close(): ends the stream
# --------------------------------------------------------------------------


def test_events_stops_after_close_with_nothing_produced() -> None:
    events = run_and_collect(lambda s: None)

    assert events == []
