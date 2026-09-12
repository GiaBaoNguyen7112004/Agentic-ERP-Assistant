"""The orchestrator: the engine plus the stores, proven together.

Each test below wires a real engine -- real planner over a scripted model,
real tool gateway over its own ``MockErp`` copy -- into the in-memory trace
stores, so a pause is produced by the actual policy pipeline and filed by the
actual composition code, not by a stub agreeing with itself. What is faked is
only the model itself and the composer, which is what a scripted turn needs.

The property under test throughout is the one the future web layer will lean
on: whatever a turn did -- finished, paused, resumed, paused again -- there is
exactly one run record for it, a pause in the queue exactly when a human is
owed a decision, and a decision that settles once no matter who asks again.
"""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentic_erp_assistant.engine.workflow import DENIED_REPLY
from agentic_erp_assistant.engine.orchestrator import (
    ApprovalAlreadySettled,
    RunOrchestrator,
)
from agentic_erp_assistant.erp.mock import DEFAULT_DATASET_PATH, MockErp
from agentic_erp_assistant.llm.schemas import Citation, GroundedAnswer
from agentic_erp_assistant.llm.tools import ToolCallResult
from agentic_erp_assistant.memory.conversation import (
    ConversationMemory,
    InMemoryConversationStore,
)
from agentic_erp_assistant.memory.models import MemoryDecision
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.engine.workflow import WorkflowRuntime
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.tools.gateway import ToolGateway
from agentic_erp_assistant.tools.registry import build_default_registry
from agentic_erp_assistant.trace import InMemoryPauseStore, InMemoryTraceStore

from tests.memory.builders import make_turn

SCOPES = frozenset(
    {
        "project.status.read",
        "project.sprint.read",
        "project.budget.read",
        "project.risk.read",
        "project.risk.write",
    }
)

SNIPPET = EvidenceSnippet(
    source_id="m2-status.md",
    locator="p.2",
    text="Finance module cutover slipped two weeks after the vendor delay.",
)


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class ScriptedModel:
    """Stands in for the provider: returns the choices a real one would make."""

    def __init__(self, *results: ToolCallResult) -> None:
        self.results = list(results)
        self.calls = 0
        self.shown_history: tuple = ()

    def decide(
        self,
        question,
        evidence=(),
        observations=(),
        memories=(),
        history=(),
        *,
        tools=(),
        tool_choice="auto",
    ):
        self.calls += 1
        self.shown_history = tuple(history)
        return self.results[min(self.calls - 1, len(self.results) - 1)]


class FakeRetriever:
    def __init__(self, *snippets: EvidenceSnippet) -> None:
        self.snippets = snippets

    def search(self, query: str, *, limit: int):
        return self.snippets[:limit]


class FakeComposer:
    def __init__(self, answer: GroundedAnswer) -> None:
        self.answer_value = answer

    def answer(self, question: str, evidence, memories=(), history=()):
        return self.answer_value


def called(name: str, **arguments) -> ToolCallResult:
    return ToolCallResult.from_tool_call(tool_name=name, arguments=arguments)


def answered(text: str) -> ToolCallResult:
    return ToolCallResult.from_content(text)


def grounded_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer="M2 slipped two weeks after a vendor delay.",
        citations=[Citation(source_id="m2-status.md", locator="p.2")],
        grounded=True,
        confidence=0.9,
    )


def a_writable_store() -> MockErp:
    """A store over a throwaway copy of the fixture.

    Writes really reach the file now, so no gateway in this file may sit on
    the repo's fixture: an approved ``create_risk`` would edit review material.
    """
    target = Path(tempfile.mkdtemp()) / "project.json"
    target.write_text(
        DEFAULT_DATASET_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return MockErp.load(target)


def an_orchestrator(*decisions, erp: MockErp | None = None) -> tuple[
    RunOrchestrator, InMemoryTraceStore, InMemoryPauseStore, ScriptedModel, MockErp
]:
    """The full stack: real engine, in-memory stores, scripted model.

    Returns everything a test needs to assert with: the orchestrator itself,
    both stores to inspect, the model to count calls on, and the ERP store to
    see what actually executed.
    """
    store = erp or a_writable_store()
    model = ScriptedModel(*decisions)
    runtime = WorkflowRuntime(
        retriever=FakeRetriever(SNIPPET),
        tools=ToolGateway(registry=build_default_registry(store)),
        planner=Planner(model),
        composer=FakeComposer(grounded_answer()),
        sleep=lambda seconds: None,
    )
    traces, pauses = InMemoryTraceStore(), InMemoryPauseStore()
    return RunOrchestrator(runtime, traces, pauses), traces, pauses, model, store


def start(request: str = "Record the risk.") -> AgentState:
    return AgentState(
        request=request,
        actor="bao",
        project_code="atlas",
        trace_id="run-1",
        scopes=SCOPES,
    )


def start_in_session(
    request: str = "Record the risk.", session_id: str = "sess-1"
) -> AgentState:
    return AgentState(
        request=request,
        actor="bao",
        project_code="atlas",
        trace_id="run-1",
        scopes=SCOPES,
        session_id=session_id,
    )


def a_write(title: str = "vendor risk") -> ToolCallResult:
    return called(
        "create_risk", project_id="atlas", title=title, severity="low"
    )


# --------------------------------------------------------------------------
# handle: a turn is filed, whatever it ended as
# --------------------------------------------------------------------------


def test_a_finished_turn_is_filed_as_one_terminal_run() -> None:
    orchestrator, traces, pauses, _, _ = an_orchestrator(
        called("search_project_documents", query="M2 cutover")
    )

    final = orchestrator.handle(start())

    assert final.route == "answer"
    assert traces.runs["run-1"].outcome == "terminal"
    assert traces.load_run("run-1") == final
    assert pauses.pending("run-1") is None


def test_a_paused_turn_files_its_pause_alongside_its_run() -> None:
    """The run record lands first, so an approver opening the queue always
    has the trace behind the request they are being asked to judge."""
    orchestrator, traces, pauses, _, _ = an_orchestrator(a_write())

    final = orchestrator.handle(start())

    assert final.route == "request_approval"
    assert traces.runs["run-1"].outcome == "paused"
    assert pauses.pending("run-1") == final


# --------------------------------------------------------------------------
# resume: a decision settles once, and carries the turn on
# --------------------------------------------------------------------------


def test_an_approved_resume_completes_the_turn_and_empties_the_queue() -> None:
    orchestrator, traces, pauses, _, store = an_orchestrator(
        a_write(), answered("Risk recorded.")
    )
    orchestrator.handle(start())

    final = orchestrator.resume("run-1", approved=True)

    assert final.terminal
    assert final.route == "answer"
    assert traces.runs["run-1"].outcome == "terminal"
    assert pauses.pending("run-1") is None
    assert any(risk.title == "vendor risk" for risk in store.risks)


def test_resume_records_who_decided_it() -> None:
    orchestrator, _, pauses, _, _ = an_orchestrator(a_write(), answered("Risk recorded."))
    orchestrator.handle(start())

    final = orchestrator.resume("run-1", approved=True, decided_by="priya")

    assert pauses.decided_by("run-1") == "priya"
    assert any(
        event.kind == "approval_recorded" and "by priya" in event.detail
        for event in final.events
    )


def test_a_denied_resume_ends_the_turn_without_executing_anything() -> None:
    orchestrator, traces, pauses, _, store = an_orchestrator(a_write())
    orchestrator.handle(start())
    risks_before = len(store.risks)

    final = orchestrator.resume("run-1", approved=False)

    assert (final.route, final.response) == ("refuse", DENIED_REPLY.format(tool="create_risk"))
    assert traces.runs["run-1"].outcome == "terminal"
    assert pauses.pending("run-1") is None
    assert len(store.risks) == risks_before


def test_a_second_resume_raises_and_nothing_runs() -> None:
    """The property the whole approval story rests on: settling twice must not
    execute twice. The first decision wins, the second raises before the
    engine is touched, and the model call count proves it."""
    orchestrator, _, _, model, store = an_orchestrator(
        a_write(), answered("Risk recorded.")
    )
    orchestrator.handle(start())
    orchestrator.resume("run-1", approved=True)
    calls_after_first_resume = model.calls
    risks_after_first_resume = len(store.risks)

    with pytest.raises(ApprovalAlreadySettled):
        orchestrator.resume("run-1", approved=True)

    assert model.calls == calls_after_first_resume
    assert len(store.risks) == risks_after_first_resume


def test_resuming_a_run_nobody_paused_raises() -> None:
    orchestrator, _, _, _, _ = an_orchestrator(answered("Nothing to do."))

    with pytest.raises(ApprovalAlreadySettled):
        orchestrator.resume("no-such-run", approved=True)


def test_a_resumed_turn_can_pause_again_and_waits_anew() -> None:
    """Two writes asked for, one approval given: the second call pauses with
    its own queue entry, and the run record says paused again -- one trace id
    still one run."""
    orchestrator, traces, pauses, _, store = an_orchestrator(
        a_write("first"), a_write("second"), answered("Both recorded.")
    )
    orchestrator.handle(start("Record two risks."))

    final = orchestrator.resume("run-1", approved=True)

    assert final.route == "request_approval"
    assert final.tool_arguments is not None and final.tool_arguments["title"] == "second"
    assert pauses.pending("run-1") == final
    assert traces.runs["run-1"].outcome == "paused"
    assert [risk.title for risk in store.risks].count("first") == 1
    assert [risk.title for risk in store.risks].count("second") == 0

# --------------------------------------------------------------------------
# Memory: recall before the run, consolidation after it, and never a failure
# --------------------------------------------------------------------------


class RecordingMemory:
    """A TurnMemoryPort that hands back what it was built with, and counts."""

    def __init__(self, *, recalled=(), decisions=()) -> None:
        self.recalled = recalled
        self.decisions = decisions
        self.recalls: list[AgentState] = []
        self.consolidations: list[AgentState] = []
        self.evicted_seen: list[tuple] = []

    def recall(self, state):
        self.recalls.append(state)
        return self.recalled

    def consolidate(self, state, *, evicted=()):
        self.consolidations.append(state)
        self.evicted_seen.append(tuple(evicted))
        return self.decisions


class BrokenMemory:
    def recall(self, state):
        raise RuntimeError("the store is unreachable")

    def consolidate(self, state, *, evicted=()):
        raise RuntimeError("the store is unreachable")


def remembered() -> MemoryRecord:
    return MemoryRecord(
        memory_id="mem-1",
        kind="preference",
        key="reply_language",
        statement="Prefers replies written in Vietnamese.",
        project_code="atlas",
        required_scope="project.docs.read",
        actor="bao",
        session_id="sess-1",
        recorded_in_run="run-0",
        recorded_at=datetime(2026, 9, 8, tzinfo=UTC),
        confidence=0.9,
    )


def with_memory(memory, *decisions, erp: MockErp | None = None):
    orchestrator, traces, pauses, model, store = an_orchestrator(*decisions, erp=erp)
    orchestrator.memory = memory
    return orchestrator, traces, pauses, model, store


def test_no_memory_is_a_complete_configuration() -> None:
    """A replay, an evaluation case and a one-shot script have no conversation
    to remember anything for."""
    orchestrator, _, _, _, _ = an_orchestrator(answered("Nothing to do."))

    assert orchestrator.memory is None
    assert orchestrator.handle(start()).terminal is True


def test_recall_fills_the_state_before_the_engine_sees_it() -> None:
    """Not the caller's job: a caller that supplied memories would be a second
    place recall could happen."""
    memory = RecordingMemory(recalled=(remembered(),))
    orchestrator, _, _, _, _ = with_memory(memory, answered("Xin chào."))

    final = orchestrator.handle(start())

    assert final.memories == (remembered(),)
    assert memory.recalls[0].memories == ()


def test_a_recalled_turn_says_so_in_its_trace() -> None:
    orchestrator, traces, _, _, _ = with_memory(
        RecordingMemory(recalled=(remembered(),)), answered("Xin chào.")
    )

    orchestrator.handle(start())

    kinds = [event.kind for event in traces.load_run("run-1").events]
    assert "memory_recalled" in kinds


def test_recalling_nothing_adds_no_event() -> None:
    """Most turns recall nothing; an event per turn saying so would bury the
    ones that did."""
    orchestrator, traces, _, _, _ = with_memory(
        RecordingMemory(), answered("Nothing to do.")
    )

    orchestrator.handle(start())

    kinds = [event.kind for event in traces.load_run("run-1").events]
    assert "memory_recalled" not in kinds


def test_consolidation_sees_the_finished_turn() -> None:
    memory = RecordingMemory()
    orchestrator, _, _, _, _ = with_memory(memory, answered("Nothing to do."))

    orchestrator.handle(start())

    assert len(memory.consolidations) == 1
    assert memory.consolidations[0].terminal is True


def test_a_paused_turn_is_never_consolidated() -> None:
    """Its central question -- will a human approve this? -- has no answer yet,
    and storing facts from it would remember a decision nobody has made."""
    memory = RecordingMemory()
    orchestrator, _, _, _, _ = with_memory(memory, a_write())

    orchestrator.handle(start())

    assert memory.consolidations == []


def test_a_resumed_turn_is_consolidated_once_it_ends() -> None:
    memory = RecordingMemory()
    orchestrator, _, _, _, _ = with_memory(
        memory, a_write(), answered("Risk recorded.")
    )
    orchestrator.handle(start())

    orchestrator.resume("run-1", approved=True)

    assert len(memory.consolidations) == 1
    assert memory.consolidations[0].terminal is True


def test_resuming_does_not_recall_again() -> None:
    """An approver's decision was made against one set of background; executing
    it against another would make the two disagree."""
    memory = RecordingMemory()
    orchestrator, _, _, _, _ = with_memory(
        memory, a_write(), answered("Risk recorded.")
    )
    orchestrator.handle(start())

    orchestrator.resume("run-1", approved=True)

    assert len(memory.recalls) == 1


def test_what_was_written_and_what_was_refused_are_separate_events() -> None:
    """A run that refused four proposals and kept one is a run where the policy
    worked, and it must not read the same as one that stored five."""
    memory = RecordingMemory(
        decisions=(
            MemoryDecision(decision="write", reason="new preference"),
            MemoryDecision(
                decision="reject",
                rejection="instruction_like",
                reason="reads as standing instruction",
            ),
        )
    )
    orchestrator, traces, _, _, _ = with_memory(memory, answered("Nothing to do."))

    orchestrator.handle(start())

    events = traces.load_run("run-1").events
    assert [e.kind for e in events if e.kind.startswith("memory_")] == [
        "memory_written",
        "memory_rejected",
    ]
    assert "instruction_like" in next(
        e.detail for e in events if e.kind == "memory_rejected"
    )


def test_the_memory_events_are_in_the_run_record_that_gets_filed() -> None:
    """Consolidation happens before the record is written, so its events are in
    this trace rather than in the next one."""
    orchestrator, traces, _, _, _ = with_memory(
        RecordingMemory(decisions=(MemoryDecision(decision="write"),)),
        answered("Nothing to do."),
    )

    orchestrator.handle(start())

    assert "memory_written" in [
        event.kind for event in traces.runs["run-1"].state.events
    ]


def test_a_memory_layer_that_is_down_cannot_fail_a_turn() -> None:
    """A question answered correctly and reported as an error is the one
    outcome worse than answering it with no background."""
    orchestrator, traces, _, _, _ = with_memory(BrokenMemory(), answered("Answered."))

    final = orchestrator.handle(start())

    assert final.terminal is True
    assert final.response == "Answered."
    assert traces.runs["run-1"].outcome == "terminal"


# --------------------------------------------------------------------------
# History: recalled before the run, recorded after it is filed, promoted
# during consolidation
# --------------------------------------------------------------------------


class RecordingConversation:
    """A SessionHistoryPort that hands back what it was built with, and counts."""

    def __init__(self, *, recalled=(), evicted=()) -> None:
        self.recalled = tuple(recalled)
        self.evicted_turns = tuple(evicted)
        self.recalls: list[AgentState] = []
        self.records: list[tuple[AgentState, datetime]] = []
        self.eviction_checks: list[AgentState] = []
        self.promotions: list[tuple[tuple[str, ...], str]] = []

    def recall(self, state):
        self.recalls.append(state)
        return self.recalled

    def record(self, state, *, started_at):
        self.records.append((state, started_at))

    def evicted(self, state):
        self.eviction_checks.append(state)
        return self.evicted_turns

    def promoted(self, turns, *, run):
        self.promotions.append((tuple(turn.trace_id for turn in turns), run))


class BrokenConversation:
    def recall(self, state):
        raise RuntimeError("the store is unreachable")

    def record(self, state, *, started_at):
        raise RuntimeError("the store is unreachable")

    def evicted(self, state):
        raise RuntimeError("the store is unreachable")

    def promoted(self, turns, *, run):
        raise RuntimeError("the store is unreachable")


class BrokenTraceStore:
    """A TraceStore whose save_run raises, for ordering assertions."""

    def save_run(self, record):
        raise RuntimeError("the trace store is down")


def prior_turn(trace_id: str = "run-0") -> ConversationTurn:
    return make_turn(trace_id=trace_id, actor="bao", request="Record the risk.")


def with_conversation(
    conversation, *decisions, memory=None, erp: MockErp | None = None
):
    orchestrator, traces, pauses, model, store = an_orchestrator(*decisions, erp=erp)
    if memory is not None:
        orchestrator.memory = memory
    orchestrator.conversation = conversation
    return orchestrator, traces, pauses, model, store


def test_history_is_on_the_state_before_the_engine_sees_it() -> None:
    conversation = RecordingConversation(recalled=(prior_turn(),))
    orchestrator, traces, _, model, _ = with_conversation(
        conversation, answered("On track.")
    )

    final = orchestrator.handle(start_in_session())

    assert conversation.recalls[0].history == ()  # nothing had arrived yet
    assert model.shown_history == (prior_turn(),)  # the engine was shown it
    assert final.history == (prior_turn(),)
    assert traces.runs["run-1"].state.history == (prior_turn(),)


def test_no_session_means_no_history_read_and_no_turn_recorded() -> None:
    conversation = RecordingConversation(recalled=(prior_turn(),))
    orchestrator, _, _, _, _ = with_conversation(conversation, answered("On track."))

    orchestrator.handle(start())

    assert conversation.recalls == []
    assert conversation.records == []


def test_the_turn_is_recorded_after_the_run_is_filed() -> None:
    """The trace store's contract is that an unsaved run did not happen, so a
    turn whose run record could not be filed joins no window either."""
    conversation = RecordingConversation()
    orchestrator, _, _, _, _ = with_conversation(conversation, answered("On track."))
    orchestrator.traces = BrokenTraceStore()

    with pytest.raises(RuntimeError):
        orchestrator.handle(start_in_session())

    assert conversation.records == []


def test_a_paused_turn_is_recorded_with_no_reply() -> None:
    """The session's next request must see the wait in the window, not a
    re-request of the write."""
    conversation = ConversationMemory(
        store=InMemoryConversationStore(), model="gpt-4o"
    )
    orchestrator, _, _, _, _ = with_conversation(conversation, a_write())

    orchestrator.handle(start_in_session())

    kept = conversation.store.recent("sess-1", actor="bao", limit=10)
    assert len(kept) == 1
    assert kept[0].paused is True
    assert kept[0].response is None
    assert kept[0].tool_name == "create_risk"


def test_resume_records_the_same_trace_id_with_the_reply() -> None:
    conversation = ConversationMemory(
        store=InMemoryConversationStore(), model="gpt-4o"
    )
    orchestrator, _, _, _, _ = with_conversation(
        conversation, a_write(), answered("Risk recorded.")
    )
    orchestrator.handle(start_in_session())

    orchestrator.resume("run-1", approved=True)

    kept = conversation.store.recent("sess-1", actor="bao", limit=10)
    assert [t.trace_id for t in kept] == ["run-1"]
    assert kept[0].response.startswith("Risk recorded.")  # plus its sources trailer
    assert kept[0].settled is True


def test_resume_does_not_recall_history_again() -> None:
    """The approver's decision was judged against one window; executing it
    against another would make the two disagree."""
    conversation = RecordingConversation(recalled=(prior_turn(),))
    orchestrator, _, _, _, _ = with_conversation(
        conversation, a_write(), answered("Risk recorded.")
    )
    orchestrator.handle(start_in_session())

    orchestrator.resume("run-1", approved=True)

    assert len(conversation.recalls) == 1


def test_a_broken_conversation_store_cannot_fail_a_turn() -> None:
    orchestrator, traces, _, _, _ = with_conversation(
        BrokenConversation(), answered("Answered.")
    )

    final = orchestrator.handle(start_in_session())

    assert final.terminal is True
    assert final.response == "Answered."
    assert traces.runs["run-1"].outcome == "terminal"


def test_history_recalled_is_in_the_run_record_only_when_non_empty() -> None:
    """Most turns recall nothing; an event per turn saying so would bury the
    ones that did."""
    empty, empty_traces, _, _, _ = with_conversation(
        RecordingConversation(), answered("On track.")
    )
    empty.handle(start_in_session())
    kinds = [event.kind for event in empty_traces.runs["run-1"].state.events]
    assert "history_recalled" not in kinds

    full, full_traces, _, _, _ = with_conversation(
        RecordingConversation(recalled=(prior_turn(),)), answered("On track.")
    )
    full.handle(start_in_session())
    events = full_traces.runs["run-1"].state.events
    assert "history_recalled" in [event.kind for event in events]
    event = next(event for event in events if event.kind == "history_recalled")
    assert event.node == "history"
    assert "1 prior turn" in event.detail


def test_evicted_turns_are_handed_to_consolidation_and_then_marked_promoted() -> None:
    evicted = (prior_turn("run-0"),)
    memory = RecordingMemory(decisions=(MemoryDecision(decision="write"),))
    conversation = RecordingConversation(evicted=evicted)
    orchestrator, traces, _, _, _ = with_conversation(
        conversation, answered("On track."), memory=memory
    )

    orchestrator.handle(start_in_session())

    assert memory.evicted_seen == [evicted]
    assert conversation.promotions == [(("run-0",), "run-1")]
    kinds = [event.kind for event in traces.runs["run-1"].state.events]
    assert "history_promoted" in kinds


def test_a_consolidation_that_raises_leaves_the_turns_unpromoted() -> None:
    """Marking happens only when consolidation returned: a raise leaves the
    turns in the store, retried on the session's next turn."""
    conversation = RecordingConversation(evicted=(prior_turn("run-0"),))
    orchestrator, traces, _, _, _ = with_conversation(
        conversation, answered("On track."), memory=BrokenMemory()
    )

    final = orchestrator.handle(start_in_session())

    assert final.terminal is True
    assert conversation.promotions == []
    kinds = [event.kind for event in traces.runs["run-1"].state.events]
    assert "history_promoted" not in kinds


def test_a_paused_turn_promotes_nothing() -> None:
    """Consolidation is skipped for a pause, so eviction is too: the turns
    this pause would push out stay in the window until the turn resumes."""
    memory = RecordingMemory()
    conversation = RecordingConversation(evicted=(prior_turn("run-0"),))
    orchestrator, _, _, _, _ = with_conversation(
        conversation, a_write(), memory=memory
    )

    orchestrator.handle(start_in_session())

    assert memory.consolidations == []
    assert conversation.eviction_checks == []
    assert conversation.promotions == []


def test_the_memory_layer_can_be_absent_while_history_is_present() -> None:
    """``memory=None, conversation=...`` is a legitimate configuration: turns
    are shown and recorded, and nothing is ever promoted."""
    conversation = RecordingConversation(
        recalled=(prior_turn(),), evicted=(prior_turn("run-0"),)
    )
    orchestrator, traces, _, _, _ = with_conversation(
        conversation, answered("On track.")
    )
    assert orchestrator.memory is None

    final = orchestrator.handle(start_in_session())

    assert final.history == (prior_turn(),)
    assert len(conversation.records) == 1
    assert conversation.eviction_checks == []
    assert conversation.promotions == []
    kinds = [event.kind for event in traces.runs["run-1"].state.events]
    assert "history_recalled" in kinds
    assert "history_promoted" not in kinds
