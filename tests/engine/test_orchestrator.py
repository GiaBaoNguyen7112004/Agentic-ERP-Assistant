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
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.engine.workflow import WorkflowRuntime
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.tools.gateway import ToolGateway
from agentic_erp_assistant.tools.registry import build_default_registry
from agentic_erp_assistant.trace import InMemoryPauseStore, InMemoryTraceStore

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

    def decide(self, question, evidence=(), observations=(), *, tools=()):
        self.calls += 1
        return self.results[min(self.calls - 1, len(self.results) - 1)]


class FakeRetriever:
    def __init__(self, *snippets: EvidenceSnippet) -> None:
        self.snippets = snippets

    def search(self, query: str, *, limit: int):
        return self.snippets[:limit]


class FakeComposer:
    def __init__(self, answer: GroundedAnswer) -> None:
        self.answer_value = answer

    def answer(self, question: str, evidence):
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
        request=request, actor="bao", trace_id="run-1", scopes=SCOPES
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