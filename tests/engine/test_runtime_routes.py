"""Every route a turn can take, driven through the real engine.

The checklist for this unit names ten paths, and each has a test below: a
document answer, a refusal on empty evidence, a read tool call, a write that
pauses and resumes to success, a write that resumes to denial, a throttled call
that retries once and succeeds, a throttled call that exhausts its budget, an
empty request, an unsupported request, and a run forced through the step-budget
guard.

What is real here and what is faked is a deliberate choice. The **planner is
real** -- a scripted :class:`ToolCallResult` goes through
:class:`~agentic_erp_assistant.reasoning.planner.Planner`, so the
function-calling mapping is exercised rather than assumed. The **tool gateway is
real** for the approval paths, running against its own
:class:`~agentic_erp_assistant.erp.mock.MockErp` store, so a pause is produced
by the actual policy pipeline instead of by a stub agreeing with itself; a
scripted gateway stands in only where a specific status has to be forced. The
retriever and the composer are fakes: their real implementations are a RAG index
and a provider call, and neither is what this file is about.

Nothing here touches the network, and nothing sleeps.
"""

import tempfile
from pathlib import Path

import pytest

from agentic_erp_assistant.erp.mock import DEFAULT_DATASET_PATH, MockErp
from agentic_erp_assistant.llm.schemas import Citation, GroundedAnswer
from agentic_erp_assistant.llm.tools import ToolCallResult
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.engine.nodes import NO_EVIDENCE_REPLY, SOURCES_PREFIX
from agentic_erp_assistant.engine.workflow import (
    DENIED_REPLY,
    NotPaused,
    UnroutableState,
    WorkflowRuntime,
    is_paused,
)
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.state.tool_request import ToolRequest
from agentic_erp_assistant.tools.gateway import ToolGateway
from agentic_erp_assistant.tools.registry import build_default_registry

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
        return self.results[min(self.calls - 1, len(self.results) - 1)]


class FakeRetriever:
    def __init__(self, *snippets: EvidenceSnippet) -> None:
        self.snippets = snippets

    def search(self, query: str, *, limit: int):
        return self.snippets[:limit]


class FakeComposer:
    def __init__(self, answer: GroundedAnswer) -> None:
        self.answer_value = answer

    def answer(self, question: str, evidence, memories=(), history=(), observations=()):
        return self.answer_value


class ScriptedGateway:
    """A gateway forced to return particular statuses, for the paths where the
    real one would need a clock to reproduce them."""

    def __init__(self, *outcomes: ToolOutcome) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ToolRequest] = []

    def execute(self, request: ToolRequest) -> ToolOutcome:
        self.requests.append(request)
        return self.outcomes[min(len(self.requests) - 1, len(self.outcomes) - 1)]


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
    the repo's fixture: an approved ``create_risk`` in a route test would edit
    review material. Each store gets its own copy, so one route test's write is
    invisible to the next one's read.
    """
    target = Path(tempfile.mkdtemp()) / "project.json"
    target.write_text(
        DEFAULT_DATASET_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return MockErp.load(target)


def runtime(
    *decisions: ToolCallResult,
    erp: MockErp | None = None,
    tools=None,
    retriever=None,
    composer=None,
    **overrides,
) -> WorkflowRuntime:
    """The engine, wired to a scripted model and whichever gateway a case needs."""
    settings = {
        "retriever": retriever or FakeRetriever(SNIPPET),
        "tools": tools
        or ToolGateway(registry=build_default_registry(erp or a_writable_store())),
        "planner": Planner(ScriptedModel(*decisions)),
        "composer": composer or FakeComposer(grounded_answer()),
        "sleep": lambda seconds: None,
    }
    settings.update(overrides)
    return WorkflowRuntime(**settings)


def start(request: str = "How is M2 tracking?") -> AgentState:
    return AgentState(
        request=request,
        actor="bao",
        project_code="atlas",
        trace_id="run-1",
        scopes=SCOPES,
    )


def kinds(state: AgentState) -> list[str]:
    return [event.kind for event in state.events]


# --------------------------------------------------------------------------
# 1-2. Documents
# --------------------------------------------------------------------------


def test_a_document_question_is_answered_with_its_sources() -> None:
    engine = runtime(called("search_project_documents", query="M2 cutover"))

    final = engine.run(start())

    assert final.route == "answer"
    assert final.terminal
    assert final.response.endswith(f"{SOURCES_PREFIX}[m2-status.md#p.2]")
    assert final.evidence == (SNIPPET,)
    assert kinds(final).count("evidence_retrieved") == 1


def test_a_document_question_with_no_passages_refuses_instead_of_guessing() -> None:
    engine = runtime(
        called("search_project_documents", query="M2 cutover"),
        retriever=FakeRetriever(),
    )

    final = engine.run(start())

    assert (final.route, final.failure) == ("refuse", "insufficient_evidence")
    assert final.response == NO_EVIDENCE_REPLY
    assert final.evidence == ()


# --------------------------------------------------------------------------
# 3. A read tool call
# --------------------------------------------------------------------------


def test_a_read_tool_runs_and_its_result_is_answered_from() -> None:
    """Two model turns, not one: the tool result goes back to the planner
    before the turn is declared done."""
    engine = runtime(
        called("list_risks", project_id="atlas"),
        answered("Two risks are open on atlas."),
    )

    final = engine.run(start("What could go wrong on atlas?"))

    assert final.route == "answer"
    assert final.observations[0].status == "ok"
    assert final.response.startswith("Two risks are open on atlas.")
    assert SOURCES_PREFIX in final.response
    assert kinds(final).count("tool_called") == 1


def test_a_read_the_actor_is_not_entitled_to_fails_the_turn() -> None:
    """The scope check lives at the gateway, and the engine reports what it
    said rather than deciding for itself."""
    engine = runtime(called("list_risks", project_id="atlas"))
    unentitled = start("What could go wrong?").evolve(scopes=frozenset({"nothing"}))

    final = engine.run(unentitled)

    assert (final.route, final.failure) == ("fail", "tool_failure")
    assert "denied" in (final.error_detail or "")


# --------------------------------------------------------------------------
# 4-5. The write, and the human in front of it
# --------------------------------------------------------------------------


def create_risk_call() -> ToolCallResult:
    return called(
        "create_risk",
        project_id="atlas",
        title="Vendor may miss the cutover window",
        severity="high",
    )


def test_a_write_pauses_before_anything_changes() -> None:
    erp = a_writable_store()
    before = len(erp.risks)
    engine = runtime(create_risk_call(), erp=erp)

    paused = engine.run(start("Record a risk about the vendor."))

    assert is_paused(paused)
    assert not paused.terminal
    assert (paused.tool_name, paused.tool_mutating) == ("create_risk", True)
    assert paused.tool_arguments is None or "project_id" in paused.tool_arguments
    assert "approval_requested" in kinds(paused)
    assert len(erp.risks) == before


def test_an_approved_write_runs_and_the_turn_finishes() -> None:
    erp = a_writable_store()
    before = len(erp.risks)
    engine = runtime(
        create_risk_call(),
        answered("Recorded the vendor risk."),
        erp=erp,
    )
    paused = engine.run(start("Record a risk about the vendor."))

    final = engine.resume_approval(paused, approved=True)

    assert final.route == "answer"
    assert final.approval == "approved"
    assert len(erp.risks) == before + 1
    assert "approval_recorded" in kinds(final)
    assert final.observations[-1].status == "ok"


def test_a_denied_write_ends_the_turn_and_changes_nothing() -> None:
    erp = a_writable_store()
    before = len(erp.risks)
    engine = runtime(create_risk_call(), erp=erp)
    paused = engine.run(start("Record a risk about the vendor."))

    final = engine.resume_approval(paused, approved=False)

    assert (final.route, final.approval) == ("refuse", "denied")
    assert final.response == DENIED_REPLY.format(tool="create_risk")
    assert final.failure == "none", "a denial is policy working, not an outage"
    assert len(erp.risks) == before


def test_a_turn_nobody_asked_about_cannot_have_an_approval_recorded() -> None:
    engine = runtime(answered("Nothing to do."))
    final = engine.run(start())

    with pytest.raises(NotPaused):
        engine.resume_approval(final, approved=True)


def test_the_paused_state_carries_everything_a_resume_needs() -> None:
    """The pause can outlive the process, so it has to be complete on its own."""
    engine = runtime(create_risk_call())

    paused = engine.run(start("Record a risk about the vendor."))
    restored = AgentState.model_validate(paused.model_dump())

    assert restored == paused
    assert is_paused(restored)


# --------------------------------------------------------------------------
# 6-7. Rate limits
# --------------------------------------------------------------------------


def throttled(seconds: float = 0.25) -> ToolOutcome:
    return ToolOutcome(
        tool_name="list_risks",
        status="rate_limited",
        error="budget spent",
        retry_after_seconds=seconds,
    )


def succeeded() -> ToolOutcome:
    return ToolOutcome(
        tool_name="list_risks",
        status="ok",
        summary="2 open risks",
        source_ids=("risk:R-1",),
    )


def test_a_throttled_call_waits_once_and_then_succeeds() -> None:
    waited: list[float] = []
    gateway = ScriptedGateway(throttled(), succeeded())
    engine = runtime(
        called("list_risks", project_id="atlas"),
        answered("Two risks are open."),
        tools=gateway,
        sleep=waited.append,
    )

    final = engine.run(start("What could go wrong?"))

    assert final.route == "answer"
    assert final.retry_count == 1
    assert waited == [0.25]
    assert len(gateway.requests) == 2
    assert "retry_scheduled" in kinds(final)
    assert "rate_limited" in kinds(final)


def test_a_throttle_that_outlives_the_budget_fails_loudly() -> None:
    gateway = ScriptedGateway(throttled())
    engine = runtime(
        called("list_risks", project_id="atlas"),
        tools=gateway,
        tool_retry_budget=1,
    )

    final = engine.run(start("What could go wrong?"))

    assert (final.route, final.failure) == ("fail", "tool_failure")
    assert final.retry_count == 1
    assert len(gateway.requests) == 2
    assert "rate limited" in (final.error_detail or "")


# --------------------------------------------------------------------------
# 8-9. Nothing to answer, and nothing that should be answered
# --------------------------------------------------------------------------


def test_an_empty_request_asks_back() -> None:
    engine = runtime(called("ask_clarification", question="Which milestone?"))

    final = engine.run(start("   "))

    assert (final.route, final.terminal) == ("clarify", True)
    assert final.response
    assert final.failure == "none"


def test_an_under_specified_request_asks_back() -> None:
    engine = runtime(called("ask_clarification", question="Which milestone?"))

    final = engine.run(start("how is it going?"))

    assert final.route == "clarify"
    assert final.response == "Which milestone?"


def test_an_unsupported_request_is_refused_and_not_failed() -> None:
    engine = runtime(
        called("refuse", reason="That is not a project-delivery question.")
    )

    final = engine.run(start("Write me a poem about sprint 12."))

    assert (final.route, final.failure) == ("refuse", "none")
    assert final.response == "That is not a project-delivery question."


# --------------------------------------------------------------------------
# 10. The loop guard
# --------------------------------------------------------------------------


def test_a_planner_that_never_settles_is_stopped_by_the_step_budget() -> None:
    """The realistic hang: a model that keeps calling the same tool because the
    result never satisfies it. Nothing terminates, so the engine does."""
    gateway = ScriptedGateway(succeeded())
    engine = runtime(
        called("list_risks", project_id="atlas"),
        tools=gateway,
        max_steps=6,
    )

    final = engine.run(start("What could go wrong?"))

    assert (final.route, final.failure) == ("fail", "max_steps_exceeded")
    assert final.terminal
    assert final.step_count == 6
    assert "run_failed" in kinds(final)
    assert "max_steps=6" in (final.error_detail or "")


def test_a_node_that_returns_an_unchanged_state_is_stopped_too() -> None:
    """The other shape of the same bug: a node that ends nothing. Forced with
    an injected table, because no real node can produce it."""
    engine = runtime(
        answered("hi"),
        nodes={None: lambda state: state, "think": lambda state: state},
        max_steps=3,
    )

    final = engine.run(start())

    assert final.failure == "max_steps_exceeded"
    assert final.step_count == 3


def test_the_guard_leaves_the_whole_trace_attached() -> None:
    """A run that ends in a hang with no evidence is the failure this replaces."""
    engine = runtime(
        answered("hi"),
        nodes={None: lambda state: state, "think": lambda state: state},
        max_steps=3,
    )

    final = engine.run(start())

    assert kinds(final).count("node_entered") == 3
    assert kinds(final).count("node_exited") == 3
    assert kinds(final)[-1] == "run_failed"


def test_a_route_with_no_node_is_a_bug_and_says_so() -> None:
    engine = runtime(answered("hi"), nodes={})

    with pytest.raises(UnroutableState, match="no node handles"):
        engine.run(start())


# --------------------------------------------------------------------------
# What every run leaves behind
# --------------------------------------------------------------------------


def test_every_run_ends_terminal_or_paused_and_never_in_between() -> None:
    cases = [
        (runtime(called("search_project_documents", query="M2")), start()),
        (runtime(create_risk_call()), start("Record a risk.")),
        (
            runtime(called("refuse", reason="Out of scope.")),
            start("Write me a poem."),
        ),
    ]

    for engine, state in cases:
        final = engine.run(state)
        assert final.terminal or is_paused(final)


def test_a_finished_turn_carries_a_response_or_a_failure() -> None:
    engine = runtime(called("search_project_documents", query="M2"))

    final = engine.run(start())

    assert final.response is not None or final.failure != "none"


def test_the_trace_records_entering_and_leaving_every_node() -> None:
    engine = runtime(called("search_project_documents", query="M2"))

    final = engine.run(start())

    assert kinds(final).count("node_entered") == final.step_count
    assert kinds(final).count("node_exited") == final.step_count


def test_the_step_counter_counts_node_executions_and_not_transitions() -> None:
    engine = runtime(
        called("list_risks", project_id="atlas"),
        answered("Two risks are open."),
    )

    final = engine.run(start("What could go wrong?"))

    assert final.step_count == 3  # think, call_tool, think
