"""``WorkflowRuntime.observer`` -- a live view of each step.

The observer is a read-only tap: it must see one state per node execution, in
the order the engine produced them, and its own failures must never touch the
turn it is watching.
"""

from agentic_erp_assistant.engine.workflow import WorkflowRuntime
from agentic_erp_assistant.llm.schemas import Citation, GroundedAnswer
from agentic_erp_assistant.llm.tools import ToolCallResult
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet

SCOPES = frozenset({"project.status.read"})

SNIPPET = EvidenceSnippet(
    source_id="m2-status.md",
    locator="p.2",
    text="Finance module cutover slipped two weeks after the vendor delay.",
)


class ScriptedModel:
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
        allow_tools=True,
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

    def answer(self, question: str, evidence, memories=(), history=()):
        return self.answer_value


class FailingTools:
    def execute(self, request):
        raise AssertionError("no tool call expected in this test")


def grounded_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer="M2 slipped two weeks after a vendor delay.",
        citations=[Citation(source_id="m2-status.md", locator="p.2")],
        grounded=True,
        confidence=0.9,
    )


def called(name: str, **arguments) -> ToolCallResult:
    return ToolCallResult.from_tool_call(tool_name=name, arguments=arguments)


def start(request: str = "How is M2 tracking?") -> AgentState:
    return AgentState(
        request=request,
        actor="bao",
        project_code="atlas",
        trace_id="run-1",
        scopes=SCOPES,
    )


def runtime(*decisions: ToolCallResult, **overrides) -> WorkflowRuntime:
    settings = {
        "retriever": FakeRetriever(SNIPPET),
        "tools": FailingTools(),
        "planner": Planner(ScriptedModel(*decisions)),
        "composer": FakeComposer(grounded_answer()),
        "sleep": lambda seconds: None,
    }
    settings.update(overrides)
    return WorkflowRuntime(**settings)


def test_observer_receives_one_state_per_node_in_order_ending_on_the_returned_state() -> (
    None
):
    seen: list[AgentState] = []
    engine = runtime(
        called("search_project_documents", query="M2 cutover"),
        observer=seen.append,
    )

    final = engine.run(start())

    assert final.terminal
    # `think` routes to `retrieve_project_documents`, then `retrieve_and_answer`
    # ends the turn: two node executions, observed in that order.
    assert [s.route for s in seen] == ["retrieve_project_documents", "answer"]
    assert seen[-1] is final


def test_a_raising_observer_does_not_change_the_outcome() -> None:
    def boom(state: AgentState) -> None:
        raise RuntimeError("screen went away")

    engine = runtime(
        called("search_project_documents", query="M2 cutover"),
        observer=boom,
    )

    final = engine.run(start())

    assert final.terminal
    assert final.route == "answer"


def test_an_observer_that_raised_once_is_dropped_for_the_rest_of_the_run() -> None:
    calls = {"n": 0}

    def boom_once(state: AgentState) -> None:
        calls["n"] += 1
        raise RuntimeError("boom")

    engine = runtime(
        called("search_project_documents", query="M2 cutover"),
        observer=boom_once,
    )

    engine.run(start())

    assert calls["n"] == 1


def test_out_of_budget_state_is_observed_before_it_is_returned() -> None:
    seen: list[AgentState] = []

    engine = WorkflowRuntime(
        retriever=FakeRetriever(SNIPPET),
        tools=FailingTools(),
        planner=Planner(ScriptedModel(called("search_project_documents", query="x"))),
        composer=FakeComposer(grounded_answer()),
        max_steps=2,
        nodes={None: lambda state: state},
        observer=seen.append,
    )

    final = engine.run(start())

    assert final.failure == "max_steps_exceeded"
    assert seen[-1] is final


def test_resume_approval_denied_observes_the_terminal_refusal() -> None:
    from agentic_erp_assistant.engine.transitions import advance

    seen: list[AgentState] = []
    engine = runtime(observer=seen.append)

    paused = advance(
        start("Record a risk."),
        "request_approval",
        tool_name="create_risk",
        tool_arguments={"project_id": "atlas"},
        tool_mutating=True,
        approval="pending",
    )

    final = engine.resume_approval(paused, approved=False)

    assert final.terminal
    assert final.route == "refuse"
    assert seen[-1] is final
    # the intermediate `decided` state (approval recorded) was observed too
    assert any(s.approval == "denied" for s in seen)
