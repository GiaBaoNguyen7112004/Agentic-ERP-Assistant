"""ADR 0019: a decision that repeats an already-successful call is refused a
re-run, and the planner is re-asked once with tools withheld instead.

A11 in the manual walkthrough (docs/manual-test.md) is the live shape this
guards against: after an approved ``create_risk`` executed, the planner kept
calling ``list_risks`` -- a call that had already succeeded once, before the
write, as the contract's own "check first" step -- until the step budget
ended the run with no reply. Nothing here simulates the whole approval pause;
what matters is one fact already sitting in ``observations`` as ``ok`` and
whether ``think()`` lets a decision repeat it.

Two levels: ``think()`` in isolation (the mechanics -- which call got
withheld, what the trace says, what fails and how) and the real engine
end-to-end (the step count, and the case this guard must *not* touch: two
different writes in the same turn, proven already by
``tests/engine/test_orchestrator.py::test_a_resumed_turn_can_pause_again_and_waits_anew``
and re-asserted here as a boundary).
"""

from agentic_erp_assistant.reasoning.decision import ReasoningDecision
from agentic_erp_assistant.engine.nodes import GraphNodes
from agentic_erp_assistant.engine.workflow import WorkflowRuntime
from agentic_erp_assistant.llm.tools import ToolCallResult
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.state.tool_request import ToolRequest

SCOPES = frozenset({"project.risk.read", "project.risk.write"})


# --------------------------------------------------------------------------
# think() in isolation
# --------------------------------------------------------------------------


class FakePlanner:
    """Returns decisions in sequence, and records the tool_choice each call saw."""

    def __init__(self, *decisions: ReasoningDecision) -> None:
        self.decisions = list(decisions)
        self.tool_choice_seen: list[str] = []

    def plan(
        self,
        state: AgentState,
        *,
        tool_choice: str = "auto",
        withhold: frozenset[str] = frozenset(),
    ) -> ReasoningDecision:
        self.tool_choice_seen.append(tool_choice)
        index = len(self.tool_choice_seen) - 1
        return self.decisions[min(index, len(self.decisions) - 1)]


class FakeGateway:
    """Preflight always says "needs a human" -- the only outcome
    ``request_approval`` decisions in this file's fixtures ever propose."""

    def preflight(self, request: ToolRequest) -> ToolOutcome:
        return ToolOutcome(
            tool_name=request.tool_name,
            arguments_summary=f"{request.tool_name}(…)",
            status="approval_required",
            error=f"{request.tool_name} needs a human",
        )

    def execute(self, request: ToolRequest) -> ToolOutcome:  # pragma: no cover
        raise AssertionError("think() calling execute was not expected here")


def nodes(planner: FakePlanner) -> GraphNodes:
    return GraphNodes(
        retriever=None,  # not exercised by any test here
        tools=FakeGateway(),
        planner=planner,
        composer=None,  # not exercised by any test here
    )


def state(**changes) -> AgentState:
    base = AgentState(
        request="Record a medium risk on orion: legal has not signed off.",
        actor="orion.lead",
        project_code="orion",
        trace_id="run-1",
        scopes=SCOPES,
    )
    return base.evolve(**changes) if changes else base


def a_successful_list_risks() -> ToolOutcome:
    return ToolOutcome(
        tool_name="list_risks",
        arguments_summary="list_risks(project_id=orion)",
        status="ok",
        summary="No open risks recorded for orion.",
        source_ids=("project-orion",),
    )


def called(name: str, **arguments) -> ReasoningDecision:
    return ReasoningDecision(
        route="call_tool",
        confidence=0.5,
        required_tool=name,
        tool_arguments=arguments,
    )


def answered(text: str) -> ReasoningDecision:
    return ReasoningDecision(route="answer", confidence=0.5, message=text)


def kinds(result: AgentState) -> list[str]:
    return [event.kind for event in result.events]


def test_a_genuinely_new_call_after_a_success_is_not_forced() -> None:
    """A read that has not been made before is not a repeat -- nothing about
    ADR 0019 fires just because *something* already succeeded this turn."""
    planner = FakePlanner(called("get_budget_summary", project_id="orion"))

    result = nodes(planner).think(
        state(observations=(a_successful_list_risks(),))
    )

    assert planner.tool_choice_seen == ["auto"]
    assert result.route == "call_tool"
    assert result.tool_name == "get_budget_summary"


def test_a_repeated_call_is_forced_once_and_the_forced_answer_wins() -> None:
    """The ordinary case: the real provider's tool_choice: 'none' guarantees
    an answer, and that answer becomes the reply -- one extra model call,
    zero extra graph steps (both calls happen inside this one think())."""
    planner = FakePlanner(
        called("list_risks", project_id="orion"),
        answered("No new risk recorded: nothing has changed since the last check."),
    )

    result = nodes(planner).think(
        state(observations=(a_successful_list_risks(),))
    )

    assert planner.tool_choice_seen == ["auto", "none"]
    assert result.route == "answer"
    assert "nothing has changed" in result.response
    assert any(
        "repeated with the same arguments" in event.detail for event in result.events
    )


def test_a_repeat_that_survives_the_forced_call_fails_as_planner_loop() -> None:
    """Unreachable against the real provider (tool_choice: 'none' cannot
    produce a tool call); a fake that ignores the flag anyway proves the
    guard still ends the turn, and ends it distinctly from a provider
    hiccup or the step-budget guard."""
    planner = FakePlanner(
        called("list_risks", project_id="orion"),
        called("list_risks", project_id="orion"),
    )

    result = nodes(planner).think(
        state(observations=(a_successful_list_risks(),))
    )

    assert planner.tool_choice_seen == ["auto", "none"]
    assert result.route == "fail"
    assert result.failure == "planner_loop"
    assert result.failure != "max_steps_exceeded"
    assert "list_risks" in result.error_detail


def test_a_repeated_write_request_is_also_forced() -> None:
    """The guard is not read-only: a mutating tool named identically to an
    already-succeeded one is refused the same way."""
    already_written = ToolOutcome(
        tool_name="create_risk",
        arguments_summary="create_risk(project_id=orion, severity=medium)",
        status="ok",
        summary="Recorded R-3 (medium) against orion.",
        source_ids=("risk-r-3",),
    )
    planner = FakePlanner(
        ReasoningDecision(
            route="request_approval",
            confidence=0.5,
            required_tool="create_risk",
            tool_arguments={"project_id": "orion", "severity": "medium"},
            mutating=True,
            approval_required=True,
        ),
        answered("R-3 already covers this; nothing further to record."),
    )

    result = nodes(planner).think(state(observations=(already_written,)))

    assert planner.tool_choice_seen == ["auto", "none"]
    assert result.route == "answer"


# --------------------------------------------------------------------------
# The real engine, end to end
# --------------------------------------------------------------------------


class ScriptedRealPlannerModel:
    """A ``DecisionModel`` that returns choices in sequence and honors
    ``tool_choice`` the way the real provider does: at "none", a queued
    tool call is never even offered a chance to be seen as one -- the fake
    answers with its own queued content instead, matching a well-behaved
    (real) provider's tool_choice: "none"."""

    def __init__(self, *results: ToolCallResult) -> None:
        self.results = list(results)
        self.calls = 0

    def decide(
        self, question, evidence=(), observations=(), memories=(), history=(),
        *, tools=(), tool_choice="auto",
    ):
        self.calls += 1
        result = self.results[min(self.calls - 1, len(self.results) - 1)]
        if tool_choice == "none" and result.tool_name is not None:
            raise AssertionError(
                "a well-behaved model was asked for a queued tool call with "
                "tool_choice='none' -- fix the test's queue, not this fake"
            )
        return result


class ScriptedGateway:
    def __init__(self, *outcomes: ToolOutcome) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def execute(self, request: ToolRequest) -> ToolOutcome:
        self.calls += 1
        return self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]

    def preflight(self, request: ToolRequest) -> ToolOutcome:  # pragma: no cover
        raise AssertionError("no write is proposed in these engine-level tests")


def tool_called(name: str, **arguments) -> ToolCallResult:
    return ToolCallResult.from_tool_call(tool_name=name, arguments=arguments)


def content(text: str) -> ToolCallResult:
    return ToolCallResult.from_content(text)


def real_engine(*results: ToolCallResult, outcomes: tuple[ToolOutcome, ...]) -> WorkflowRuntime:
    return WorkflowRuntime(
        retriever=None,
        tools=ScriptedGateway(*outcomes),
        planner=Planner(ScriptedRealPlannerModel(*results)),
        composer=None,
        sleep=lambda seconds: None,
    )


def start() -> AgentState:
    return AgentState(
        request="Are there any open risks on orion?",
        actor="orion.lead",
        project_code="orion",
        trace_id="run-1",
        scopes=SCOPES,
    )


def test_the_real_engine_ends_a_repeat_in_one_extra_step() -> None:
    """list_risks succeeds once, is asked for again, and the guard ends the
    turn with an answer -- never touching the eight-step budget."""
    ok = ToolOutcome(
        tool_name="list_risks",
        arguments_summary="list_risks(project_id=orion)",
        status="ok",
        summary="No open risks recorded for orion.",
        source_ids=("project-orion",),
    )
    engine = real_engine(
        tool_called("list_risks", project_id="orion"),
        tool_called("list_risks", project_id="orion"),  # the repeat
        content("Still no open risks recorded for orion."),  # forced answer
        outcomes=(ok,),
    )

    final = engine.run(start())

    assert final.route == "answer"
    assert final.failure == "none"
    assert final.step_count <= 4
    assert "Still no open risks" in final.response


def test_two_different_writes_in_one_turn_are_both_still_reachable() -> None:
    """The boundary this guard must not cross -- already proved end to end by
    tests/engine/test_orchestrator.py's resumed-pause test; reasserted here,
    at the think()-unit level, as the fact ADR 0019 is written against."""
    first_write = ToolOutcome(
        tool_name="create_risk",
        arguments_summary="create_risk(project_id=orion, severity=medium)",
        status="ok",
        summary="Recorded R-3 (medium) against orion.",
        source_ids=("risk-r-3",),
    )
    planner = FakePlanner(
        ReasoningDecision(
            route="request_approval",
            confidence=0.5,
            required_tool="create_risk",
            tool_arguments={"project_id": "orion", "severity": "high"},
            mutating=True,
            approval_required=True,
        )
    )

    result = nodes(planner).think(state(observations=(first_write,)))

    # A different write (different arguments_summary) is never a "repeat":
    # tool_choice="none" is never even reached for it.
    assert planner.tool_choice_seen == ["auto"]
    assert result.route == "request_approval"
    assert result.tool_name == "create_risk"
