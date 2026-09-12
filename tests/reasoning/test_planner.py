"""A function call is a routing signal only if the mapping is the whole rule.

Every test here drives a scripted :class:`ToolCallResult` -- the shape a real
adapter produces -- through the planner and asserts on the typed decision. No
provider, no HTTP, no prompt: the point of the port is that this file needs
none of them.
"""

from datetime import UTC, datetime

import pytest

from agentic_erp_assistant.llm.tools import (
    DEFAULT_TOOLS,
    PLANNING_TOOLS,
    ToolCallResult,
)
from agentic_erp_assistant.reasoning.planner import (
    BLANK_REQUEST_QUESTION,
    DecisionModel,
    Planner,
    UNSCORED_CONFIDENCE,
)
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.tool_outcome import ToolOutcome


class ScriptedModel:
    """Returns the choices it was given, and records what it was shown."""

    def __init__(self, *results: ToolCallResult) -> None:
        self.results = list(results)
        self.calls: list[tuple] = []
        self.history_calls: list[tuple] = []
        self.tool_choice_calls: list[str] = []

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
        self.calls.append((question, tuple(evidence), tuple(observations), tuple(tools)))
        self.history_calls.append(tuple(history))
        self.tool_choice_calls.append(tool_choice)
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


def called(name: str, **arguments) -> ToolCallResult:
    return ToolCallResult.from_tool_call(tool_name=name, arguments=arguments)


def state(request: str = "How is M2 tracking?", **changes) -> AgentState:
    base = AgentState(
        request=request, actor="bao", project_code="atlas", trace_id="run-1"
    )
    return base.evolve(**changes) if changes else base


def plan_for(result: ToolCallResult, **state_changes):
    model = ScriptedModel(result)
    return Planner(model).plan(state(**state_changes)), model


# --------------------------------------------------------------------------
# The approval rule, stated once and read off the declaration
# --------------------------------------------------------------------------


def test_a_read_tool_routes_straight_to_execution() -> None:
    decision, _ = plan_for(called("list_risks", project_id="atlas"))

    assert decision.route == "call_tool"
    assert decision.required_tool == "list_risks"
    assert not decision.mutating
    assert not decision.approval_required


def test_a_write_never_routes_straight_to_execution() -> None:
    """The gate is the tool's own mutating flag, not the tool's name and not
    anything the model said."""
    decision, _ = plan_for(
        called("create_risk", project_id="atlas", title="Vendor slip", severity="high")
    )

    assert decision.route == "request_approval"
    assert decision.mutating
    assert decision.approval_required


def test_the_write_decision_names_the_tool_an_approver_will_be_asked_about() -> None:
    decision, _ = plan_for(
        called("create_risk", project_id="atlas", title="Vendor slip", severity="high")
    )

    assert decision.required_tool == "create_risk"


# --------------------------------------------------------------------------
# The three functions the graph carries out itself
# --------------------------------------------------------------------------


def test_choosing_search_routes_to_retrieval_and_carries_the_query() -> None:
    decision, _ = plan_for(
        called("search_project_documents", query="M2 delivery commitments")
    )

    assert decision.route == "retrieve_project_documents"
    assert decision.search_query == "M2 delivery commitments"
    assert decision.required_tool is None


def test_choosing_clarification_routes_to_clarify_with_the_question() -> None:
    decision, _ = plan_for(
        called("ask_clarification", question="Which milestone did you mean?"),
        request="how is it going?",
    )

    assert decision.route == "clarify"
    assert decision.message == "Which milestone did you mean?"


def test_choosing_refusal_routes_to_refuse_with_the_reason() -> None:
    decision, _ = plan_for(
        called("refuse", reason="That is not a project-delivery question."),
        request="write me a poem",
    )

    assert decision.route == "refuse"
    assert decision.message == "That is not a project-delivery question."


def test_control_arguments_are_validated_here_because_the_graph_reads_them() -> None:
    """A tool call's arguments are the gateway's job; these become the words a
    user sees, so a missing one must not reach a node."""
    decision, _ = plan_for(called("ask_clarification"))

    assert decision.route == "fail"
    assert "ask_clarification" in decision.rationale


# --------------------------------------------------------------------------
# Answering, and the choices that cannot be acted on
# --------------------------------------------------------------------------


def test_content_with_no_call_is_the_answer_route() -> None:
    decision, _ = plan_for(ToolCallResult.from_content("Nothing is at risk."))

    assert decision.route == "answer"
    assert decision.message == "Nothing is at risk."


def test_a_tool_that_was_never_offered_fails_the_turn_rather_than_crashing() -> None:
    decision, _ = plan_for(called("delete_project", project_id="atlas"))

    assert decision.route == "fail"
    assert "delete_project" in decision.rationale


# --------------------------------------------------------------------------
# tool_choice: how a turn's planning loop is ended (ADR 0019) and how a
# declared reply contract is enforced (ADR 0021)
# --------------------------------------------------------------------------


def test_tool_choice_defaults_to_auto_and_reaches_the_model() -> None:
    model = ScriptedModel(called("list_risks", project_id="atlas"))

    Planner(model).plan(state())

    assert model.tool_choice_calls == ["auto"]


def test_tool_choice_none_reaches_the_model() -> None:
    model = ScriptedModel(ToolCallResult.from_content("Recorded R-6 against orion."))

    Planner(model).plan(state(), tool_choice="none")

    assert model.tool_choice_calls == ["none"]


def test_tool_choice_none_with_an_answer_is_the_answer_route() -> None:
    """The ordinary case: the real provider's tool_choice: 'none' guarantees
    this, and this is what forces a reply immediately after a write."""
    model = ScriptedModel(ToolCallResult.from_content("Recorded R-6 against orion."))

    decision = Planner(model).plan(state(), tool_choice="none")

    assert decision.route == "answer"
    assert decision.message == "Recorded R-6 against orion."


def test_tool_choice_none_with_a_tool_call_anyway_fails_rather_than_routes() -> None:
    """The real provider cannot produce this (tool_choice: 'none' forbids a
    call); seeing it means whatever is standing in for the client did not
    honor the request. Never executed -- the turn ends at 'fail'."""
    model = ScriptedModel(called("list_risks", project_id="orion"))

    decision = Planner(model).plan(state(), tool_choice="none")

    assert decision.route == "fail"
    assert decision.required_tool is None
    assert "withheld" in decision.rationale


def test_tool_choice_required_reaches_the_model() -> None:
    model = ScriptedModel(called("get_project_status", milestone_id="M2"))

    Planner(model).plan(state(), tool_choice="required")

    assert model.tool_choice_calls == ["required"]


def test_tool_choice_required_with_prose_anyway_fails_rather_than_answers() -> None:
    """The real provider cannot produce this (tool_choice: 'required' forbids
    content-only); seeing it means whatever is standing in for the client
    did not honor the request."""
    model = ScriptedModel(ToolCallResult.from_content("Two days late."))

    decision = Planner(model).plan(state(), tool_choice="required")

    assert decision.route == "fail"
    assert "required" in decision.rationale


def test_withhold_removes_a_tool_from_what_is_offered() -> None:
    model = ScriptedModel(called("get_project_status", milestone_id="M2"))

    Planner(model).plan(state(), withhold=frozenset({"search_project_documents"}))

    offered = model.calls[0][3]
    assert "search_project_documents" not in {spec.name for spec in offered}
    assert "get_project_status" in {spec.name for spec in offered}


def test_a_call_to_a_withheld_tool_fails_rather_than_routes() -> None:
    """The real provider cannot call a function it was not offered; seeing
    this means whatever is standing in for the client did not honor the
    withheld set."""
    model = ScriptedModel(called("search_project_documents", query="M2 delay"))

    decision = Planner(model).plan(
        state(), withhold=frozenset({"search_project_documents"})
    )

    assert decision.route == "fail"
    assert decision.required_tool is None
    assert "withheld" in decision.rationale


def test_an_empty_request_asks_back_without_spending_a_model_call() -> None:
    model = ScriptedModel(called("list_risks", project_id="atlas"))

    decision = Planner(model).plan(state(request="   "))

    assert decision.route == "clarify"
    assert decision.message == BLANK_REQUEST_QUESTION
    assert model.calls == []


# --------------------------------------------------------------------------
# What the model is shown, and what the trace records
# --------------------------------------------------------------------------


def test_the_model_sees_the_whole_turn_and_not_just_the_question() -> None:
    """Without the observations, a second decision cannot differ from the
    first, and the loop repeats the call it already made."""
    snippet = EvidenceSnippet(source_id="m2.md", locator="p.1", text="M2 slipped.")
    outcome = ToolOutcome(
        tool_name="list_risks", status="ok", summary="2 open", source_ids=("atlas",)
    )

    _, model = plan_for(
        ToolCallResult.from_content("Two risks are open."),
        evidence=(snippet,),
        observations=(outcome,),
    )

    question, evidence, observations, offered = model.calls[0]
    assert question == "How is M2 tracking?"
    assert evidence == (snippet,)
    assert observations == (outcome,)
    assert offered == PLANNING_TOOLS


def test_the_model_is_shown_the_history_on_the_state() -> None:
    turn = ConversationTurn(
        trace_id="run-0",
        session_id="sess-1",
        actor="bao",
        request="How is M2 tracking?",
        response="On track.",
        route="answer",
        started_at=datetime(2026, 9, 8, tzinfo=UTC),
        finished_at=datetime(2026, 9, 8, tzinfo=UTC),
    )

    _, model = plan_for(
        ToolCallResult.from_content("Still on track."),
        session_id="sess-1",
        history=(turn,),
    )

    assert model.history_calls[0] == (turn,)


def test_the_rationale_states_what_was_chosen_and_not_a_reported_thought() -> None:
    """A narrated reason would read like evidence of a decision process and
    would not be one; the trace records the mechanism instead."""
    decision, _ = plan_for(called("list_risks", project_id="atlas"))

    assert decision.rationale == "called list_risks"


def test_every_decision_records_that_nothing_scored_its_confidence() -> None:
    decision, _ = plan_for(called("list_risks", project_id="atlas"))

    assert decision.confidence == UNSCORED_CONFIDENCE


def test_the_offered_menu_is_injectable() -> None:
    model = ScriptedModel(called("list_risks", project_id="atlas"))

    Planner(model, tools=DEFAULT_TOOLS).plan(state())

    assert model.calls[0][3] == DEFAULT_TOOLS


def test_a_narrowed_menu_makes_the_missing_tool_unroutable() -> None:
    """The offered set is the authority: a tool the planner did not offer is
    not a tool it will route to, whatever the model returns."""
    model = ScriptedModel(
        called("create_risk", project_id="atlas", title="x", severity="low")
    )

    decision = Planner(model, tools=(DEFAULT_TOOLS[0],)).plan(state())

    assert decision.route == "fail"


# --------------------------------------------------------------------------
# The seams
# --------------------------------------------------------------------------


def test_a_scripted_model_satisfies_the_decision_port() -> None:
    assert isinstance(ScriptedModel(), DecisionModel)


def test_the_planner_satisfies_the_runtime_port_without_importing_it() -> None:
    from agentic_erp_assistant.engine.ports import PlannerPort

    assert isinstance(Planner(ScriptedModel()), PlannerPort)


def test_the_real_gateway_satisfies_the_decision_port() -> None:
    from agentic_erp_assistant.llm.gateway import LLMGateway

    assert hasattr(LLMGateway, "decide")


@pytest.mark.parametrize(
    "name", ["search_project_documents", "ask_clarification", "refuse"]
)
def test_every_control_function_the_planner_offers_has_a_route(name: str) -> None:
    """A function offered without an entry would fall through to the tool path,
    where the registry has no handler for it."""
    from agentic_erp_assistant.reasoning.planner import CONTROL_ROUTES

    assert name in CONTROL_ROUTES
