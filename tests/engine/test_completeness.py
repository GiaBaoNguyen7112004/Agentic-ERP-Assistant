"""ADR 0021: think() holds every attempt to answer or search to the reply
contract the turn declared -- a missing need is redirected once, and the
redirect never delivers less than the planner would have.

Two levels, the same split test_think_after_write.py uses: think() in
isolation (the mechanics -- which call got which tool_choice and withhold,
what the trace says, what falls through unmodified) and the real engine
end-to-end (a call that actually executes, and the step count that results).
The full R1 shape -- retrieval finding passages and the composer being handed
what the tool call returned -- is Phase Q3's, tested there once
retrieve_and_answer is wired to see observations; this file proves the
redirect itself, not composition.
"""

from agentic_erp_assistant.reasoning.decision import ReasoningDecision
from agentic_erp_assistant.engine.nodes import GraphNodes, RETRIEVAL_TOOL
from agentic_erp_assistant.engine.workflow import WorkflowRuntime
from agentic_erp_assistant.llm.tools import ToolCallResult
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.reply_contract import ReplyContract
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.state.tool_request import ToolRequest

SCOPES = frozenset({"project.status.read", "project.risk.read", "project.risk.write"})


# --------------------------------------------------------------------------
# think() in isolation
# --------------------------------------------------------------------------


class FakePlanner:
    """Returns decisions in sequence, and records every call's tool_choice
    and withhold set."""

    def __init__(self, *decisions: ReasoningDecision) -> None:
        self.decisions = list(decisions)
        self.calls: list[tuple[str, frozenset]] = []

    def plan(
        self,
        state: AgentState,
        *,
        tool_choice: str = "auto",
        withhold: frozenset[str] = frozenset(),
    ) -> ReasoningDecision:
        self.calls.append((tool_choice, withhold))
        index = len(self.calls) - 1
        return self.decisions[min(index, len(self.decisions) - 1)]


class FakeGateway:
    def __init__(self, outcome: ToolOutcome | None = None) -> None:
        self.outcome = outcome
        self.executed: list[ToolRequest] = []

    def preflight(self, request: ToolRequest) -> ToolOutcome:  # pragma: no cover
        raise AssertionError("no write is proposed in these tests")

    def execute(self, request: ToolRequest) -> ToolOutcome:
        self.executed.append(request)
        if self.outcome is None:
            raise AssertionError("execute() was not expected in this test")
        return self.outcome


def nodes(planner: FakePlanner, gateway: FakeGateway | None = None) -> GraphNodes:
    return GraphNodes(
        retriever=None,  # not exercised by any test here
        tools=gateway or FakeGateway(),
        planner=planner,
        composer=None,  # not exercised by any test here
    )


def state(**changes) -> AgentState:
    base = AgentState(
        request="Why is milestone M2 late and by how much?",
        actor="priya",
        project_code="atlas",
        trace_id="run-1",
        scopes=SCOPES,
    )
    return base.evolve(**changes) if changes else base


def called(name: str, **arguments) -> ReasoningDecision:
    return ReasoningDecision(
        route="call_tool", confidence=0.5, required_tool=name, tool_arguments=arguments
    )


def answered(text: str) -> ReasoningDecision:
    return ReasoningDecision(route="answer", confidence=0.5, message=text)


def clarified(question: str) -> ReasoningDecision:
    return ReasoningDecision(route="clarify", confidence=0.5, message=question)


def searched(query: str) -> ReasoningDecision:
    return ReasoningDecision(
        route="retrieve_project_documents", confidence=0.5, search_query=query
    )


def kinds(result: AgentState) -> list[str]:
    return [event.kind for event in result.events]


def an_ok_observation(tool_name: str = "get_project_status") -> ToolOutcome:
    return ToolOutcome(
        tool_name=tool_name, status="ok", summary="2 days late.", source_ids=("milestone-m2",)
    )


BOTH = ReplyContract(needs=frozenset({"document_passage", "erp_field"}), document_query="why")
ERP_FIELD_ONLY = ReplyContract(needs=frozenset({"erp_field"}))
DOCUMENT_ONLY = ReplyContract(needs=frozenset({"document_passage"}), document_query="why")


def test_no_contract_redirects_nothing() -> None:
    """contract=None is every existing engine test's shape -- proven en
    masse by the rest of the suite passing unchanged; this is the direct
    version of that claim."""
    planner = FakePlanner(answered("Two days late."))

    result = nodes(planner).think(state(contract=None))

    assert len(planner.calls) == 1
    assert result.route == "answer"
    assert result.failure == "none"
    assert "contract_enforced" not in kinds(result)


def test_an_unmet_field_redirects_a_decision_to_answer() -> None:
    """The model tried to answer with nothing fetched; the field is
    required, and search stays withheld -- there is nothing to search for
    yet that the contract's own query would help with here."""
    planner = FakePlanner(
        answered("Two days late."), called("get_project_status", milestone_id="M2")
    )

    result = nodes(planner).think(state(contract=ERP_FIELD_ONLY))

    assert planner.calls[1] == ("required", frozenset({RETRIEVAL_TOOL}))
    assert result.route == "call_tool"
    assert result.tool_name == "get_project_status"
    assert result.redirected_needs == frozenset({"erp_field"})
    assert "erp_field: a call was required, search withheld" in [
        event.detail for event in result.events
    ]


def test_an_unmet_field_redirects_a_decision_to_search_too() -> None:
    """D5: fields before passages. A model that goes straight for
    retrieve_project_documents on a compound question is redirected exactly
    like one that tried to answer -- the field still has to come first."""
    planner = FakePlanner(searched("M2 delay"), called("get_project_status", milestone_id="M2"))

    result = nodes(planner).think(state(contract=BOTH))

    assert planner.calls[1] == ("required", frozenset({RETRIEVAL_TOOL}))
    assert result.route == "call_tool"


def test_the_forced_call_can_still_choose_to_ask_back() -> None:
    """The model's choice is honored, not forced twice: a clarification
    ends the turn exactly as it would without any contract in play."""
    planner = FakePlanner(answered("Two days late."), clarified("Which milestone?"))

    result = nodes(planner).think(state(contract=ERP_FIELD_ONLY))

    assert len(planner.calls) == 2
    assert result.route == "clarify"
    assert result.response == "Which milestone?"


def test_a_call_already_satisfying_the_field_needs_no_redirect() -> None:
    """T1's shape: the model calls the ERP tool on its own, on the very
    first decision -- nothing here is missing yet to redirect."""
    planner = FakePlanner(called("get_project_status", milestone_id="M2"))

    result = nodes(planner).think(state(contract=ERP_FIELD_ONLY))

    assert len(planner.calls) == 1
    assert result.route == "call_tool"
    assert "contract_enforced" not in kinds(result)


def test_answering_once_the_field_is_already_observed_needs_no_redirect() -> None:
    """The other half of T1's shape: by the time the model answers, the
    field is already sitting in observations from an earlier think() -- the
    residual risk ADR 0014 named, closed for this contract."""
    planner = FakePlanner(answered("On track."))

    result = nodes(planner).think(
        state(contract=ERP_FIELD_ONLY, observations=(an_ok_observation(),))
    )

    assert len(planner.calls) == 1
    assert result.route == "answer"
    assert result.failure == "none"
    assert "contract_enforced" not in kinds(result)


def test_an_unmet_passage_redirects_an_answer_to_search() -> None:
    """The document_passage half, once the field is already satisfied: the
    planner's own draft is withheld and kept, and the turn moves to
    retrieval with the contract's own query -- not the model's, since the
    model never named one for a route it did not choose."""
    planner = FakePlanner(answered("Two days late."))

    result = nodes(planner).think(
        state(contract=BOTH, observations=(an_ok_observation(),))
    )

    assert len(planner.calls) == 1
    assert result.route == "retrieve_project_documents"
    assert result.tool_arguments == {"query": "why"}
    assert result.draft == "Two days late."
    assert result.redirected_needs == frozenset({"document_passage"})
    assert any(
        event.kind == "contract_enforced"
        and "document_passage: answer withheld, searching 'why'" in event.detail
        for event in result.events
    )
    assert any(
        event.kind == "route_selected"
        and "contract needs a document passage" in event.detail
        for event in result.events
    )


def test_a_model_chosen_search_is_left_alone_even_with_a_passage_missing() -> None:
    """The model already chose to search, with its own query -- overriding
    it with the contract's query would discard a choice that was already
    correct. No redirect fires; the ordinary retrieval branch runs as-is."""
    planner = FakePlanner(searched("finance module delay"))

    result = nodes(planner).think(
        state(contract=BOTH, observations=(an_ok_observation(),))
    )

    assert len(planner.calls) == 1
    assert result.route == "retrieve_project_documents"
    assert result.tool_arguments == {"query": "finance module delay"}
    assert result.draft is None
    assert "contract_enforced" not in kinds(result)


def test_a_need_is_redirected_at_most_once() -> None:
    """Already spent -- next_redirect will not offer erp_field again, so a
    second answer attempt with the field still missing is delivered,
    marked, rather than looping."""
    planner = FakePlanner(answered("Two days late."))

    result = nodes(planner).think(
        state(contract=ERP_FIELD_ONLY, redirected_needs=frozenset({"erp_field"}))
    )

    assert len(planner.calls) == 1
    assert result.route == "answer"
    assert result.failure == "incomplete_reply"
    assert "erp_field" in result.error_detail
    assert any(
        event.kind == "contract_enforced" and "unmet after redirect: erp_field" in event.detail
        for event in result.events
    )


def test_adr_0019s_forced_answer_satisfying_the_field_delivers_complete() -> None:
    """A11's shape, contract {erp_field}: the repeat guard forces an answer
    after a write; the write's own receipt already satisfies the field, so
    nothing here needs redirecting -- the two guards do not interfere."""
    already_written = ToolOutcome(
        tool_name="create_risk",
        arguments_summary="create_risk(project_id=atlas, severity=medium)",
        status="ok",
        summary="Recorded R-3 (medium) against atlas.",
        source_ids=("risk-r-3",),
    )
    planner = FakePlanner(
        called("create_risk", project_id="atlas", severity="medium"),
        answered("R-3 already covers this; nothing further to record."),
    )

    result = nodes(planner).think(
        state(
            request="Record a medium risk on atlas.",
            contract=ERP_FIELD_ONLY,
            observations=(already_written,),
        )
    )

    assert len(planner.calls) == 2
    assert planner.calls[1] == ("none", frozenset())
    assert result.route == "answer"
    assert result.failure == "none"
    assert "contract_enforced" not in kinds(result)


def test_adr_0019s_forced_answer_still_redirects_a_missing_passage() -> None:
    """The two guards compose: contract {both} still has document_passage
    unmet after the write, so the forced answer is redirected to search --
    the repeat guard does not exempt a turn from the completeness check."""
    already_written = ToolOutcome(
        tool_name="create_risk",
        arguments_summary="create_risk(project_id=atlas, severity=medium)",
        status="ok",
        summary="Recorded R-3 (medium) against atlas.",
        source_ids=("risk-r-3",),
    )
    contract = ReplyContract(
        needs=frozenset({"document_passage", "erp_field"}), document_query="why atlas is at risk"
    )
    planner = FakePlanner(
        called("create_risk", project_id="atlas", severity="medium"),
        answered("R-3 already covers this; nothing further to record."),
    )

    result = nodes(planner).think(
        state(
            request="Record a medium risk on atlas.",
            contract=contract,
            observations=(already_written,),
        )
    )

    assert len(planner.calls) == 2
    assert result.route == "retrieve_project_documents"
    assert result.draft == "R-3 already covers this; nothing further to record."
    assert result.redirected_needs == frozenset({"document_passage"})


# --------------------------------------------------------------------------
# The real engine, end to end
# --------------------------------------------------------------------------


class ScriptedRealPlannerModel:
    """A DecisionModel honoring tool_choice and withhold the way the real
    provider does -- see test_think_after_write.py's twin for why this
    matters: a queued tool call surviving tool_choice='none' or a withheld
    name would prove nothing about think(), only about a broken fixture."""

    def __init__(self, *results: ToolCallResult) -> None:
        self.results = list(results)
        self.calls = 0

    def decide(
        self, question, evidence=(), observations=(), memories=(), history=(),
        *, tools=(), tool_choice="auto",
    ):
        self.calls += 1
        result = self.results[min(self.calls - 1, len(self.results) - 1)]
        offered_names = {tool.name for tool in tools}
        if result.tool_name is not None and result.tool_name not in offered_names:
            raise AssertionError(
                f"queued a call to {result.tool_name!r}, which this call did "
                f"not offer -- fix the test's queue, not this fake"
            )
        if tool_choice == "none" and result.tool_name is not None:
            raise AssertionError("queued a tool call under tool_choice='none'")
        if tool_choice == "required" and result.tool_name is None:
            raise AssertionError("queued content under tool_choice='required'")
        return result


def real_called(name: str, **arguments) -> ToolCallResult:
    return ToolCallResult.from_tool_call(tool_name=name, arguments=arguments)


def real_answered(text: str) -> ToolCallResult:
    return ToolCallResult.from_content(text)


class ScriptedGateway:
    def __init__(self, *outcomes: ToolOutcome) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def execute(self, request: ToolRequest) -> ToolOutcome:
        self.calls += 1
        return self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]

    def preflight(self, request: ToolRequest) -> ToolOutcome:  # pragma: no cover
        raise AssertionError("no write is proposed in these tests")


def runtime(model: ScriptedRealPlannerModel, gateway: ScriptedGateway) -> WorkflowRuntime:
    return WorkflowRuntime(
        retriever=None,  # not exercised: no route here reaches retrieval
        tools=gateway,
        planner=Planner(model),
        composer=None,  # not exercised
        sleep=lambda seconds: None,
    )


def test_a_field_redirect_that_fails_ends_the_turn_without_a_second_redirect() -> None:
    """The forced call chose a real tool; it failed; execute_tool ends the
    turn on its own terms, exactly as it would for any other failed call --
    the completeness check gets no second attempt, because the turn is over."""
    model = ScriptedRealPlannerModel(
        real_answered("Two days late."),
        real_called("get_project_status", milestone_id="M2"),
    )
    failed = ToolOutcome(
        tool_name="get_project_status", status="invalid_arguments", error="bad milestone id"
    )
    engine = runtime(model, ScriptedGateway(failed))

    result = engine.run(
        AgentState(
            request="Why is milestone M2 late and by how much?",
            actor="priya",
            project_code="atlas",
            trace_id="run-1",
            scopes=SCOPES,
            contract=ERP_FIELD_ONLY,
        )
    )

    assert model.calls == 2
    assert result.route == "fail"
    assert result.failure == "tool_failure"
    assert result.terminal is True


def test_a_field_redirect_that_succeeds_returns_to_think_and_completes() -> None:
    """The full field-redirect round trip: answer withheld, a call forced,
    the call succeeds, think() runs again and now delivers complete."""
    model = ScriptedRealPlannerModel(
        real_answered("Two days late."),
        real_called("get_project_status", milestone_id="M2"),
        real_answered("Two days late, per the ERP."),
    )
    ok = ToolOutcome(
        tool_name="get_project_status", status="ok", summary="2 days late.",
        source_ids=("milestone-m2",),
    )
    engine = runtime(model, ScriptedGateway(ok))

    result = engine.run(
        AgentState(
            request="Why is milestone M2 late and by how much?",
            actor="priya",
            project_code="atlas",
            trace_id="run-1",
            scopes=SCOPES,
            contract=ERP_FIELD_ONLY,
        )
    )

    assert model.calls == 3
    assert result.route == "answer"
    assert result.failure == "none"
    assert result.step_count == 3
