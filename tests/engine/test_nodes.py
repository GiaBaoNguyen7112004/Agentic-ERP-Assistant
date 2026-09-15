"""Each node, on its own, against four fakes and no I/O.

The loop that calls these is tested in test_runtime_routes.py. What is proved
here is what one node does with one state: which route it moves to, what it
writes into the trace, and -- the load-bearing one -- that an answer citing a
source nobody retrieved never reaches a user.
"""

from datetime import UTC, datetime

import pytest

from agentic_erp_assistant.llm.schemas import Citation, GroundedAnswer
from agentic_erp_assistant.reasoning.decision import ReasoningDecision
from agentic_erp_assistant.engine.nodes import (
    GraphNodes,
    NO_EVIDENCE_REPLY,
    RETRIEVAL_TOOL,
    SOURCES_PREFIX,
)
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.reply_contract import ReplyContract
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.state.tool_request import ToolRequest

SCOPES = frozenset({"project.risk.read", "project.risk.write"})
NOW = datetime(2026, 9, 8, tzinfo=UTC)


class FakeRetriever:
    def __init__(self, *snippets: EvidenceSnippet) -> None:
        self.snippets = snippets
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int):
        self.calls.append((query, limit))
        return self.snippets[:limit]


class QueryAwareRetriever:
    """Unlike FakeRetriever, returns different hits for different queries --
    what proving ADR 0027's union and dedup actually needs."""

    def __init__(self, by_query: dict[str, tuple[EvidenceSnippet, ...]]) -> None:
        self.by_query = by_query
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int):
        self.calls.append((query, limit))
        return self.by_query.get(query, ())[:limit]


class RaisingRetriever:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def search(self, query: str, *, limit: int):
        raise self.error


class FakeGateway:
    def __init__(
        self, *outcomes: ToolOutcome, preflight_outcome: ToolOutcome | None = None
    ) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[ToolRequest] = []
        self.preflight_requests: list[ToolRequest] = []
        self.preflight_outcome = preflight_outcome

    def execute(self, request: ToolRequest) -> ToolOutcome:
        self.requests.append(request)
        return self.outcomes[min(len(self.requests) - 1, len(self.outcomes) - 1)]

    def preflight(self, request: ToolRequest) -> ToolOutcome:
        self.preflight_requests.append(request)
        if self.preflight_outcome is not None:
            return self.preflight_outcome
        return ToolOutcome(
            tool_name=request.tool_name,
            status="approval_required",
            error=f"{request.tool_name} needs a human",
        )


class FakePlanner:
    def __init__(self, *decisions, raises: Exception | None = None) -> None:
        self.decisions = list(decisions)
        self.raises = raises
        self.seen: list[AgentState] = []

    def plan(self, state: AgentState) -> ReasoningDecision:
        self.seen.append(state)
        if self.raises is not None:
            raise self.raises
        return self.decisions[min(len(self.seen) - 1, len(self.decisions) - 1)]


class FakeComposer:
    def __init__(self, answer: GroundedAnswer | None = None, raises=None) -> None:
        self.answer_value = answer
        self.raises = raises
        self.calls: list[tuple] = []
        self.history_calls: list[tuple] = []
        self.observation_calls: list[tuple] = []

    def answer(self, question: str, evidence, memories=(), history=(), observations=()):
        self.calls.append((question, tuple(evidence)))
        self.history_calls.append(tuple(history))
        self.observation_calls.append(tuple(observations))
        if self.raises is not None:
            raise self.raises
        return self.answer_value


def snippet(source_id: str = "m2-status.md") -> EvidenceSnippet:
    return EvidenceSnippet(
        source_id=source_id, locator="p.2", text="M2 slipped by two weeks."
    )


def grounded(*source_ids: str) -> GroundedAnswer:
    return GroundedAnswer(
        answer="M2 slipped by two weeks.",
        citations=[Citation(source_id=sid, locator="p.2") for sid in source_ids],
        grounded=True,
        confidence=0.9,
    )


def outcome(status: str = "ok", **changes) -> ToolOutcome:
    fields = {
        "tool_name": "list_risks",
        "status": status,
        "summary": "2 open risks",
        "source_ids": ("risk:R-1", "risk:R-2"),
    }
    if status != "ok":
        fields["summary"] = ""
        fields["source_ids"] = ()
        fields.setdefault("error", "something went wrong")
    fields.update(changes)
    return ToolOutcome(**fields)


def nodes(**overrides) -> GraphNodes:
    settings = {
        "retriever": FakeRetriever(snippet()),
        "tools": FakeGateway(outcome()),
        "planner": FakePlanner(ReasoningDecision(route="answer", confidence=0.5)),
        "composer": FakeComposer(grounded("m2-status.md")),
    }
    settings.update(overrides)
    return GraphNodes(**settings)


def state(**changes) -> AgentState:
    base = AgentState(
        request="How is M2 tracking?",
        actor="bao",
        project_code="atlas",
        trace_id="run-1",
        scopes=SCOPES,
    )
    return base.evolve(**changes) if changes else base


def kinds(result: AgentState) -> list[str]:
    return [event.kind for event in result.events]


# --------------------------------------------------------------------------
# think
# --------------------------------------------------------------------------


def test_a_retrieval_decision_carries_its_queries_onto_the_state() -> None:
    """Chosen like a tool, recorded like a tool, executed by the retriever."""
    graph = nodes(
        planner=FakePlanner(
            ReasoningDecision(
                route="retrieve_project_documents",
                confidence=0.5,
                search_queries=("M2 delivery commitments",),
            )
        )
    )

    result = graph.think(state())

    assert result.route == "retrieve_project_documents"
    assert result.tool_name == RETRIEVAL_TOOL
    assert result.tool_arguments == {"queries": ["M2 delivery commitments"]}


def test_a_retrieval_decision_with_several_queries_carries_all_of_them() -> None:
    """ADR 0027: one query per document, in order, all reaching the state."""
    graph = nodes(
        planner=FakePlanner(
            ReasoningDecision(
                route="retrieve_project_documents",
                confidence=0.5,
                search_queries=("risk register severity", "sprint 13 schedule slip"),
            )
        )
    )

    result = graph.think(state())

    assert result.tool_arguments == {
        "queries": ["risk register severity", "sprint 13 schedule slip"]
    }


def test_a_write_decision_pauses_the_turn_and_says_so_in_the_trace() -> None:
    graph = nodes(
        planner=FakePlanner(
            ReasoningDecision(
                route="request_approval",
                confidence=0.5,
                required_tool="create_risk",
                tool_arguments={"project_id": "atlas", "title": "x", "severity": "low"},
                mutating=True,
                approval_required=True,
            )
        )
    )

    result = graph.think(state())

    assert (result.route, result.approval) == ("request_approval", "pending")
    assert result.tool_mutating
    assert "approval_requested" in kinds(result)
    assert not result.terminal
    assert result.observations[-1].status == "approval_required"


def test_a_write_the_preflight_refuses_never_pauses() -> None:
    """ADR 0016: a human is asked only after the checks that would refuse the
    call anyway have already passed. A missing scope is exactly such a check,
    so the turn ends here instead of pausing on a call the gateway will
    refuse whatever the human says."""
    gateway = FakeGateway(
        preflight_outcome=ToolOutcome(
            tool_name="create_risk", status="denied", error="missing scope"
        )
    )
    graph = nodes(
        tools=gateway,
        planner=FakePlanner(
            ReasoningDecision(
                route="request_approval",
                confidence=0.5,
                required_tool="create_risk",
                tool_arguments={"project_id": "atlas", "title": "x", "severity": "low"},
                mutating=True,
                approval_required=True,
            )
        ),
    )

    result = graph.think(state())

    assert (result.route, result.failure) == ("fail", "tool_failure")
    assert result.terminal
    assert result.approval != "pending"
    assert result.observations[-1].status == "denied"
    assert "create_risk -> denied" in (result.error_detail or "")
    assert any(
        event.kind == "failed" and "refused before approval" in event.detail
        for event in result.events
    )
    assert gateway.requests == [], "execute() must never be reached"
    assert len(gateway.preflight_requests) == 1


def test_a_read_decision_goes_straight_to_execution() -> None:
    graph = nodes(
        planner=FakePlanner(
            ReasoningDecision(
                route="call_tool",
                confidence=0.5,
                required_tool="list_risks",
                tool_arguments={"project_id": "atlas"},
            )
        )
    )

    result = graph.think(state())

    assert (result.route, result.approval) == ("call_tool", "not_required")
    assert not result.tool_mutating


def test_an_answer_lists_the_records_the_calls_actually_touched() -> None:
    """Built from the observations, not from anything the model wrote."""
    graph = nodes(
        planner=FakePlanner(
            ReasoningDecision(route="answer", confidence=0.5, message="Two are open.")
        )
    )

    result = graph.think(state(observations=(outcome(),)))

    assert result.terminal
    assert result.response == f"Two are open.\n\n{SOURCES_PREFIX}risk:R-1, risk:R-2"


def test_the_rationale_reaches_the_trace_and_not_a_branch() -> None:
    graph = nodes(
        planner=FakePlanner(
            ReasoningDecision(
                route="clarify",
                confidence=0.5,
                message="Which milestone?",
                rationale="called ask_clarification",
            )
        )
    )

    result = graph.think(state())

    assert result.response == "Which milestone?"
    assert any(
        event.kind == "route_selected" and "called ask_clarification" in event.detail
        for event in result.events
    )


def test_a_planner_that_raises_ends_the_turn_instead_of_the_process() -> None:
    graph = nodes(planner=FakePlanner(raises=TimeoutError("upstream gone")))

    result = graph.think(state())

    assert (result.route, result.failure) == ("fail", "provider_failure")
    assert "TimeoutError" in (result.error_detail or "")
    assert result.terminal


def test_a_planner_decision_to_fail_is_reported_with_its_reason() -> None:
    graph = nodes(
        planner=FakePlanner(
            ReasoningDecision(
                route="fail",
                confidence=0.5,
                rationale="the model called 'delete_project', which was not offered",
            )
        )
    )

    result = graph.think(state())

    assert result.failure == "provider_failure"
    assert "delete_project" in (result.error_detail or "")


# --------------------------------------------------------------------------
# retrieve_and_answer
# --------------------------------------------------------------------------


def test_a_document_answer_carries_the_sources_it_rests_on() -> None:
    graph = nodes()

    result = graph.retrieve_and_answer(state(route="retrieve_project_documents"))

    assert result.route == "answer"
    assert result.response.endswith(f"{SOURCES_PREFIX}[m2-status.md#p.2]")
    assert result.evidence == (snippet(),)
    assert "evidence_retrieved" in kinds(result)


def test_the_composer_is_shown_this_turns_own_observations() -> None:
    """ADR 0021: a compound question redirected to retrieval after a tool
    call already succeeded needs the composer to see what that call
    returned, or the field it established is silently dropped."""
    composer = FakeComposer(grounded("m2-status.md"))
    graph = nodes(composer=composer)
    field = outcome(tool_name="get_project_status", source_ids=("milestone-m2",))

    graph.retrieve_and_answer(
        state(route="retrieve_project_documents", observations=(field,))
    )

    assert composer.calls[0][0] == "How is M2 tracking?"
    assert composer.observation_calls[0] == (field,)


def test_sources_merge_citations_and_observed_ids_citations_first() -> None:
    """Deduped, citations first -- the reply names what it quoted before
    what it merely read off a live field."""
    graph = nodes(composer=FakeComposer(grounded("m2-status.md")))
    field = outcome(tool_name="get_project_status", source_ids=("milestone-m2",))

    result = graph.retrieve_and_answer(
        state(route="retrieve_project_documents", observations=(field,))
    )

    assert result.response.endswith(
        f"{SOURCES_PREFIX}[m2-status.md#p.2], milestone-m2"
    )


def test_the_composer_is_shown_the_history_on_the_state() -> None:
    composer = FakeComposer(grounded("m2-status.md"))
    graph = nodes(composer=composer)
    turn = ConversationTurn(
        trace_id="run-0",
        session_id="sess-1",
        actor="bao",
        request="How is M2 tracking?",
        response="On track.",
        route="answer",
        started_at=NOW,
        finished_at=NOW,
    )

    graph.retrieve_and_answer(
        state(route="retrieve_project_documents", session_id="sess-1", history=(turn,))
    )

    assert composer.history_calls[0] == (turn,)


def test_the_planner_query_is_what_gets_searched() -> None:
    """A state built before ADR 0027 (or hand-built with the legacy singular
    key) still runs, as one query."""
    retriever = FakeRetriever(snippet())
    graph = nodes(retriever=retriever)

    graph.retrieve_and_answer(
        state(
            route="retrieve_project_documents",
            tool_name=RETRIEVAL_TOOL,
            tool_arguments={"query": "M2 delivery commitments"},
        )
    )

    assert retriever.calls == [("M2 delivery commitments", graph.evidence_limit)]


def test_every_planner_query_is_searched_in_order() -> None:
    """ADR 0027: one query per document, each actually searched -- not just
    the first or a blend of all of them."""
    retriever = QueryAwareRetriever(
        {
            "risk register severity": (snippet("risk-register"),),
            "sprint 13 schedule slip": (snippet("sprint-13-report"),),
        }
    )
    graph = nodes(retriever=retriever)

    result = graph.retrieve_and_answer(
        state(
            route="retrieve_project_documents",
            tool_name=RETRIEVAL_TOOL,
            tool_arguments={
                "queries": ["risk register severity", "sprint 13 schedule slip"]
            },
        )
    )

    assert [call[0] for call in retriever.calls] == [
        "risk register severity",
        "sprint 13 schedule slip",
    ]
    assert {snip.source_id for snip in result.evidence} == {
        "risk-register",
        "sprint-13-report",
    }


def test_a_passage_two_queries_both_match_is_shown_to_the_composer_once() -> None:
    """Union, deduped by citation tag -- first occurrence wins, so the
    passage keeps the rank its earlier query gave it."""
    shared = snippet("risk-register")
    retriever = QueryAwareRetriever(
        {
            "risk severity": (shared, snippet("sprint-13-report")),
            "schedule slip": (snippet("sprint-13-report"), shared),
        }
    )
    composer = FakeComposer(grounded("risk-register", "sprint-13-report"))
    graph = nodes(retriever=retriever, composer=composer)

    result = graph.retrieve_and_answer(
        state(
            route="retrieve_project_documents",
            tool_name=RETRIEVAL_TOOL,
            tool_arguments={"queries": ["risk severity", "schedule slip"]},
        )
    )

    # Two queries, three raw hits each turned up, only two distinct tags.
    assert len(result.evidence) == 2
    assert composer.calls[0][1] == result.evidence
    assert any(
        event.kind == "evidence_retrieved"
        and "2+2 passage(s) for 2 queries" in event.detail
        for event in result.events
    )


def test_the_evidence_retrieved_event_names_every_query() -> None:
    retriever = QueryAwareRetriever(
        {"risk severity": (snippet("risk-register"),), "schedule slip": ()}
    )
    graph = nodes(retriever=retriever)

    result = graph.retrieve_and_answer(
        state(
            route="retrieve_project_documents",
            tool_name=RETRIEVAL_TOOL,
            tool_arguments={"queries": ["risk severity", "schedule slip"]},
        )
    )

    (event,) = [e for e in result.events if e.kind == "evidence_retrieved"]
    assert "1+0 passage(s) for 2 queries" in event.detail
    assert "'risk severity'" in event.detail
    assert "'schedule slip'" in event.detail


def test_a_state_with_no_query_falls_back_to_the_request() -> None:
    """A replayed or hand-built state stays runnable without a planner."""
    retriever = FakeRetriever(snippet())
    graph = nodes(retriever=retriever)

    graph.retrieve_and_answer(state(route="retrieve_project_documents"))

    assert retriever.calls[0][0] == "How is M2 tracking?"


def test_nothing_retrieved_is_a_refusal_and_not_a_guess() -> None:
    graph = nodes(retriever=FakeRetriever())

    result = graph.retrieve_and_answer(state(route="retrieve_project_documents"))

    assert (result.route, result.failure) == ("refuse", "insufficient_evidence")
    assert result.response == NO_EVIDENCE_REPLY
    assert result.terminal


def test_a_redirected_search_finding_nothing_delivers_the_withheld_draft() -> None:
    """ADR 0021: the check's own redirect sent this search out and it found
    nothing -- the planner's withheld reply is delivered anyway, marked,
    never a bare refusal of a question the ERP already half-answered."""
    graph = nodes(retriever=FakeRetriever())
    field = outcome(tool_name="get_project_status", source_ids=("milestone-m2",))

    result = graph.retrieve_and_answer(
        state(
            route="retrieve_project_documents",
            observations=(field,),
            draft="Two days late.",
            redirected_needs=frozenset({"document_passage"}),
        )
    )

    assert result.route == "answer"
    assert result.failure == "incomplete_reply"
    assert result.response.startswith("Two days late.")
    assert "milestone-m2" in result.response
    assert "contract needs a document passage" in (result.error_detail or "")
    assert any(
        event.kind == "contract_enforced"
        and "unmet after redirect: document_passage" in event.detail
        for event in result.events
    )


def test_a_model_chosen_search_finding_nothing_still_refuses() -> None:
    """The softer landing is only for the check's own redirect -- draft is
    never set by anything else, so a model-chosen search that finds nothing
    refuses exactly as test_nothing_retrieved_is_a_refusal_and_not_a_guess
    already proves. This asserts the second half of the guard directly:
    redirected_needs alone, with no draft, is not enough either."""
    graph = nodes(retriever=FakeRetriever())

    result = graph.retrieve_and_answer(
        state(
            route="retrieve_project_documents",
            redirected_needs=frozenset({"document_passage"}),
        )
    )

    assert (result.route, result.failure) == ("refuse", "insufficient_evidence")


def test_the_defensive_field_check_marks_a_reply_still_missing_it() -> None:
    """Belt and suspenders: unreachable against a well-behaved PlannerPort
    (see reasoning/completeness.py's module docstring), reachable only
    against a fake that redirects to retrieval with erp_field still unmet."""
    contract = ReplyContract(
        needs=frozenset({"document_passage", "erp_field"}), document_query="why"
    )
    graph = nodes(composer=FakeComposer(grounded("m2-status.md")))

    result = graph.retrieve_and_answer(state(route="retrieve_project_documents", contract=contract))

    assert result.route == "answer"
    assert result.failure == "incomplete_reply"
    assert "erp_field" in (result.error_detail or "")
    assert any(
        event.kind == "contract_enforced" and "unmet after redirect: erp_field" in event.detail
        for event in result.events
    )


def test_a_composer_refusal_is_passed_through_as_a_refusal() -> None:
    graph = nodes(
        composer=FakeComposer(
            GroundedAnswer(
                answer="",
                citations=[],
                grounded=False,
                confidence=0.2,
                refusal_reason="The passages do not mention M2.",
            )
        )
    )

    result = graph.retrieve_and_answer(state(route="retrieve_project_documents"))

    assert result.route == "refuse"
    assert result.response == "The passages do not mention M2."
    assert result.failure == "insufficient_evidence"


def test_an_answer_citing_a_source_nobody_retrieved_is_refused() -> None:
    """The check the whole grounding claim rests on: a composer is trusted to
    write, never to have cited something real."""
    graph = nodes(composer=FakeComposer(grounded("ghost-report.md")))

    result = graph.retrieve_and_answer(state(route="retrieve_project_documents"))

    assert result.route == "refuse"
    assert "ghost-report.md" in (result.error_detail or "")
    assert "M2 slipped" not in (result.response or "")


def test_a_composer_that_raises_ends_the_turn_as_a_provider_failure() -> None:
    graph = nodes(composer=FakeComposer(raises=ConnectionError("reset")))

    result = graph.retrieve_and_answer(state(route="retrieve_project_documents"))

    assert (result.route, result.failure) == ("fail", "provider_failure")
    assert result.evidence == (snippet(),)


def test_a_retriever_that_raises_ends_the_turn_as_a_provider_failure() -> None:
    graph = nodes(retriever=RaisingRetriever(RuntimeError("embeddings down")))

    result = graph.retrieve_and_answer(state(route="retrieve_project_documents"))

    assert (result.route, result.failure) == ("fail", "provider_failure")
    assert result.evidence == ()
    assert "embeddings down" in (result.error_detail or "")
    assert kinds(result)[-1] == "failed"


# --------------------------------------------------------------------------
# execute_tool
# --------------------------------------------------------------------------


def executing(**changes) -> AgentState:
    return state(
        route="call_tool",
        tool_name="list_risks",
        tool_arguments={"project_id": "atlas"},
        **changes,
    )


def test_the_gateway_is_handed_the_actor_the_scopes_and_the_approval() -> None:
    """This node never decides that a call is permitted; it reports what it was
    told and lets the boundary decide."""
    gateway = FakeGateway(outcome())
    graph = nodes(tools=gateway)

    graph.execute_tool(executing())

    request = gateway.requests[0]
    assert (request.actor, request.scopes) == ("bao", SCOPES)
    assert request.tool_name == "list_risks"
    assert request.arguments == {"project_id": "atlas"}


def test_a_successful_call_hands_its_observation_back_to_the_planner() -> None:
    """Not straight to an answer: the model has to see what the call returned."""
    graph = nodes()

    result = graph.execute_tool(executing())

    assert result.route == "think"
    assert result.observations == (outcome(),)
    assert "tool_called" in kinds(result)
    assert not result.terminal


def test_a_throttled_call_waits_and_stays_on_the_same_route() -> None:
    """A retry is a repeated node execution, never a graph edge."""
    waited: list[float] = []
    graph = nodes(
        tools=FakeGateway(outcome("rate_limited", retry_after_seconds=2.5)),
        sleep=waited.append,
    )

    result = graph.execute_tool(executing())

    assert result.route == "call_tool"
    assert result.retry_count == 1
    assert waited == [2.5]
    assert "retry_scheduled" in kinds(result)
    assert "rate_limited" in kinds(result)


def test_a_throttle_that_outlives_the_budget_fails_the_turn() -> None:
    graph = nodes(tools=FakeGateway(outcome("rate_limited", retry_after_seconds=1.0)))

    result = graph.execute_tool(executing(retry_count=1))

    assert (result.route, result.failure) == ("fail", "tool_failure")
    assert "rate limited" in (result.error_detail or "")
    assert result.terminal


def test_a_zero_retry_budget_reports_the_throttle_without_waiting() -> None:
    waited: list[float] = []
    graph = nodes(
        tools=FakeGateway(outcome("rate_limited", retry_after_seconds=9.0)),
        tool_retry_budget=0,
        sleep=waited.append,
    )

    result = graph.execute_tool(executing())

    assert result.failure == "tool_failure"
    assert waited == []


def test_a_read_the_registry_escalated_is_routed_to_a_human() -> None:
    """Only the registry knows a read was put in front of an approver, so an
    approval_required coming back is policy working, not a contradiction."""
    graph = nodes(tools=FakeGateway(outcome("approval_required", error="needs a yes")))

    result = graph.execute_tool(executing())

    assert (result.route, result.approval) == ("request_approval", "pending")
    assert "approval_requested" in kinds(result)
    assert not result.terminal


@pytest.mark.parametrize(
    "status", ["denied", "invalid_arguments", "transient_failure", "failed"]
)
def test_every_other_refusal_ends_the_turn_with_the_status_recorded(
    status: str,
) -> None:
    graph = nodes(tools=FakeGateway(outcome(status, error="no")))

    result = graph.execute_tool(executing())

    assert (result.route, result.failure) == ("fail", "tool_failure")
    assert status in (result.error_detail or "")
    assert result.observations[-1].status == status


def test_an_approved_write_may_be_executed() -> None:
    gateway = FakeGateway(
        ToolOutcome(
            tool_name="create_risk",
            status="ok",
            summary="Recorded R-9.",
            source_ids=("risk:R-9",),
        )
    )
    graph = nodes(tools=gateway)

    result = graph.execute_tool(
        state(
            route="call_tool",
            tool_name="create_risk",
            tool_arguments={"project_id": "atlas", "title": "x", "severity": "low"},
            tool_mutating=True,
            approval="approved",
        )
    )

    assert gateway.requests[0].approval == "approved"
    assert result.route == "think"


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


def test_an_unrouted_state_starts_by_thinking() -> None:
    """None is a real key: a table starting at the second step would leave the
    first one undispatched."""
    graph = nodes()

    assert graph.table()[None] == graph.table()["think"]


def test_the_table_covers_every_route_a_node_runs_on() -> None:
    assert set(nodes().table()) == {
        None,
        "think",
        "retrieve_project_documents",
        "call_tool",
    }
