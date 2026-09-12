"""Planner.declare: turning a DecisionModel.declare() reply into a typed
ReplyContract, never raising for an unreadable one (ADR 0021).

Every unreadable shape -- prose, the wrong function, arguments the schema
rejects, a query/needs mismatch -- is a turn that goes unchecked
(EMPTY_CONTRACT), not a turn that fails. A provider exception is the one
thing this method does not absorb; that is the orchestrator's job.
"""

import logging
from datetime import UTC, datetime

import pytest

from agentic_erp_assistant.llm.tools import ToolCallResult
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.reply_contract import EMPTY_CONTRACT, ReplyContract


class ScriptedDeclarer:
    """A DecisionModel stand-in exercising only declare()."""

    def __init__(self, result: ToolCallResult) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def decide(self, *args, **kwargs):  # pragma: no cover - unused here
        raise AssertionError("decide() is exercised in test_planner.py")

    def declare(self, question, history=()):
        self.calls.append((question, tuple(history)))
        return self.result


def declared(needs: list[str], document_query: str | None = None) -> ToolCallResult:
    arguments = {"needs": needs, "document_query": document_query}
    return ToolCallResult.from_tool_call(
        tool_name="declare_reply_contract", arguments=arguments
    )


def state(request: str = "Why is milestone M2 late and by how much?", **changes) -> AgentState:
    base = AgentState(
        request=request, actor="bao", project_code="atlas", trace_id="run-1"
    )
    return base.evolve(**changes) if changes else base


def test_both_needs_with_a_query_builds_the_contract() -> None:
    model = ScriptedDeclarer(declared(["document_passage", "erp_field"], "why is M2 late"))

    contract = Planner(model).declare(state())

    assert contract == ReplyContract(
        needs=frozenset({"document_passage", "erp_field"}),
        document_query="why is M2 late",
    )


def test_erp_field_alone_needs_no_query() -> None:
    model = ScriptedDeclarer(declared(["erp_field"], None))

    contract = Planner(model).declare(state())

    assert contract == ReplyContract(needs=frozenset({"erp_field"}))


def test_no_needs_is_a_real_declaration() -> None:
    model = ScriptedDeclarer(declared([], None))

    contract = Planner(model).declare(state())

    assert contract.needs == frozenset()


def test_the_question_and_history_reach_the_model() -> None:
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
    model = ScriptedDeclarer(declared(["erp_field"]))

    Planner(model).declare(state(history=(turn,)))

    question, history = model.calls[0]
    assert question == "Why is milestone M2 late and by how much?"
    assert history == (turn,)


def test_prose_instead_of_a_call_is_treated_as_unchecked(caplog) -> None:
    model = ScriptedDeclarer(ToolCallResult.from_content("both, I think"))

    with caplog.at_level(logging.WARNING):
        contract = Planner(model).declare(state())

    assert contract == EMPTY_CONTRACT
    assert "prose" in caplog.text or "declare_reply_contract" in caplog.text


def test_a_different_function_is_treated_as_unchecked(caplog) -> None:
    model = ScriptedDeclarer(
        ToolCallResult.from_tool_call(
            tool_name="get_project_status", arguments={"milestone_id": "M2"}
        )
    )

    with caplog.at_level(logging.WARNING):
        contract = Planner(model).declare(state())

    assert contract == EMPTY_CONTRACT
    assert "get_project_status" in caplog.text


def test_arguments_the_schema_rejects_are_treated_as_unchecked(caplog) -> None:
    """A needs value outside the two-member enum -- exactly what strict
    function calling should prevent in production, and exactly the shape a
    non-conforming or future model could still produce."""
    model = ScriptedDeclarer(
        ToolCallResult.from_tool_call(
            tool_name="declare_reply_contract",
            arguments={"needs": ["something_else"], "document_query": None},
        )
    )

    with caplog.at_level(logging.WARNING):
        contract = Planner(model).declare(state())

    assert contract == EMPTY_CONTRACT


def test_a_query_naming_document_passage_but_none_given_is_treated_as_unchecked(
    caplog,
) -> None:
    """The raw arguments validate on their own (document_query is nullable),
    but ReplyContract's cross-field rule -- a query is required exactly when
    document_passage is declared -- still catches the mismatch."""
    model = ScriptedDeclarer(declared(["document_passage"], None))

    with caplog.at_level(logging.WARNING):
        contract = Planner(model).declare(state())

    assert contract == EMPTY_CONTRACT


def test_a_provider_exception_propagates() -> None:
    """Not absorbed here -- a call that never produced a reply is not a
    garbled reply, and the caller decides what that means."""

    class RaisingModel:
        def decide(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

        def declare(self, question, history=()):
            raise RuntimeError("transport failure")

    with pytest.raises(RuntimeError, match="transport failure"):
        Planner(RaisingModel()).declare(state())
