"""What code alone can say about evicted turns, and what a model may add to it.

Every way the proposer's answer can be unusable ends in folding the turns
structurally rather than in a failed request -- the same severity judgement
tests/memory/test_extractor.py makes about the long-term proposer.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from tests.memory.builders import RECORDED, make_turn

from agentic_erp_assistant.llm.tools import DEFAULT_TOOLS, PLANNING_TOOLS, ToolCallResult, ToolSpec
from agentic_erp_assistant.memory.promotion import (
    LLMSessionSummaryProposer,
    MAX_ITEMS_PER_SECTION,
    PROPOSABLE_SECTIONS,
    PROPOSE_SESSION_SUMMARY_TOOL,
    SessionSummaryProposal,
    conversation_state,
    structural_state,
)


class FakeModel:
    """Returns a scripted choice and records what it was asked."""

    def __init__(self, result: ToolCallResult) -> None:
        self.result = result
        self.messages: list[Sequence[Any]] = []
        self.offered: list[Sequence[ToolSpec]] = []

    def call_tools(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
    ) -> ToolCallResult:
        self.messages.append(messages)
        self.offered.append(tools)
        return self.result


def proposal_arguments(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "user_goal": None,
        "decisions": [],
        "unresolved_questions": [],
        "accepted_facts": [],
    }
    fields.update(overrides)
    return fields


def called(**overrides: object) -> ToolCallResult:
    return ToolCallResult.from_tool_call(
        tool_name=PROPOSE_SESSION_SUMMARY_TOOL.name,
        arguments=proposal_arguments(**overrides),
    )


def proposal(**overrides: object) -> SessionSummaryProposal:
    return SessionSummaryProposal(**proposal_arguments(**overrides))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# structural_state: what code alone can say
# --------------------------------------------------------------------------


def test_structural_state_takes_the_goal_from_the_oldest_turn() -> None:
    older = make_turn(trace_id="run-0", request="get the cutover scheduled")
    newer = make_turn(
        trace_id="run-1",
        request="and what about the budget?",
        started_at=RECORDED.replace(minute=RECORDED.minute + 1),
        finished_at=RECORDED.replace(minute=RECORDED.minute + 1),
    )

    state = structural_state((newer, older))

    assert state["user_goal"] == "get the cutover scheduled"


def test_clarify_turns_become_unresolved_questions() -> None:
    clarifying = make_turn(trace_id="run-0", route="clarify", request="which project?")

    state = structural_state((clarifying,))

    assert state["unresolved_questions"] == ["which project?"]


def test_a_paused_turn_becomes_a_pending_approval() -> None:
    paused = make_turn(
        trace_id="run-0",
        route="request_approval",
        response=None,
        approval="pending",
        tool_name="create_risk",
    )

    state = structural_state((paused,))

    assert state["pending_approvals"] == ["create_risk awaiting approval (run run-0)"]


def test_no_turns_means_an_empty_state() -> None:
    assert structural_state(()) == {}


# --------------------------------------------------------------------------
# conversation_state: the proposal overlays only what it is allowed to
# --------------------------------------------------------------------------


def test_pending_approvals_are_never_taken_from_the_proposal() -> None:
    paused = make_turn(
        trace_id="run-0",
        route="request_approval",
        response=None,
        approval="pending",
        tool_name="create_risk",
    )

    # A proposal cannot even name pending_approvals -- it is not a field on
    # SessionSummaryProposal. This proves the merge does not invent one either.
    state = conversation_state((paused,), proposal(user_goal="something else"))

    assert state["pending_approvals"] == ["create_risk awaiting approval (run run-0)"]


def test_the_proposal_overlays_only_the_four_proposable_sections() -> None:
    turn = make_turn(trace_id="run-0", request="get the cutover scheduled")

    state = conversation_state(
        (turn,),
        proposal(
            user_goal="schedule the cutover",
            decisions=["cutover moves to Thursday"],
            unresolved_questions=["who signs off?"],
            accepted_facts=["the vendor confirmed readiness"],
        ),
    )

    assert state["user_goal"] == "schedule the cutover"
    assert state["decisions"] == ["cutover moves to Thursday"]
    assert state["unresolved_questions"] == ["who signs off?"]
    assert state["accepted_facts"] == ["the vendor confirmed readiness"]
    assert set(state) <= {*PROPOSABLE_SECTIONS, "pending_approvals"}


def test_a_null_goal_leaves_the_structural_goal_in_place() -> None:
    turn = make_turn(trace_id="run-0", request="get the cutover scheduled")

    state = conversation_state((turn,), proposal())

    assert state["user_goal"] == "get the cutover scheduled"


def test_no_proposal_is_the_structural_state_alone() -> None:
    turn = make_turn(trace_id="run-0", request="get the cutover scheduled")

    assert conversation_state((turn,), None) == structural_state((turn,))


def test_lists_are_capped_after_merging() -> None:
    turn = make_turn(trace_id="run-0", route="clarify", request="q0")

    state = conversation_state(
        (turn,), proposal(unresolved_questions=["q1", "q2", "q3"])
    )

    assert len(state["unresolved_questions"]) == MAX_ITEMS_PER_SECTION


# --------------------------------------------------------------------------
# LLMSessionSummaryProposer: every unusable answer means "propose nothing"
# --------------------------------------------------------------------------


def test_a_well_formed_call_becomes_a_proposal() -> None:
    model = FakeModel(called(user_goal="schedule the cutover"))
    turns = (make_turn(trace_id="run-0"),)

    result = LLMSessionSummaryProposer(model=model).propose(turns, previous=None)

    assert result is not None
    assert result.user_goal == "schedule the cutover"


def test_prose_reply_means_no_proposal() -> None:
    model = FakeModel(ToolCallResult.from_content("Nothing durable happened."))

    result = LLMSessionSummaryProposer(model=model).propose(
        (make_turn(trace_id="run-0"),), previous=None
    )

    assert result is None


def test_wrong_tool_means_no_proposal() -> None:
    model = FakeModel(
        ToolCallResult.from_tool_call(tool_name="propose_memories", arguments={"candidates": []})
    )

    result = LLMSessionSummaryProposer(model=model).propose(
        (make_turn(trace_id="run-0"),), previous=None
    )

    assert result is None


def test_bad_arguments_mean_no_proposal() -> None:
    model = FakeModel(
        ToolCallResult.from_tool_call(
            tool_name=PROPOSE_SESSION_SUMMARY_TOOL.name, arguments={"user_goal": 5}
        )
    )

    result = LLMSessionSummaryProposer(model=model).propose(
        (make_turn(trace_id="run-0"),), previous=None
    )

    assert result is None


# --------------------------------------------------------------------------
# The tool is asked for on its own, not offered mid-turn
# --------------------------------------------------------------------------


def test_the_tool_is_not_on_any_planning_menu() -> None:
    assert PROPOSE_SESSION_SUMMARY_TOOL not in DEFAULT_TOOLS
    assert PROPOSE_SESSION_SUMMARY_TOOL not in PLANNING_TOOLS
