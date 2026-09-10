"""A transition table proves nothing until the runtime cannot go around it."""

import ast
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.reasoning.decision import DecisionRoute
from agentic_erp_assistant.engine import transitions
from agentic_erp_assistant.engine.transitions import (
    ALLOWED,
    advance,
    assert_transition,
    IllegalTransition,
    TERMINAL_ROUTES,
)
from agentic_erp_assistant.state.agent_state import AgentState

ROUTES: tuple[DecisionRoute, ...] = get_args(DecisionRoute)


def at(route: DecisionRoute | None = None, **changes: object) -> AgentState:
    """A state sitting on ``route``, with whatever else the case needs."""
    base = AgentState(request="How is M2 tracking?", actor="bao", trace_id="run-1")
    return base.evolve(route=route, **changes)


def waiting_on_approval(decision: str = "pending") -> AgentState:
    return at("request_approval", tool_name="close_milestone", approval=decision)


# --------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------


def test_every_route_has_a_row_including_the_unrouted_start() -> None:
    """A route added to DecisionRoute without its edges decided must fail here,
    not later as a move nobody reviewed."""
    assert set(ALLOWED) == {None, *ROUTES}


def test_every_declared_target_is_a_real_route() -> None:
    for origin, targets in ALLOWED.items():
        assert targets <= set(ROUTES), origin


def test_the_terminal_routes_are_exactly_the_rows_with_no_way_out() -> None:
    """Two statements of the same fact, checked against each other, so neither
    can drift into being the quiet one."""
    dead_ends = {origin for origin, targets in ALLOWED.items() if not targets}

    assert dead_ends == set(TERMINAL_ROUTES)


@pytest.mark.parametrize("origin", sorted(TERMINAL_ROUTES))
@pytest.mark.parametrize("target", ROUTES)
def test_a_terminal_route_has_no_legal_move_at_all(
    origin: DecisionRoute, target: DecisionRoute
) -> None:
    """Every target, not a sampled one: 'the turn is over' has to hold against
    the whole vocabulary."""
    ending = at(origin, response="done")

    with pytest.raises(IllegalTransition, match="ends the turn"):
        assert_transition(ending, target, mutating=False)


def test_the_first_move_of_a_run_is_guarded_like_any_other() -> None:
    """The unrouted start is a real row, not an exemption."""
    assert None in ALLOWED
    assert ALLOWED[None]


def test_an_undeclared_origin_raises_instead_of_permitting_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure mode a missing row must not have: silently unrestricted."""
    monkeypatch.delitem(transitions.ALLOWED, "call_tool")

    with pytest.raises(IllegalTransition, match="no row in ALLOWED"):
        assert_transition(at("call_tool"), "answer", mutating=False)


def test_an_undeclared_move_between_two_real_routes_raises() -> None:
    """Retrieval cannot reach a tool call today -- there is no node for it, so
    the table must not describe a path the runtime cannot walk."""
    with pytest.raises(IllegalTransition, match="not a declared move"):
        assert_transition(
            at("retrieve_project_documents"), "call_tool", mutating=False
        )


def test_retrieval_cannot_loop_back_to_itself() -> None:
    with pytest.raises(IllegalTransition, match="not a declared move"):
        assert_transition(
            at("retrieve_project_documents"), "retrieve_project_documents"
        )


# --------------------------------------------------------------------------
# The two rules that are not about routes alone
# --------------------------------------------------------------------------


def test_a_pending_approval_cannot_proceed_to_the_call() -> None:
    with pytest.raises(IllegalTransition, match="pending"):
        assert_transition(waiting_on_approval("pending"), "call_tool", mutating=True)


def test_a_call_that_waited_for_a_human_stays_waiting_even_when_read_only() -> None:
    """The two approval rules are not the same rule. The mutating one is about
    what the tool does; this one is about what the graph already decided -- once
    a call has been put to a human, an unanswered human is not a yes, whatever
    the tool touches."""
    escalated = at("request_approval", tool_name="export_full_ledger")

    with pytest.raises(IllegalTransition, match="granted"):
        assert_transition(escalated, "call_tool", mutating=False)


def test_a_denied_approval_cannot_proceed_to_the_call() -> None:
    with pytest.raises(IllegalTransition):
        assert_transition(waiting_on_approval("denied"), "call_tool", mutating=True)


def test_a_granted_approval_may_proceed_to_the_call() -> None:
    assert_transition(waiting_on_approval("approved"), "call_tool", mutating=True)


def test_a_denied_approval_ends_the_turn_in_a_refusal() -> None:
    assert_transition(waiting_on_approval("denied"), "refuse")


def test_a_mutating_call_cannot_be_left_for_a_reply_without_approval() -> None:
    """The unit's rule: no jump from a write straight to a finished answer."""
    ran_unapproved = at("call_tool", tool_name="close_milestone")

    with pytest.raises(IllegalTransition, match="mutates ERP data"):
        assert_transition(ran_unapproved, "answer", mutating=True)


def test_a_mutating_call_cannot_even_be_entered_without_approval() -> None:
    """Stricter than guarding the exit, deliberately. A check that only fires
    on the way out lets the write reach the node, execute, and be objected to
    afterwards -- which is not a guard."""
    with pytest.raises(IllegalTransition, match="mutates ERP data"):
        assert_transition(
            at(None, tool_name="close_milestone"), "call_tool", mutating=True
        )


def test_an_approved_mutating_call_may_be_entered_and_left() -> None:
    approved = at("call_tool", tool_name="close_milestone", approval="approved")

    assert_transition(approved, "answer", mutating=True)


def test_a_read_only_call_needs_no_approval() -> None:
    """mutating is read from the tool's own declaration, so a read-only tool is
    not slowed down by a rule it is not subject to."""
    reading = at("call_tool", tool_name="get_project_status")

    assert_transition(reading, "answer", mutating=False)


def test_a_call_transition_must_declare_whether_the_tool_mutates() -> None:
    """Not defaulted to False: a default is exactly how this check would be
    skipped in silence, by the caller who forgot to look the tool up."""
    with pytest.raises(IllegalTransition, match="must declare"):
        assert_transition(at("call_tool", tool_name="close_milestone"), "answer")


def test_a_transition_that_touches_no_tool_need_not_declare_anything() -> None:
    assert_transition(at("retrieve_project_documents"), "answer")


def test_an_illegal_transition_is_a_runtime_error() -> None:
    """A bug in the graph, not a condition it can route on -- unlike a tool
    failure, which comes back as data."""
    assert issubclass(IllegalTransition, RuntimeError)


# --------------------------------------------------------------------------
# advance: the guard is on the road, not beside it
# --------------------------------------------------------------------------


def test_advance_applies_a_legal_move() -> None:
    moved = advance(at(), "retrieve_project_documents")

    assert moved.route == "retrieve_project_documents"
    assert moved.terminal is False


def test_advance_refuses_an_illegal_move() -> None:
    with pytest.raises(IllegalTransition):
        advance(waiting_on_approval("pending"), "call_tool", mutating=True)


def test_a_refused_move_leaves_nothing_half_applied() -> None:
    """It checks before it builds, so the state a node was handed is intact
    when the exception reaches the engine."""
    state = waiting_on_approval("pending")
    before = state.model_dump()

    with pytest.raises(IllegalTransition):
        advance(state, "call_tool", mutating=True, response="slipped through")

    assert state.model_dump() == before


def test_advance_carries_other_field_changes_in_the_same_transition() -> None:
    answered = advance(at("call_tool", tool_name="get_project_status"), "answer",
                       mutating=False, response="M2 is two days behind.")

    assert answered.response == "M2 is two days behind."


@pytest.mark.parametrize("route", sorted(TERMINAL_ROUTES))
def test_advance_marks_a_terminal_route_terminal(route: DecisionRoute) -> None:
    """The table is the authority on what ends a turn; a node that could
    disagree would be a second, quieter table."""
    changes = {"response": "done"} if route != "fail" else {"failure": "provider_failure"}

    assert advance(at(), route, **changes).terminal is True


def test_a_terminal_move_still_has_to_bring_a_response_or_a_failure() -> None:
    """advance sets the flag; AgentState is what refuses to end saying nothing."""
    with pytest.raises(ValidationError, match="terminal"):
        advance(at(), "answer")


@pytest.mark.parametrize("owned", ["route", "terminal"])
def test_advance_will_not_take_a_second_opinion_on_what_it_decides(
    owned: str,
) -> None:
    with pytest.raises(TypeError, match=owned):
        advance(at(), "retrieve_project_documents", **{owned: "answer"})


# --------------------------------------------------------------------------
# The mistake the unit warns about, checked against the source
# --------------------------------------------------------------------------


ENGINE_PACKAGE = (
    Path(__file__).resolve().parents[2] / "src" / "agentic_erp_assistant" / "engine"
)


def sets_route_directly(source: Path) -> list[int]:
    """Line numbers where this file changes ``route`` without the guard."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    lines: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        writes_state = (isinstance(target, ast.Attribute) and target.attr == "evolve") or (
            isinstance(target, ast.Name) and target.id == "AgentState"
        )
        if writes_state and any(word.arg == "route" for word in node.keywords):
            lines.append(node.lineno)

    return lines


def test_no_other_runtime_module_sets_route_behind_the_guards_back() -> None:
    """The failure the unit names: a table written, then never called, proving
    only that its own test passes. Every node has to go through advance(), and
    this is what makes 'has to' mean something."""
    offenders = {
        source.name: sets_route_directly(source)
        for source in sorted(ENGINE_PACKAGE.rglob("*.py"))
        if source.name != "transitions.py" and sets_route_directly(source)
    }

    assert offenders == {}


def test_the_guard_would_actually_catch_an_offender(tmp_path: Path) -> None:
    """A check that cannot fail proves nothing about the code it guards."""
    node = tmp_path / "sneaky_node.py"
    node.write_text('def run(state):\n    return state.evolve(route="answer")\n')

    assert sets_route_directly(node) == [2]


def test_transitions_is_the_one_module_that_does_set_route() -> None:
    """And it is excluded above by name, so the exemption is one file, visible."""
    assert sets_route_directly(ENGINE_PACKAGE / "transitions.py")


# --------------------------------------------------------------------------
# The cycle: an action can hand back to a second thought
# --------------------------------------------------------------------------


def test_a_turn_starts_by_thinking() -> None:
    state = AgentState(request="How is M2 tracking?", actor="bao", trace_id="run-1")

    assert advance(state, "think").route == "think"


def test_a_read_tool_can_hand_its_observation_back_to_the_planner() -> None:
    """Without this edge the graph is a single shot and the step budget guards
    nothing."""
    state = AgentState(
        request="q",
        actor="bao",
        trace_id="run-1",
        route="call_tool",
        tool_name="list_risks",
    )

    assert advance(state, "think", mutating=False).route == "think"


def test_retrieval_can_hand_back_to_the_planner() -> None:
    state = AgentState(
        request="q", actor="bao", trace_id="run-1", route="retrieve_project_documents"
    )

    assert advance(state, "think").route == "think"


def test_a_second_thought_is_not_a_move_the_table_allows() -> None:
    """A think that produced no route is a planner bug, not an edge."""
    state = AgentState(request="q", actor="bao", trace_id="run-1", route="think")

    with pytest.raises(IllegalTransition):
        advance(state, "think")


def test_an_unapproved_write_still_cannot_leave_call_tool_for_a_new_thought() -> None:
    state = AgentState(
        request="q",
        actor="bao",
        trace_id="run-1",
        route="call_tool",
        tool_name="create_risk",
        approval="pending",
    )

    with pytest.raises(IllegalTransition, match="approval"):
        advance(state, "think", mutating=True)
