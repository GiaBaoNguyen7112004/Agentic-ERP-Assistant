"""Which moves the graph is allowed to make, as a table rather than as code.

Every edge this graph has is a row in :data:`ALLOWED`. That is the design
document and the enforcement at once: adding a path through the runtime means
editing a table a reviewer can read in one screen, and it shows up in a diff as
a decision instead of hiding in a new ``elif`` three files away.

The guard has to be on the road, not beside it
----------------------------------------------

A transition table is easy to write and easy to leave unused -- it then proves
nothing except that its own test passes. So the table is not offered to nodes
as advice. :func:`advance` is the only supported way to change
:attr:`~agentic_erp_assistant.state.agent_state.AgentState.route`, it calls
:func:`assert_transition` before it builds anything, and
``tests/runtime/test_transitions.py`` parses every other file under
``runtime/`` and fails if one sets ``route`` itself. A node cannot take an
illegal step by accident, and it cannot take one on purpose without the diff
showing that it went around the front door.

Why an illegal move raises
--------------------------

Everywhere else in this runtime a failure is data: a tool that was denied or
timed out comes back as a value the graph routes on, because those are things
that happen to a working system. An illegal transition is not that. It means
the graph itself is wrong -- a node moved somewhere no design ever allowed --
and there is no sensible route for it, no message for a user, and nothing for
an approver to decide. :class:`IllegalTransition` is a ``RuntimeError`` for the
same reason an assertion is not a return value.

What the table deliberately leaves out
--------------------------------------

``retrieve_project_documents`` cannot reach ``call_tool``. Looking something up
and then acting on it is a reasonable flow, and there is no node for it yet;
allowing the edge now would mean the table describes a path the runtime cannot
walk. It also cannot reach itself: a second retrieval pass is a real design
question (what changed about the query? when does it stop?) and opening the
edge before answering that invites a loop with nothing but the step budget
between it and the user.

Both are one line to add, in a file where adding them is visible.
"""

from collections.abc import Mapping
from typing import Any

from agentic_erp_assistant.reasoning.decision import DecisionRoute
from agentic_erp_assistant.state.agent_state import AgentState

__all__ = [
    "ALLOWED",
    "advance",
    "assert_transition",
    "IllegalTransition",
    "TERMINAL_ROUTES",
]


TERMINAL_ROUTES: frozenset[DecisionRoute] = frozenset(
    {"answer", "clarify", "refuse", "fail"}
)
"""The routes a turn ends on. Each maps to no outgoing edge at all.

``clarify`` is here, which is the one worth arguing. Asking the user a question
back ends this turn: their reply arrives as a new request, with its own trace,
its own budget and its own routing. Treating a clarification as a step inside
the same turn would make one graph run span an unbounded wait for a human and
leave the step budget measuring something that is not work.
"""


_TOOL_EXECUTION_ROUTE: DecisionRoute = "call_tool"
"""The one route on which a tool actually runs, and so the one the approval
rule is written around."""


ALLOWED: Mapping[DecisionRoute | None, frozenset[DecisionRoute]] = {
    # A turn that has not routed yet. Present as a real row because the first
    # move of every run is a transition too, and a table that started at the
    # second one would leave it unguarded.
    None: frozenset(
        {
            "think",
            "retrieve_project_documents",
            "call_tool",
            "request_approval",
            "answer",
            "clarify",
            "refuse",
            "fail",
        }
    ),
    # Where a turn decides. Every action is reachable from here, and so is
    # every ending: a planner that finds nothing worth doing must be able to
    # say so without first pretending to act.
    "think": frozenset(
        {
            "retrieve_project_documents",
            "call_tool",
            "request_approval",
            "answer",
            "clarify",
            "refuse",
            "fail",
        }
    ),
    # Retrieval grounds a reply, discovers it cannot, or hands back to a second
    # thought -- passages can reveal that a tool call is what was needed.
    "retrieve_project_documents": frozenset(
        {"think", "answer", "clarify", "refuse", "fail"}
    ),
    # A tool ran. Its result answers the question, or feeds the next thought,
    # or the turn broke. The edge back to "think" is what makes this a cycle
    # rather than a single shot, and it is the reason the runtime carries a
    # step budget: an unbounded cycle needs a guard, not an absent edge.
    #
    # The edge to request_approval is the escalated read: the registry may put
    # a non-mutating tool in front of a human, the gateway refuses the call
    # with "approval_required", and the node has to be able to go and ask.
    # Without it that policy would be unreachable and would fail the turn.
    "call_tool": frozenset({"think", "request_approval", "answer", "fail"}),
    # Waiting on a human: granted goes on to the call, denied ends in a
    # refusal, and anything breaking on the way ends in a failure.
    "request_approval": frozenset({"call_tool", "refuse", "fail"}),
    # The four terminal routes. Empty, not absent: "this turn is over" is a
    # declared fact, and a missing row would be an undeclared one.
    "answer": frozenset(),
    "clarify": frozenset(),
    "refuse": frozenset(),
    "fail": frozenset(),
}
"""Every legal move, keyed by the route being left.

``None`` is the starting row. A route missing from this mapping is not treated
as unrestricted -- :func:`assert_transition` raises on it -- so adding a member
to :data:`~agentic_erp_assistant.reasoning.decision.DecisionRoute` without
deciding its edges fails loudly rather than quietly permitting everything.
"""


class IllegalTransition(RuntimeError):
    """The graph tried to make a move no design allows.

    A bug in the runtime, not a condition it can recover from -- see the module
    docstring for why this is raised rather than routed.
    """


def assert_transition(
    state: AgentState,
    target: DecisionRoute,
    *,
    mutating: bool | None = None,
) -> None:
    """Raise unless moving ``state`` to ``target`` is a declared, permitted move.

    Takes the whole state rather than a pair of routes because the two rules
    that matter most are not about routes alone: both read
    :attr:`~agentic_erp_assistant.state.agent_state.AgentState.approval`, and
    asking a caller to unpack it and hand it back would create the one place
    where the wrong value could be supplied.

    Args:
        state: The state being moved, whose ``route`` is the move's origin.
        target: The route being moved to.
        mutating: Whether the tool named in the state changes ERP data. Read
            from the tool's own
            :attr:`~agentic_erp_assistant.llm.tools.ToolSpec.mutating` flag by
            the caller -- this module does not import the tool registry, which
            is what keeps the runtime free of the ``llm`` package. Required
            whenever ``call_tool`` is the origin or the target, and ignored
            otherwise; see below for why it is not simply defaulted to
            ``False``.

    Raises:
        IllegalTransition: The origin route is not declared in :data:`ALLOWED`;
            or the move is not among that row's targets; or a tool would run,
            or has run, without the approval it needs; or ``mutating`` was not
            declared for a transition that turns on it.
    """
    origin = state.route

    if origin not in ALLOWED:
        raise IllegalTransition(
            f"{origin!r} has no row in ALLOWED, so every move out of it is "
            f"undeclared. A route added to DecisionRoute needs its edges "
            f"decided here."
        )

    if target not in ALLOWED[origin]:
        if origin in TERMINAL_ROUTES:
            raise IllegalTransition(
                f"{origin!r} ends the turn; there is no move from it to "
                f"{target!r}"
            )
        raise IllegalTransition(f"{origin!r} -> {target!r} is not a declared move")

    touches_execution = _TOOL_EXECUTION_ROUTE in (origin, target)

    # Not defaulted to False. A default is precisely how this check would be
    # skipped in silence: the caller that forgot to look the tool up is the
    # caller whose write must not run.
    if touches_execution and mutating is None:
        raise IllegalTransition(
            f"{origin!r} -> {target!r} runs or has just run a tool, so the "
            f"caller must declare whether that tool mutates; it was not passed"
        )

    # The rule reads on both sides of call_tool, not only on the way out of it.
    # Guarding the exit alone would let an unapproved write reach the node,
    # execute, and be refused afterwards -- which is not a guard.
    if touches_execution and mutating and state.approval != "approved":
        raise IllegalTransition(
            f"{state.tool_name!r} mutates ERP data and approval is "
            f"{state.approval!r}; a write may not be entered or left on route "
            f"{_TOOL_EXECUTION_ROUTE!r} without a recorded approval"
        )

    # A separate rule from the one above, not a repetition of it. That one is
    # about what the tool does; this one is about what the graph already
    # decided. Once a call has been put to a human -- for any reason, including
    # a read-only tool escalated by policy -- an unanswered human is not a yes.
    if (
        origin == "request_approval"
        and target == _TOOL_EXECUTION_ROUTE
        and state.approval != "approved"
    ):
        raise IllegalTransition(
            f"approval is {state.approval!r}; a call waiting on a human may "
            f"only proceed once that human has granted it"
        )


def advance(
    state: AgentState,
    target: DecisionRoute,
    *,
    mutating: bool | None = None,
    **changes: Any,
) -> AgentState:
    """Move ``state`` to ``target``, or raise. The only way a node changes route.

    Checks first and builds second, so a rejected move leaves nothing
    half-applied. ``terminal`` is set from :data:`TERMINAL_ROUTES` rather than
    accepted from the caller: the table is the authority on what ends a turn,
    and a node that could disagree with it would be a second, quieter table.
    Supplying a response or a failure alongside a terminal move is still the
    node's job -- :class:`~agentic_erp_assistant.state.agent_state.AgentState`
    refuses to be terminal without one.

    Args:
        state: The state to move.
        target: The route to move to.
        mutating: Passed through to :func:`assert_transition`.
        **changes: Any other field updates to apply in the same transition,
            handed to
            :meth:`~agentic_erp_assistant.state.agent_state.AgentState.evolve`
            and validated there.

    Returns:
        The new state, with ``route`` set to ``target``.

    Raises:
        IllegalTransition: The move is not allowed.
        TypeError: ``route`` or ``terminal`` was passed in ``changes``; both
            are decided here and accepting them would create two answers.
        ValidationError: The resulting state is not one the runtime allows.
    """
    for owned in ("route", "terminal"):
        if owned in changes:
            raise TypeError(
                f"advance() decides {owned!r} from the target route; passing "
                f"it as a change would give the transition two answers"
            )

    assert_transition(state, target, mutating=mutating)

    return state.evolve(route=target, terminal=target in TERMINAL_ROUTES, **changes)
