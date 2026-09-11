"""The engine: apply nodes until the turn ends, pauses, or runs out of budget.

Three ways out, and that is the whole loop:

* the state is terminal -- a node ended the turn and left a response or a
  failure on it;
* the state is paused -- a mutating call is waiting on a human, and only
  :meth:`WorkflowRuntime.resume_approval` moves it again;
* the step budget is spent -- no node ended the turn, so the engine does, as a
  failure that says exactly that.

The third one is the reason this class is worth writing down. A graph whose
nodes hand control back to a planner can, in principle, keep producing states
forever: a routing bug, a model that keeps choosing the same tool, a tool whose
result never satisfies the question. Without a ceiling that failure mode is a
hang -- the worst kind of bug, because it produces no output to read. With one,
it is a run that ends in ``max_steps_exceeded`` with the whole trace attached,
and a reviewer can see which node kept firing.

Why the ceiling lives here and not on the state
-----------------------------------------------

:attr:`~agentic_erp_assistant.state.agent_state.AgentState.step_count` is a
counter, and this is the limit. Keeping them apart means one graph can be strict
and another generous without either of them changing what a state *is*, and a
reviewer looking for the policy finds it in the engine rather than buried in a
model definition.

Why the pause is a return and not a wait
----------------------------------------

``run`` returns a paused state to its caller instead of blocking on an approver.
A human can take minutes or days, and a runtime holding an open call for that
long would tie a graph run to a process lifetime -- restart the server and the
approval is gone. The paused state is complete and serializable: it carries the
tool, the arguments, the actor and the scopes, so whatever comes back later can
resume it without reconstructing anything.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agentic_erp_assistant.engine.nodes import (
    EVIDENCE_LIMIT,
    GraphNodes,
    NodeTable,
)
from agentic_erp_assistant.engine.ports import (
    AnswerComposerPort,
    DocumentRetrieverPort,
    PlannerPort,
    ToolGatewayPort,
)
from agentic_erp_assistant.engine.transitions import advance
from agentic_erp_assistant.state.agent_state import (
    AgentState,
    ERROR_DETAIL_MAX_CHARS,
)
from agentic_erp_assistant.state.events import TraceEvent

__all__ = [
    "DENIED_REPLY",
    "MAX_STEPS",
    "NotPaused",
    "UnroutableState",
    "WorkflowRuntime",
    "is_paused",
]

logger = logging.getLogger(__name__)


MAX_STEPS = 8
"""How many node executions one turn may spend.

Enough for three actions and the thinking between them -- a search, a tool call
and an answer come to five -- and short enough that a loop is caught in under a
second of model calls rather than after a page of them. It is a ceiling on a
cycle, not a measure of how hard a question is: a turn that needs more than
three actions is a turn that should have asked something back.
"""


DENIED_REPLY = "The call to {tool} was not approved, so nothing was changed."
"""What a denied write tells the user.

Says what did not happen, because "request denied" leaves a reader wondering
whether the write went through anyway.
"""


class NotPaused(RuntimeError):
    """``resume_approval`` was given a state that is not waiting on anybody.

    A caller bug, and raised rather than routed: recording an approval against
    a turn that never asked for one would put a decision in the audit that no
    human was ever shown.
    """


class UnroutableState(RuntimeError):
    """The engine reached a route with no node behind it.

    A bug in the node table, not a condition a turn can recover from. Raised
    loudly for the reason
    :class:`~agentic_erp_assistant.engine.transitions.IllegalTransition` is: a
    silent no-op here looks exactly like a hang.
    """


def is_paused(state: AgentState) -> bool:
    """Whether this turn is waiting on a human and must not be advanced.

    Both halves are required. ``request_approval`` alone is not a pause -- the
    same route carries the state a moment after an answer comes back -- and
    ``pending`` alone cannot occur, because nothing else sets it.
    """
    return state.route == "request_approval" and state.approval == "pending"


@dataclass(frozen=True)
class WorkflowRuntime:
    """Runs one turn, from an unrouted state to a terminal or paused one.

    Built from the four ports and nothing else. Everything replaceable is
    behind one of them, so the whole engine can be exercised with four fakes
    and no network, which is what makes the routing tests worth reading.
    """

    retriever: DocumentRetrieverPort
    tools: ToolGatewayPort
    planner: PlannerPort
    composer: AnswerComposerPort

    max_steps: int = MAX_STEPS
    """The loop guard. See :data:`MAX_STEPS`."""

    tool_retry_budget: int = 1
    """How many times a turn may wait out a rate limit. Passed to the nodes."""

    evidence_limit: int = EVIDENCE_LIMIT
    """How many passages one search pulls. Passed to the nodes."""

    sleep: Callable[[float], object] = time.sleep
    """How to wait between throttled attempts. Injected so tests do not."""

    nodes: NodeTable | None = None
    """A node table to use instead of the standard three.

    For a test that has to force a path the real nodes cannot produce -- a node
    that never terminates, which is the only honest way to prove the loop guard
    fires -- and for a graph variant later. ``None`` builds the standard table
    from the ports above.
    """

    observer: Callable[[AgentState], object] | None = None
    """Called with the state after every node execution, after the engine's own
    bookkeeping, and after the loop guard or an approval changes it. Read-only by
    contract: the state is frozen, and the return value is ignored. Never allowed
    to fail a run -- an observer that raises is logged and dropped for the rest of
    the run, because a screen going away must not end a turn.
    """

    _table: NodeTable = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        table = self.nodes
        if table is None:
            table = GraphNodes(
                retriever=self.retriever,
                tools=self.tools,
                planner=self.planner,
                composer=self.composer,
                evidence_limit=self.evidence_limit,
                tool_retry_budget=self.tool_retry_budget,
                sleep=self.sleep,
            ).table()
        object.__setattr__(self, "_table", table)

    # -- observation ---------------------------------------------------------

    def _observe(self, state: AgentState) -> None:
        if self.observer is None:
            return
        try:
            self.observer(state)
        except Exception:
            logger.warning(
                "observer raised for run %s; dropping it for the rest of this "
                "run",
                state.trace_id,
                exc_info=True,
            )
            object.__setattr__(self, "observer", None)

    # -- the loop ----------------------------------------------------------

    def run(self, state: AgentState) -> AgentState:
        """Apply nodes until the turn ends, pauses, or exhausts its budget.

        Args:
            state: Where to start. Usually unrouted; a partially-run state is
                equally valid, which is what makes a resumed turn the same code
                path as a fresh one.

        Returns:
            A state that is terminal, or paused on an approval. Never anything
            else -- the budget guard makes sure of it.

        Raises:
            UnroutableState: A route in the state has no node in the table.
            IllegalTransition: A node attempted a move the table forbids. A bug
                in that node; see
                :mod:`agentic_erp_assistant.engine.transitions`.
        """
        while True:
            if state.terminal:
                return state
            if is_paused(state):
                # Checked before the budget deliberately: waiting on a person
                # is not work, and charging steps for it would let a slow
                # approver fail a turn that nothing is wrong with.
                return state
            if state.step_count >= self.max_steps:
                return self._out_of_budget(state)

            node = self._table.get(state.route)
            if node is None:
                raise UnroutableState(
                    f"no node handles route {state.route!r}; the table covers "
                    f"{sorted(str(route) for route in self._table)}"
                )

            name = state.route or "start"
            state = state.evolve(
                step_count=state.step_count + 1,
                events=state.events + (TraceEvent(node=name, kind="node_entered"),),
            )
            state = node(state)
            state = state.evolve(
                events=state.events + (TraceEvent(node=name, kind="node_exited"),)
            )
            self._observe(state)

    def _out_of_budget(self, state: AgentState) -> AgentState:
        """End a run that would not end itself, and say so in the trace.

        ``run_failed`` rather than ``failed``: a node assigning a failure and
        the engine giving up on a graph are different events, and only one of
        them means something is wrong with the graph itself.
        """
        detail = (
            f"{state.step_count} node executions without a terminal state "
            f"(max_steps={self.max_steps}); last route was {state.route!r}"
        )
        logger.error("run %s hit the step budget: %s", state.trace_id, detail)
        out_of_budget = advance(
            state,
            "fail",
            mutating=state.tool_mutating if state.route == "call_tool" else None,
            failure="max_steps_exceeded",
            error_detail=detail[:ERROR_DETAIL_MAX_CHARS],
            events=state.events
            + (
                TraceEvent(
                    node="engine", kind="run_failed", detail="max_steps_exceeded"
                ),
            ),
        )
        self._observe(out_of_budget)
        return out_of_budget

    # -- the only way past a pause -----------------------------------------

    def resume_approval(
        self, state: AgentState, approved: bool, *, decided_by: str | None = None
    ) -> AgentState:
        """Record a human's decision on a paused call and carry on, or refuse.

        The only way a paused state moves. Nothing else sets ``approval`` to
        anything but ``pending``, and
        :func:`~agentic_erp_assistant.engine.transitions.assert_transition`
        refuses to enter ``call_tool`` with a mutating tool unless it reads
        ``approved`` -- so a write cannot reach a handler without passing
        through this method, whatever a node tries.

        Args:
            state: The paused state, exactly as ``run`` returned it.
            approved: What the human said.
            decided_by: Who said it, recorded in the trace event's detail.
                ``None`` when the caller has nobody to name.

        Returns:
            An approved call runs and the turn continues to a terminal state; a
            denied one ends immediately in a refusal, with nothing executed.

        Raises:
            NotPaused: ``state`` is not waiting on an approver.
        """
        if not is_paused(state):
            raise NotPaused(
                f"route={state.route!r} approval={state.approval!r}: this turn "
                f"is not waiting on anybody, and recording a decision against "
                f"it would audit an approval nobody was asked for"
            )

        detail = f"{state.tool_name} {'approved' if approved else 'denied'}"
        if decided_by is not None:
            detail = f"{detail} by {decided_by}"
        decided = state.evolve(
            approval="approved" if approved else "denied",
            events=state.events
            + (
                TraceEvent(
                    node="approval",
                    kind="approval_recorded",
                    detail=detail,
                ),
            ),
        )
        self._observe(decided)

        if not approved:
            refused = advance(
                decided,
                "refuse",
                response=DENIED_REPLY.format(tool=state.tool_name),
            )
            self._observe(refused)
            return refused

        return self.run(
            advance(decided, "call_tool", mutating=decided.tool_mutating)
        )
