"""The one object a graph node reads and the one it returns.

Everything a turn knows lives here, in typed fields, and nothing a turn knows
lives anywhere else. That is the whole contract: a node takes an
:class:`AgentState` and returns a new one, so replaying a run means replaying a
sequence of these, and reviewing a run means reading them. There is no shared
dict a node can reach into, no attribute set on the side, and no state that
exists only inside a node's local variables.

Why a new object instead of a mutated one
-----------------------------------------

:meth:`AgentState.evolve` returns a new state and leaves the caller's untouched.
The reason is not taste. A mutating graph makes the trace unreliable in exactly
the case it matters most: when a node fails partway, an in-place update has
already half-applied its changes, and the record of "the state the node was
given" no longer exists to compare against. With immutable states, every
transition leaves both sides intact, a retry re-runs a node against the same
input it saw the first time, and a reviewer can diff two adjacent states and
see precisely what one node did.

``evolve`` re-validates rather than copying fields across
---------------------------------------------------------

The obvious implementation is pydantic's ``model_copy(update=...)``, and it is
wrong here: it writes the new values in without running a single validator, so
``model_copy(update={"step_count": -1})`` produces a state no constructor would
ever have allowed. Since the graph reaches every state after the first one
through this method, that would mean the invariants below hold only for initial
states. So ``evolve`` builds a new object through the normal constructor, and
an illegal transition fails at the transition.

What this module does not decide
--------------------------------

Budgets. :attr:`AgentState.step_count` and :attr:`AgentState.retry_count` are
counters, not limits -- the state records how many steps have been taken, and
the runtime decides how many are allowed. Putting the ceiling here would fix
one policy for every graph and hide it from the place a reviewer looks for it.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_erp_assistant.reasoning.decision import DecisionRoute, FailureMode
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.events import TraceEvent

__all__ = [
    "AgentState",
    "ApprovalDecision",
    "ERROR_DETAIL_MAX_CHARS",
    "STATE_VERSION",
]


STATE_VERSION = 1
"""The shape this module writes and is willing to read.

Carried in the state and checked at construction so a state serialized by an
older build cannot be loaded into a newer one as if the fields still meant the
same thing. Bumping it is a deliberate act that comes with a migration; a
silent load of a mismatched shape is the failure this constant exists to make
loud.
"""


ApprovalDecision = Literal[
    "not_required",  # nothing mutating is pending
    "pending",       # a human has been asked and has not answered
    "approved",      # a human said yes, and it is recorded here
    "denied",        # a human said no
]
"""Where a mutating call stands with its approver.

Four members, and ``"not_required"`` is a string rather than ``None`` for the
reason ``FailureMode`` uses ``"none"``: an optional field invites
``if state.approval:`` at the gate, which collapses "nobody needs to approve
this" and "we asked and were refused" into the same falsy value. The one place
that must never be ambiguous is the gate in front of a write.
"""


ERROR_DETAIL_MAX_CHARS = 280
"""How much free text a failure may carry alongside its typed mode."""


class AgentState(BaseModel):
    """One turn's complete state, as of one point in the graph.

    Frozen and ``extra="forbid"``. Frozen because the audit trail is built on
    the assumption that a state handed to a node is the state that node saw;
    ``extra="forbid"`` because a field smuggled in at one call site is state
    crossing a node boundary without a type, which is the thing this class
    exists to prevent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # -- what the turn is, and who it is for ------------------------------

    request: str = Field(min_length=1)
    """The user's words, verbatim.

    Never rewritten in place by a node: a node that reformulates the question
    records the reformulation elsewhere, so the trace keeps what was asked.
    """

    actor: str = Field(min_length=1)
    """Who is asking.

    Carried from the first state because approval routing has to know who a
    write would be performed on behalf of, and a decision recorded without an
    actor cannot be audited afterwards.
    """

    trace_id: str = Field(min_length=1)
    """The run this state belongs to.

    Required, with no default: a state that could exist without one would be a
    turn that emits no trace, and the project's evidence requirement would be
    optional in practice.
    """

    # -- what the runtime decided and gathered ----------------------------

    route: DecisionRoute | None = None
    """What the graph decided to do next, or ``None`` before it has decided.

    Reuses the decision layer's closed set rather than declaring a second one.
    ``None`` means "not routed yet" and nothing else -- replying from what is
    already known is the ``"answer"`` member, precisely so those two stay
    distinguishable.
    """

    evidence: tuple[EvidenceSnippet, ...] = ()
    """The passages retrieval supplied, each carrying the address it is cited by.

    The same type the prompt layer renders, not a second one that would have to
    be converted -- a conversion is where a citation loses its locator.

    A tuple, not a list: a frozen model holding a mutable sequence is not
    frozen, and evidence that can be appended to after the fact is evidence a
    reviewer cannot trust.
    """

    tool_name: str | None = Field(default=None, min_length=1)
    """The tool this turn is calling, once one has been chosen."""

    approval: ApprovalDecision = "not_required"
    """Where a pending call stands with its approver. See
    :data:`ApprovalDecision`."""

    # -- how the turn ended -----------------------------------------------

    response: str | None = None
    """The reply, once there is one. ``None`` while the turn is still running."""

    failure: FailureMode = "none"
    """Why the turn could not produce a grounded answer, or ``"none"``.

    The same Literal
    :func:`~agentic_erp_assistant.reasoning.decision.classify_failure` returns,
    so the classifier's output drops straight in and the runtime branches on
    one vocabulary instead of translating between two.
    """

    error_detail: str | None = Field(default=None, max_length=ERROR_DETAIL_MAX_CHARS)
    """A one-line human-readable note about the failure.

    Capped, and never branched on. :attr:`failure` is the fact the runtime acts
    on; this is the sentence a person reads next to it. Keeping them apart is
    what stops a stack trace from becoming a routing input.
    """

    terminal: bool = False
    """Whether the graph is finished with this turn.

    Set by the node that ends it, never inferred by the engine from the shape
    of the other fields.
    """

    # -- the record --------------------------------------------------------

    step_count: int = Field(default=0, ge=0)
    """How many node executions this turn has spent. A counter, not a ceiling."""

    retry_count: int = Field(default=0, ge=0)
    """How many retries this turn has spent. Also a counter, not a ceiling."""

    events: tuple[TraceEvent, ...] = ()
    """What happened, in order.

    Appended to as the graph moves; see
    :mod:`agentic_erp_assistant.state.events` for why the order carries the
    timing and each entry does not.
    """

    state_version: int = STATE_VERSION
    """The shape this state was written in. See :data:`STATE_VERSION`."""

    # -- invariants --------------------------------------------------------

    @model_validator(mode="after")
    def _must_describe_a_turn_that_could_exist(self) -> "AgentState":
        if self.state_version != STATE_VERSION:
            raise ValueError(
                f"state_version: this build reads version {STATE_VERSION}, got "
                f"{self.state_version}; a mismatched state needs a migration, "
                f"not a silent load"
            )

        if self.tool_name is not None and not self.tool_name.strip():
            raise ValueError("tool_name: must not be blank")

        if self.approval != "not_required" and self.tool_name is None:
            raise ValueError(
                f"approval: {self.approval!r} without a tool_name -- an approval "
                f"that does not name what is being approved cannot be audited"
            )

        if self.terminal and self.response is None and self.failure == "none":
            raise ValueError(
                "terminal: a finished turn must carry either a response or a "
                "failure; ending with neither leaves the caller nothing and the "
                "trace no reason"
            )

        return self

    # -- the only way to move ---------------------------------------------

    def evolve(self, **changes: Any) -> "AgentState":
        """Return a new state with ``changes`` applied, leaving this one alone.

        The single transition the graph uses. It re-runs construction, so every
        field constraint and every invariant above applies to a transition
        exactly as it applies to an initial state -- see the module docstring
        for why ``model_copy(update=...)`` is not used.

        An unknown keyword is rejected by ``extra="forbid"`` rather than by a
        check written here, deliberately: one mechanism guards the constructor
        and this method, so there is no second rule to keep in step.

        Args:
            **changes: Field names and their new values. Fields left out keep
                the value they have.

        Returns:
            A new :class:`AgentState`. ``self`` is unchanged.

        Raises:
            ValidationError: A value, or the combination the change produces,
                is not a state the runtime allows -- or a name is not a field.
        """
        current = {name: getattr(self, name) for name in type(self).model_fields}
        return type(self)(**{**current, **changes})
