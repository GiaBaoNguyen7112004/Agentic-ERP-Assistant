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

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from agentic_erp_assistant.reasoning.decision import DecisionRoute, FailureMode
from agentic_erp_assistant.state.approval import ApprovalDecision
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.state.reply_contract import ReplyContract, ReplyNeed
from agentic_erp_assistant.state.tool_outcome import ToolOutcome

__all__ = [
    "AgentState",
    "ApprovalDecision",
    "ERROR_DETAIL_MAX_CHARS",
    "STATE_VERSION",
]


STATE_VERSION = 2
"""The shape this module writes and is willing to read.

Carried in the state and checked at construction so a state serialized by an
older build cannot be loaded into a newer one as if the fields still meant the
same thing. Bumping it is a deliberate act that comes with a migration; a
silent load of a mismatched shape is the failure this constant exists to make
loud.

What does *not* require a bump: adding a field that has a default. A stored
state written before the field existed still validates, and the default is the
truthful reading of it -- ``memories=()`` on a run that had no memory layer, and
``session_id=None`` on a run that belonged to no session. The version guards
against fields whose *meaning* changed, which is the case no default can rescue.

v2 added the required :attr:`AgentState.project_code`: a v1 state refuses to
load rather than being read as a turn on no project. There is no honest
default to give it -- every authorization decision downstream of this field
is "this project and this entitlement", and guessing one would let a v1 run
resume with a project it never had.
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

    project_code: str = Field(min_length=1)
    """The project this turn works on, snapshotted with the actor and the scopes.

    Required for the reason ``actor`` is: every authorization decision -- a
    document, a memory, an ERP record -- is "this project and this
    entitlement", and a turn that could exist without a project is a turn
    whose tool calls cannot be bound to one. The composition root reads it
    off the user record, next to the scopes.
    """

    scopes: frozenset[str] = frozenset()
    """What the actor is entitled to do, snapshotted when the turn began.

    Here rather than fetched at the call site because
    :class:`~agentic_erp_assistant.state.tool_request.ToolRequest` requires it
    and a node has nowhere else to get it. Snapshotted rather than re-read per
    call for a reason worth defending: entitlements that could change midway
    would let one turn make two calls under two different permissions, and the
    trace would show neither.

    Defaults to empty, and empty means entitled to nothing. A caller that omits
    it gets every tool refused, which is the direction an omission should fail
    in -- the alternative default, "whatever the tool needs", is the one that
    turns a forgotten field into a granted write.
    """

    trace_id: str = Field(min_length=1)
    """The run this state belongs to.

    Required, with no default: a state that could exist without one would be a
    turn that emits no trace, and the project's evidence requirement would be
    optional in practice.
    """

    session_id: str | None = None
    """The conversation this turn belongs to, or ``None`` when it belongs to none.

    A turn is one run; a session is the sequence of runs a person works through,
    and memory is scoped to it. ``None`` is a real answer and not an oversight:
    a one-shot run -- a script, an evaluation case, a replay -- has no
    conversation to remember anything for, and the orchestrator reads this to
    decide whether recall and consolidation happen at all. Defaulting to ``None``
    means a caller that forgets it gets a turn with no memory, which is the
    direction an omission should fail in; the alternative default, "invent an
    id", would silently start a fresh session per request and fill the store
    with one-turn conversations.

    Optional here and *required* on
    :class:`~agentic_erp_assistant.state.memory.MemoryRecord`, deliberately: a
    turn may have no session, but a memory that exists without one could never
    be scoped, recalled or expired.
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

    memories: tuple[MemoryRecord, ...] = ()
    """What recall selected for this turn, in the order it will be rendered.

    Carried on the state, like :attr:`evidence`, rather than fetched inside the
    prompt layer -- and for the stronger version of the same reason. Memory is
    the one input to a turn that came from a *previous* turn's judgement, so
    "what was this turn shown, and why did it answer that way?" is unanswerable
    unless the selection is part of the record. A recall performed inside a
    prompt builder would leave the trace showing a decision with no visible
    cause.

    Filled by the orchestrator before the graph runs, never by a node: recall
    happens once per turn, outside the step budget, so a re-plan in the middle
    of a reason-act cycle cannot quietly change what the model is remembering.

    A tuple for the same reason :attr:`evidence` is one.
    """

    history: tuple[ConversationTurn, ...] = ()
    """The session's recent turns, as this turn was shown them.

    Filled by the orchestrator before the graph runs, never by a node -- the
    same rule :attr:`memories` follows, and for the same reason: a re-plan in
    the middle of a reason-act cycle must not quietly change what the model is
    reminded of. Holds the clipped, tag-stripped copies the prompt actually
    renders, not the full turns, so the trace shows exactly what the model saw
    rather than a superset of it.

    Empty on a turn with no session, and on the first turn of one. Unlike
    :attr:`memories`, nothing here is judged by a policy: it is this actor's
    own words, kept verbatim within the window rather than accepted or refused.
    """

    contract: ReplyContract | None = None
    """What the planner declared a complete reply to this turn must rest on
    (ADR 0021), or ``None``.

    Filled by the orchestrator before the graph runs, never by a node -- the
    same rule :attr:`memories` and :attr:`history` follow, and for the same
    reason: what the reply is being held to must not change mid-turn because
    a re-plan happened to run. ``None`` means "this turn was never checked" --
    a replay, a hand-built test state, a state paused before this field
    existed, or a declaration call that raised outright (as opposed to one
    that merely answered unreadably, which is
    :data:`~agentic_erp_assistant.state.reply_contract.EMPTY_CONTRACT`, a real
    declaration of nothing needed) -- and every completeness check in
    :mod:`agentic_erp_assistant.reasoning.completeness` treats it exactly
    like a contract with no needs at all: nothing to hold the turn to.
    """

    redirected_needs: frozenset[ReplyNeed] = frozenset()
    """Which of :attr:`contract`'s needs ``engine/nodes.py::think`` has
    already spent its one redirect on (ADR 0021).

    Set by ``think`` the moment it redirects a need, never read back by
    anyone but :func:`~agentic_erp_assistant.reasoning.completeness.next_redirect`
    -- which is the whole point of carrying it: a need is redirected *once*
    per turn, and without this field on the state, a resumed or replayed
    turn would have no way to know a redirect had already been spent and
    could spend it again.
    """

    draft: str | None = Field(default=None, min_length=1)
    """The reply the planner offered, and the completeness check withheld,
    on its way to redirecting a missing ``document_passage`` to a search
    (ADR 0021).

    Set once, by ``think``, at the same transition that routes to
    ``retrieve_project_documents`` for that redirect -- never by any other
    node, and never read by anyone but ``retrieve_and_answer``, which
    delivers it, marked ``incomplete_reply``, if the redirected search finds
    nothing to compose from. A retrieval the *model* chose on its own never
    sets this: only the check's own redirect does, which is exactly how
    ``retrieve_and_answer`` tells "the model wanted to search anyway" apart
    from "the check made it search instead of answering".
    """

    observations: tuple[ToolOutcome, ...] = ()
    """What running things produced, in the order they were produced.

    The observation half of the reason-act cycle: a node acts, the outcome
    lands here, and the next planning step reads it. Kept as the outcomes
    themselves rather than as rendered text because the planner branches on
    ``status`` and the answer step reads ``source_ids``; a turn that stored the
    prose would have to parse its own history back.

    A tuple for the same reason :attr:`evidence` is one, and accumulated rather
    than overwritten so a turn that called two tools can still say what the
    first one said.
    """

    tool_name: str | None = Field(default=None, min_length=1)
    """The tool this turn is calling, once one has been chosen."""

    tool_arguments: Mapping[str, Any] | None = None
    """The arguments the chosen tool will be called with.

    Carried on the state, not held in a node's locals, because of the pause: a
    write stops for a human, the run returns to its caller, and whatever
    resumes it must rebuild the identical call. An approver who read "record a
    high-severity risk on PRJ-1" has to be approving the call that actually
    runs, and arguments that lived only in a stack frame could not survive the
    wait -- or worse, could be rebuilt slightly differently.

    Stored behind a read-only view, like
    :attr:`~agentic_erp_assistant.state.tool_request.ToolRequest.arguments` and
    for the same reason.
    """

    tool_mutating: bool = False
    """Whether the chosen tool changes ERP data.

    Read from the tool's own
    :attr:`~agentic_erp_assistant.llm.tools.ToolSpec.mutating` flag by the
    layer that looked the tool up, and carried here so
    :func:`~agentic_erp_assistant.engine.transitions.assert_transition` can be
    told the truth at every transition that turns on it -- including the one
    after a pause, where the decision that knew the answer is long gone. The
    runtime deliberately cannot look this up itself; see that module for why.
    """

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

        if self.tool_arguments is not None and self.tool_name is None:
            raise ValueError(
                "tool_arguments: present without a tool_name -- arguments that "
                "do not say what they are arguments to cannot be executed or "
                "reviewed"
            )

        if self.tool_mutating and self.tool_name is None:
            raise ValueError(
                "tool_mutating: set without a tool_name -- the flag describes a "
                "tool, so there has to be one"
            )

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

    @field_validator("tool_arguments")
    @classmethod
    def _freeze_arguments(
        cls, value: Mapping[str, Any] | None
    ) -> Mapping[str, Any] | None:
        """Take a copy behind a read-only view.

        A frozen model holding a plain dict is not frozen, it merely looks it.
        Here that matters more than usual: the caller who passed the dict in
        would otherwise keep a handle on the arguments an approver has already
        read, and could edit them during the pause.
        """
        return None if value is None else MappingProxyType(dict(value))

    @field_serializer("tool_arguments")
    def _unwrap_arguments(
        self, value: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        """Hand a plain dict to the serializer; the view is an in-process guard."""
        return None if value is None else dict(value)

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
