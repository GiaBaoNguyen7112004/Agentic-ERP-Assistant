"""The task in flight: what is being worked on, what is known, what is missing.

An intent is the one thing this package stores that is not a fact. Every other
memory kind is an immutable statement that is only ever superseded; an intent has
a *lifecycle* -- it opens, collects slots turn by turn, and then either closes or
is abandoned for a different task -- and modelling something with a lifecycle as
something without one goes wrong in exactly two ways. Either every slot fill
writes a new "fact", so one task leaves five records and recall has to work out
which is current, or a mutable payload gets bolted onto a record whose whole
value is that it cannot change after it was audited.

So an intent is its own type with its own store row, and it reaches a prompt as a
:class:`~agentic_erp_assistant.state.memory.MemoryRecord` of kind ``intent``
built at recall time -- see :meth:`IntentState.as_memory`. Recall, budgeting and
access need no special case for it, and the record a turn carries stays immutable
even though the thing behind it is not.

Switching cannot leak, because switching cannot be done halfway
---------------------------------------------------------------

The failure this module is written against is a stale slot: the user abandons
"draft the Q4 risk register" for "check the M2 budget", and the assistant carries
``quarter=Q4`` into the new task and answers confidently about the wrong period.

The defence is structural rather than procedural. :meth:`IntentState.switch_to`
returns a *pair* -- the old intent, closed, and a new one with a new id and no
slots at all -- so there is no call that produces the new intent without also
producing the closed old one, and no code path where slots are carried across
because somebody forgot the clearing step. Continuing and switching are different
methods precisely so that "is this the same task?" is a decision made once, in
the open, by whoever calls.

Everything here is bounded
--------------------------

:data:`MAX_SLOTS`, :data:`GOAL_MAX_CHARS` and :data:`SLOT_VALUE_MAX_CHARS` exist
for the reason
:data:`~agentic_erp_assistant.state.memory.STATEMENT_MAX_CHARS` does: an
unbounded slot bag is where a conversation gets stored one key at a time, under a
field named for task state.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from agentic_erp_assistant.state.memory import STATEMENT_MAX_CHARS, MemoryRecord

__all__ = [
    "GOAL_MAX_CHARS",
    "INTENT_KEY_PREFIX",
    "IntentState",
    "IntentStatus",
    "MAX_SLOTS",
    "SLOT_NAME_MAX_CHARS",
    "SLOT_VALUE_MAX_CHARS",
]


IntentStatus = Literal["open", "closed"]
"""Whether the task is still being worked on.

Two members and no ``"abandoned"``: switching away closes the old intent, and a
reader counting closed tasks should not have to know whether each one finished or
was replaced. What replaced it is recoverable -- the new intent's ``opened_at``
sits at the old one's ``closed_at`` -- and inventing a third status to record it
would put the same fact in two places.
"""

GOAL_MAX_CHARS = 200
"""How long a goal may be. A task that needs a paragraph to state is several."""

SLOT_NAME_MAX_CHARS = 60
SLOT_VALUE_MAX_CHARS = 80
"""Caps on one slot. A slot holds an identifier or a short value -- ``M2``,
``atlas``, ``high`` -- and one that can hold a paragraph is a place a
conversation gets stored a key at a time."""

MAX_SLOTS = 12
"""How many slots one intent may carry, confirmed and unresolved together.

A bound rather than a guess at what tasks need: this assistant's tasks are
parameterised by a handful of identifiers, and an intent accumulating dozens is
not a task any more, it is a transcript with an index.
"""

INTENT_KEY_PREFIX = "intent:"
"""How an intent's projection is keyed as a memory.

Prefixed rather than bare so a ``(kind, key)`` collision with anything else is
impossible by inspection, and so an audit row's key column says what it is
looking at without a join.
"""


class IntentState(BaseModel):
    """One task in flight, with its slots and its completion state.

    Frozen, and every transition returns a new object -- the rule
    :meth:`~agentic_erp_assistant.state.agent_state.AgentState.evolve` follows,
    for the same reason: a caller that still holds the previous intent can diff
    the two, and a transition that failed halfway has not left a half-updated
    task behind.

    ``extra="forbid"``: a field arriving from a model's tool call that nothing
    declared is task state no check ever looked at.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_id: str = Field(min_length=1)
    """This task's identity, and the thing a switch changes.

    A new id on switch is what makes the two tasks distinguishable in the audit:
    without it, "the goal changed" and "the slots were re-elicited for the same
    goal" would be the same event.
    """

    goal: str = Field(min_length=1, max_length=GOAL_MAX_CHARS)
    """What the user is trying to accomplish, in their terms."""

    # -- whose task it is --------------------------------------------------

    project_code: str = Field(min_length=1)
    required_scope: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    """The four fields the projection needs in order to be access-checked by the
    same rule everything else is. Carried here rather than supplied at projection
    time, so an intent cannot be rendered into a prompt for somebody it does not
    belong to."""

    # -- what is known and what is missing ---------------------------------

    confirmed_slots: Mapping[str, str] = Field(default_factory=dict)
    """What the user has actually stated, name to value.

    Behind a read-only view for the reason
    :attr:`~agentic_erp_assistant.state.agent_state.AgentState.tool_arguments`
    is: a frozen model holding a plain dict is not frozen, and the caller who
    passed the dict in would keep a handle on the slots a later turn will act on.
    """

    unresolved_slots: tuple[str, ...] = ()
    """What still has to be asked for before the task can complete.

    Named rather than counted, because the point of tracking them is that the
    assistant can ask for the right one. A count would tell it only that it is
    stuck.
    """

    status: IntentStatus = "open"

    opened_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    """The three clocks, all supplied by the caller -- the decision
    :attr:`~agentic_erp_assistant.tools.models.AuditRow.occurred_at` records.
    ``updated_at`` is what recall orders by when a session somehow holds more
    than one."""

    @property
    def live(self) -> bool:
        """Whether this task is still being worked on. What recall filters on."""
        return self.status == "open"

    @property
    def complete(self) -> bool:
        """Whether everything the task needs has been stated.

        Not the same as :attr:`live`: a task can have every slot filled and still
        be open, because filling the last slot is what lets the work happen, not
        the work itself.
        """
        return not self.unresolved_slots

    @field_validator("confirmed_slots")
    @classmethod
    def _freeze_slots(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("confirmed_slots")
    def _unwrap_slots(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def _must_describe_a_task_that_could_exist(self) -> "IntentState":
        for name in (
            "intent_id",
            "goal",
            "project_code",
            "required_scope",
            "actor",
            "session_id",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name}: must not be blank")

        names = list(self.confirmed_slots) + list(self.unresolved_slots)
        if len(names) > MAX_SLOTS:
            raise ValueError(
                f"slots: {len(names)} exceeds the {MAX_SLOTS} an intent may "
                f"carry; a task with more parameters than that is a transcript "
                f"with an index"
            )
        if len(set(names)) != len(names):
            raise ValueError(
                "slots: a name is both confirmed and unresolved, so the task "
                "disagrees with itself about what it still needs"
            )
        for name in names:
            if not name.strip():
                raise ValueError("slots: names must not be blank")
            if len(name) > SLOT_NAME_MAX_CHARS:
                raise ValueError(f"slots: name {name!r} is longer than {SLOT_NAME_MAX_CHARS}")
        for name, value in self.confirmed_slots.items():
            if not value.strip():
                raise ValueError(
                    f"confirmed_slots: {name!r} is confirmed with no value, which "
                    f"is an unresolved slot filed in the wrong half"
                )
            if len(value) > SLOT_VALUE_MAX_CHARS:
                raise ValueError(
                    f"confirmed_slots: {name!r} is longer than "
                    f"{SLOT_VALUE_MAX_CHARS} characters"
                )

        if (self.status == "closed") != (self.closed_at is not None):
            raise ValueError(
                f"closed_at: status is {self.status!r}; a closed task must say "
                f"when it closed and an open one must not claim to have"
            )
        if self.updated_at < self.opened_at:
            raise ValueError("updated_at: cannot precede opened_at")
        if self.closed_at is not None and self.closed_at < self.opened_at:
            raise ValueError("closed_at: a task cannot close before it opened")
        return self

    # -- the three transitions ---------------------------------------------

    def continue_with(
        self, slots: Mapping[str, str], *, at: datetime
    ) -> "IntentState":
        """Fill some slots and stay on the same task.

        Args:
            slots: Name to value, for what the user has now stated. A name in
                :attr:`unresolved_slots` moves across; a name already confirmed
                is overwritten, because the user restating a value is the user
                correcting it.
            at: When. Becomes :attr:`updated_at`.

        Returns:
            A new :class:`IntentState`. ``self`` is unchanged.

        Raises:
            ValueError: The task is closed. Continuing a closed task would
                reopen it without anyone deciding to, and the id would then
                describe two different pieces of work.
        """
        if not self.live:
            raise ValueError(
                f"intent {self.intent_id!r} was closed at "
                f"{self.closed_at.isoformat() if self.closed_at else '?'}; "
                f"continuing it would reopen a task nobody decided to reopen"
            )

        confirmed = dict(self.confirmed_slots) | dict(slots)
        return self._evolve(
            confirmed_slots=confirmed,
            unresolved_slots=tuple(
                name for name in self.unresolved_slots if name not in confirmed
            ),
            updated_at=at,
        )

    def close(self, *, at: datetime) -> "IntentState":
        """Finish the task.

        Idempotent in intent but not in effect: closing an already-closed task
        raises, because the moment a task stopped being worked on is a fact an
        audit may already have read.
        """
        if not self.live:
            raise ValueError(
                f"intent {self.intent_id!r} is already closed; moving the date "
                f"it closed would rewrite when the work stopped"
            )
        return self._evolve(status="closed", updated_at=at, closed_at=at)

    def switch_to(
        self,
        goal: str,
        *,
        intent_id: str,
        at: datetime,
        unresolved_slots: Iterable[str] = (),
    ) -> tuple["IntentState", "IntentState"]:
        """Abandon this task for a different one, and carry nothing across.

        Returns a *pair* on purpose -- ``(closed_previous, fresh)`` -- so there
        is no call that produces the new intent without also producing the closed
        old one. A ``switch_to`` that returned only the new task would leave the
        old one open in the store for whoever remembered to close it, and the
        one that gets forgotten is the one that later gets recalled.

        The new intent starts with **no confirmed slots**. That is the whole
        point: a slot carried across a switch is how an assistant answers
        confidently about the wrong quarter.

        Args:
            goal: What the user is doing instead.
            intent_id: The new task's identity. Supplied rather than generated,
                so this stays a pure function and a test can assert on the id.
            at: When the switch happened. The old task closes and the new one
                opens at the same instant, which is what makes the sequence
                readable afterwards.
            unresolved_slots: What the new task will need.

        Returns:
            ``(previous, fresh)``. ``previous`` is this task, closed.
        """
        return (
            self.close(at=at) if self.live else self,
            type(self)(
                intent_id=intent_id,
                goal=goal,
                project_code=self.project_code,
                required_scope=self.required_scope,
                actor=self.actor,
                session_id=self.session_id,
                confirmed_slots={},
                unresolved_slots=tuple(unresolved_slots),
                status="open",
                opened_at=at,
                updated_at=at,
            ),
        )

    def _evolve(self, **changes: Any) -> "IntentState":
        """Rebuild through the constructor, so every invariant applies to a
        transition exactly as it applies to a new task -- the argument
        :meth:`~agentic_erp_assistant.state.agent_state.AgentState.evolve`
        makes against ``model_copy(update=...)``."""
        current = {name: getattr(self, name) for name in type(self).model_fields}
        return type(self)(**{**current, **changes})

    # -- how it reaches a prompt -------------------------------------------

    def render(self) -> str:
        """The one line a prompt sees, built from the fields rather than stored.

        Derived on every projection rather than kept beside the slots, for the
        reason ``context/candidate.py`` refuses a stored token count: a rendering
        sitting next to the data it describes drifts the first time somebody
        edits one and forgets the other, and the drift is silent -- the model
        would be shown a task whose slots were filled two turns ago.
        """
        parts = [f"Working on: {self.goal.rstrip('.')}."]
        if self.confirmed_slots:
            confirmed = ", ".join(
                f"{name}={value}" for name, value in sorted(self.confirmed_slots.items())
            )
            parts.append(f"Confirmed: {confirmed}.")
        else:
            parts.append("Confirmed: nothing yet.")
        if self.unresolved_slots:
            parts.append(f"Still needed: {', '.join(sorted(self.unresolved_slots))}.")
        else:
            parts.append("Still needed: nothing.")

        line = " ".join(parts)
        if len(line) <= STATEMENT_MAX_CHARS:
            return line
        return line[: STATEMENT_MAX_CHARS - 3] + "..."

    def as_memory(
        self, *, memory_id: str, recorded_in_run: str, recorded_at: datetime
    ) -> MemoryRecord:
        """This task as the immutable record a turn carries.

        The projection, not the storage. It is built at recall time so that what
        the model is shown is the task as it stands now, and it is a
        ``MemoryRecord`` so that selection, the token budget and the access rule
        need no special case for the one kind that has a lifecycle.

        Args:
            memory_id: The projection's id -- from
                :func:`~agentic_erp_assistant.memory.models.memory_id`, so a
                turn shown the same task twice shows the same id.
            recorded_in_run: The run this projection was made for.
            recorded_at: When the projection was made.

        Returns:
            A record of kind ``intent``, always live -- see the raise below.

        Raises:
            ValueError: The task is closed. A closed task is not recalled, and
                projecting one would either show the model work that has
                stopped or produce a record whose retirement predates its own
                creation. Refusing here keeps "what is being worked on" a
                question with one answer.
        """
        if not self.live:
            raise ValueError(
                f"intent {self.intent_id!r} is closed; a finished task is not "
                f"recalled, so there is nothing to project into a prompt"
            )
        return MemoryRecord(
            memory_id=memory_id,
            kind="intent",
            key=f"{INTENT_KEY_PREFIX}{self.intent_id}",
            statement=self.render(),
            project_code=self.project_code,
            required_scope=self.required_scope,
            actor=self.actor,
            session_id=self.session_id,
            recorded_in_run=recorded_in_run,
            recorded_at=recorded_at,
            confidence=1.0,
        )
