"""Where the session's recent turns live, and the short-term half of memory
built on top of them.

:mod:`agentic_erp_assistant.memory.service` is the long-term half: a model
proposes, :func:`~agentic_erp_assistant.memory.policy.decide` judges, and only
what survives is kept, forever, as a durable fact. This module keeps something
that earns no such scrutiny -- the actor's own recent words, verbatim, for as
long as they sit inside the window and not one turn longer.

Two differences from :class:`~agentic_erp_assistant.memory.store.MemoryStorePort`
are load-bearing rather than cosmetic:

* **No embeddings, no scope object.** A turn is bounded by ``(session_id,
  actor)`` alone -- there is no cross-project recall to guard against, because
  a session belongs to one project by construction and this store is never
  asked "what does anyone remember about X", only "what did this actor just
  say in this session".
* **Nothing here is judged.** :func:`~agentic_erp_assistant.memory.policy.decide`
  has no opinion on a ``ConversationTurn`` and is never called with one. The
  window's only defences are structural -- see
  :mod:`agentic_erp_assistant.context.history_injection` and the ``history``
  role in :mod:`agentic_erp_assistant.llm.prompts`.

Why ``evicted`` excludes a paused turn
---------------------------------------

A turn waiting on a human has not finished being worth something: the pause
*is* the fact this turn is about, and it belongs in the window, prominently,
for exactly as long as somebody might ask "did that write happen?". Promoting
it into a summary while it is still open would let it fall out of the
window's exact wording and into a paraphrase, at the one moment precision
matters most. See :attr:`~agentic_erp_assistant.state.conversation.ConversationTurn.settled`.

Why eviction asks for ``turn_limit - 1``
------------------------------------------

:meth:`ConversationMemory.evicted` runs *before* the current turn is recorded
(see the ordering in :class:`~agentic_erp_assistant.engine.orchestrator.RunOrchestrator`),
so the store still holds only the turns that came before this one. Once this
turn is appended, the window holds exactly :attr:`ConversationMemory.turn_limit`
turns with this one newest -- which means the store, right now, must keep only
``turn_limit - 1`` of what it already has and call everything older evicted.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from agentic_erp_assistant.context.history_injection import (
    HISTORY_BUDGET_TOKENS,
    HISTORY_TURN_LIMIT,
    select_history,
)
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn

__all__ = [
    "ConversationMemory",
    "ConversationStorePort",
    "InMemoryConversationStore",
    "PROMOTION_BATCH",
]


PROMOTION_BATCH = 12
"""How many evicted turns :mod:`agentic_erp_assistant.memory.promotion` folds
into the session summary in one run.

A ceiling, not an expectation: an ordinary turn evicts at most one turn (the
one pushed out by the newest arrival), and this bound only matters for a
session recovering from a gap -- a store that was down, a batch of turns
nobody promoted yet. Folding an unbounded backlog into one summary in one
run would spend one model call summarizing a whole session's history at once,
which is exactly the cost this module exists to avoid paying per turn.
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@runtime_checkable
class ConversationStorePort(Protocol):
    """What the short-term window assumes about where turns are kept.

    Structural conformance, like every other port in this project: the
    Postgres adapter satisfies this without importing it.
    """

    def append(self, turn: ConversationTurn) -> None:
        """Record this turn as part of its session's window.

        Idempotent on ``trace_id``: a paused turn recorded once while waiting
        and again on resume must leave one row, updated, not two. The row's
        ``started_at`` never moves forward on a later call -- the store keeps
        the earliest one it was given -- so a turn resumed a day later still
        sorts where it actually began.
        """
        ...

    def recent(
        self, session_id: str, *, actor: str, limit: int
    ) -> tuple[ConversationTurn, ...]:
        """The newest ``limit`` turns of this session for this actor, oldest first.

        Bounded by both ``session_id`` and ``actor`` -- a turn recorded for a
        different actor under the same session id is never returned, the same
        boundary :func:`~agentic_erp_assistant.memory.models.bounds` draws
        around an ``intent``.
        """
        ...

    def evicted(
        self, session_id: str, *, actor: str, keep: int, limit: int
    ) -> tuple[ConversationTurn, ...]:
        """Settled turns older than the newest ``keep``, not yet promoted.

        Oldest first, capped at ``limit`` -- see :data:`PROMOTION_BATCH`. A
        turn still :attr:`~agentic_erp_assistant.state.conversation.ConversationTurn.paused`
        is never returned, whatever its age: see the module docstring.
        """
        ...

    def mark_promoted(self, trace_ids: Sequence[str], *, run: str) -> None:
        """Record that these turns were folded into a session summary in ``run``.

        Idempotent, and silent about a ``trace_id`` this store has never seen --
        a turn that has already left every party's memory is not an error to
        report, it is the goal.
        """
        ...


class InMemoryConversationStore:
    """Dictionaries, and the default. Enough for tests and for one process.

    The window's version of
    :class:`~agentic_erp_assistant.memory.store.InMemoryMemoryStore`: it keeps
    the same contracts a real adapter must, including the ones easy to get
    wrong -- ``append``'s ``started_at`` floor, ``evicted``'s exclusion of a
    paused turn -- so a test against the fake is a test against the rule the
    SQL also has to keep.
    """

    def __init__(self) -> None:
        self._turns: dict[str, ConversationTurn] = {}
        self._promoted_in: dict[str, str] = {}

    def append(self, turn: ConversationTurn) -> None:
        existing = self._turns.get(turn.trace_id)
        started_at = (
            min(existing.started_at, turn.started_at) if existing is not None
            else turn.started_at
        )
        current = {name: getattr(turn, name) for name in type(turn).model_fields}
        self._turns[turn.trace_id] = type(turn)(**{**current, "started_at": started_at})

    def recent(
        self, session_id: str, *, actor: str, limit: int
    ) -> tuple[ConversationTurn, ...]:
        matching = [
            turn
            for turn in self._turns.values()
            if turn.session_id == session_id and turn.actor == actor
        ]
        matching.sort(key=lambda turn: (turn.started_at, turn.trace_id), reverse=True)
        newest = matching[:limit]
        newest.reverse()
        return tuple(newest)

    def evicted(
        self, session_id: str, *, actor: str, keep: int, limit: int
    ) -> tuple[ConversationTurn, ...]:
        matching = [
            turn
            for turn in self._turns.values()
            if turn.session_id == session_id
            and turn.actor == actor
            and turn.settled
            and self._promoted_in.get(turn.trace_id) is None
        ]
        matching.sort(key=lambda turn: (turn.started_at, turn.trace_id))
        if keep <= 0:
            beyond = matching
        elif keep >= len(matching):
            beyond = []
        else:
            beyond = matching[:-keep]
        return tuple(beyond[:limit])

    def mark_promoted(self, trace_ids: Sequence[str], *, run: str) -> None:
        for trace_id in trace_ids:
            if trace_id in self._turns:
                self._promoted_in.setdefault(trace_id, run)


@dataclass
class ConversationMemory:
    """The short-term window, bound to a store, the way
    :class:`~agentic_erp_assistant.memory.service.SessionMemory` binds
    :class:`~agentic_erp_assistant.memory.service.MemoryService` to a scope.

    Satisfies the ``SessionHistoryPort`` protocol
    :mod:`agentic_erp_assistant.engine.orchestrator` declares next to its one
    consumer, without importing it -- the same arrangement every port in this
    project uses.
    """

    store: ConversationStorePort
    """Where turns are kept."""

    model: str
    """Whose tokenizer :func:`~agentic_erp_assistant.context.history_injection.select_history`
    measures against. Required, with no default, for the reason
    :attr:`~agentic_erp_assistant.context.builder.ContextBuilder.model` is."""

    budget_tokens: int = HISTORY_BUDGET_TOKENS
    """How many tokens of history one turn's prompt can afford."""

    turn_limit: int = HISTORY_TURN_LIMIT
    """How many of the session's most recent turns the window holds."""

    promotion_batch: int = PROMOTION_BATCH
    """At most how many evicted turns :meth:`evicted` reports in one call."""

    now: Callable[[], datetime] = _utc_now
    """The clock, injected so a test can make time deterministic."""

    def recall(self, state: AgentState) -> tuple[ConversationTurn, ...]:
        """What this turn should be shown of its own session's recent past.

        ``()`` when ``state.session_id`` is ``None`` -- a one-shot run has no
        window to be shown, the same reason
        :meth:`~agentic_erp_assistant.memory.service.SessionMemory.recall`
        returns nothing for one.
        """
        if state.session_id is None:
            return ()
        recent = self.store.recent(
            state.session_id, actor=state.actor, limit=self.turn_limit
        )
        selection = select_history(
            recent, budget_tokens=self.budget_tokens, model=self.model
        )
        return selection.selected

    def record(self, state: AgentState, *, started_at: datetime) -> None:
        """File this turn as part of its session's window.

        A no-op on a turn with no session, for the same reason :meth:`recall`
        is. ``finished_at`` is read from :attr:`now` rather than accepted as an
        argument, the way :class:`~agentic_erp_assistant.memory.service.MemoryService`
        stamps every record's clock itself rather than trusting a caller's.
        """
        if state.session_id is None:
            return
        self.store.append(
            ConversationTurn.from_state(
                state, started_at=started_at, finished_at=self.now()
            )
        )

    def evicted(self, state: AgentState) -> tuple[ConversationTurn, ...]:
        """Turns this session no longer has room for, once this one joins.

        ``()`` when ``state.session_id`` is ``None``, for the same reason
        :meth:`recall` is. See the module docstring for why ``keep`` is
        ``turn_limit - 1`` rather than ``turn_limit``.
        """
        if state.session_id is None:
            return ()
        return self.store.evicted(
            state.session_id,
            actor=state.actor,
            keep=self.turn_limit - 1,
            limit=self.promotion_batch,
        )

    def promoted(self, turns: Sequence[ConversationTurn], *, run: str) -> None:
        """Mark these turns as folded into a session summary in ``run``.

        A no-op on an empty sequence, so a caller need not special-case the
        turn that evicted nothing.
        """
        if not turns:
            return
        self.store.mark_promoted([turn.trace_id for turn in turns], run=run)
