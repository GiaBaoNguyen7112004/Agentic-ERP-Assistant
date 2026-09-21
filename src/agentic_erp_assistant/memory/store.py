"""Where memory lives, as a shape -- and the dict that satisfies it for tests.

Two protocols, because a session has two things worth keeping and they have
different natures. **Records** are immutable statements that accumulate and are
retired; **an intent** is one mutable task per session with a lifecycle. Putting
both behind one method set would mean either pretending the intent is immutable
(a new "fact" per slot fill) or pretending records are mutable (an update that
edits history), and both are the wrong lie. They are declared in one file because
one adapter implements both against one database, and a reader asking "what does
memory need from storage?" should get the answer on one screen.

Everything is scoped, and there is no unscoped read
---------------------------------------------------

Every read takes a
:class:`~agentic_erp_assistant.memory.models.MemoryScope`. There is deliberately
no ``all()``, no ``get(memory_id)`` that skips the scope, and no way to ask for
"everything" -- the same decision
:class:`~agentic_erp_assistant.rag.retriever.RetrievalService` makes by having no
method that searches without a context. An unscoped read is the call that
eventually gets made from somewhere that forgot to pass a scope, and the memory
it returns is somebody else's.

Retiring returns what it retired
--------------------------------

:meth:`MemoryStorePort.supersede` hands back the records it retired rather than
returning ``None``. The writer has to record a ``forget`` audit row against each
one, and a store that only reported a count would leave the writer guessing which
ids to name -- or, worse, re-reading them afterwards, when they are already
retired and the read no longer returns them.
"""

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from agentic_erp_assistant.memory.intent import IntentState
from agentic_erp_assistant.memory.models import MemoryKind, MemoryScope, in_bounds
from agentic_erp_assistant.state.memory import MemoryRecord

__all__ = [
    "InMemoryMemoryStore",
    "IntentStorePort",
    "MemoryStorePort",
]


@runtime_checkable
class MemoryStorePort(Protocol):
    """What the memory layer assumes about a place records are kept.

    Structural conformance, so the Postgres adapter satisfies this without
    importing it -- the arrangement every port in this project uses.
    """

    def write(self, record: MemoryRecord) -> None:
        """Store one record.

        Idempotent on ``memory_id``: the id is derived from the scope and the
        content (see
        :func:`~agentic_erp_assistant.memory.models.memory_id`), so writing the
        same fact twice must leave one row rather than raise or fork. A store
        that raised would turn a re-proposal -- which is ordinary -- into a
        failed consolidation.

        May raise. A record that cannot be persisted is a memory that does not
        exist, and the caller has to be able to tell the audit that.
        """
        ...

    def supersede(
        self, memory_ids: Sequence[str], *, at: datetime
    ) -> tuple[MemoryRecord, ...]:
        """Retire these records, and hand back the ones that were retired.

        Nothing is deleted -- see
        :class:`~agentic_erp_assistant.state.memory.MemoryRecord` on why
        forgetting is superseding. An id that is unknown, or already retired, is
        skipped rather than raising: the caller is asking for a state of the
        world, not for an operation to have been performed on each id, and a
        second call after a crash must be able to finish the job.

        Returns:
            The records this call retired, in the order the ids were given.
            Empty when every id was already retired.
        """
        ...

    def live(
        self,
        scope: MemoryScope,
        *,
        kinds: Iterable[MemoryKind] = (),
        limit: int | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """The records this scope owns that nothing has retired, newest first.

        Args:
            scope: Whose memory. Bounds are
                :func:`~agentic_erp_assistant.memory.models.bounds`' -- one
                function, so the store and the recall filter cannot disagree
                about whose memory a record is.
            kinds: Narrow to these. Empty means every kind, which is what the
                policy needs when it checks for conflicts and what a small
                session's recall can afford.
            limit: At most this many. ``None`` means no ceiling, which is
                honest for a store this size and is where a bound goes if that
                ever stops being true.

        Ordering is part of the contract: newest first, because recall keeps the
        head of the list when the budget is tight, and a store returning them
        unordered would make that truncation arbitrary.
        """
        ...

    def by_id(
        self, memory_ids: Sequence[str], scope: MemoryScope
    ) -> tuple[MemoryRecord, ...]:
        """Live, in-scope records for these ids, in the order given.

        The hydration step behind semantic recall: the vector index returns ids
        and scores, and this turns them into records. The scope is not optional
        and the liveness filter is not skippable, which is what makes the index
        an *optimization* rather than a second source of truth -- an id the
        index was slow to retire simply does not come back.

        An unknown, retired or out-of-scope id is omitted, not raised on.
        """
        ...


@runtime_checkable
class IntentStorePort(Protocol):
    """What the memory layer assumes about where the task in flight is kept.

    One open intent per session, and the store is what enforces it. Not a
    convention: two open intents mean recall has to choose, and the one it
    chooses is the one whose slots leak into the wrong task.
    """

    def save_intent(self, intent: IntentState) -> None:
        """Store or replace this task.

        Upserts on ``intent_id``. Saving an *open* intent for a session that
        already has a different open one must close the other or refuse -- a
        store that quietly allowed both would break the invariant the whole
        module rests on.
        """
        ...

    def open_intent(self, scope: MemoryScope) -> IntentState | None:
        """The task this session is working on, or ``None``.

        Only an open one. A closed task is history, and a recall path that could
        receive one would show the model work that has stopped.
        """
        ...


class InMemoryMemoryStore:
    """Dictionaries, and the default. Enough for tests and for one process.

    It exists for the two reasons
    :class:`~agentic_erp_assistant.trace.memory.InMemoryTraceStore` does: a store
    has to exist before a real one does, and a test has to be able to assert on
    one without a database. It is not the answer for a deployment -- restart the
    process and every preference is gone -- but it keeps the same contracts,
    including the ones that matter: ``live`` filters by
    :func:`~agentic_erp_assistant.memory.models.in_bounds` and by liveness, and
    ``save_intent`` refuses a second open task, so a test against the fake is a
    test against the rule the SQL also has to keep.
    """

    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}
        self.intents: dict[str, IntentState] = {}

    # -- records -----------------------------------------------------------

    def write(self, record: MemoryRecord) -> None:
        self.records[record.memory_id] = record

    def supersede(
        self, memory_ids: Sequence[str], *, at: datetime
    ) -> tuple[MemoryRecord, ...]:
        retired: list[MemoryRecord] = []
        for memory_id in memory_ids:
            record = self.records.get(memory_id)
            if record is None or not record.live:
                continue
            self.records[memory_id] = record.retired(at)
            retired.append(self.records[memory_id])
        return tuple(retired)

    def live(
        self,
        scope: MemoryScope,
        *,
        kinds: Iterable[MemoryKind] = (),
        limit: int | None = None,
    ) -> tuple[MemoryRecord, ...]:
        wanted = frozenset(kinds)
        found = [
            record
            for record in self.records.values()
            if record.live
            and in_bounds(record, scope)
            and (not wanted or record.kind in wanted)
        ]
        # Newest first, then by id, so two records written in the same instant
        # do not swap places between runs and make recall irreproducible.
        found.sort(key=lambda record: (record.recorded_at, record.memory_id), reverse=True)
        return tuple(found if limit is None else found[:limit])

    def by_id(
        self, memory_ids: Sequence[str], scope: MemoryScope
    ) -> tuple[MemoryRecord, ...]:
        found = (self.records.get(memory_id) for memory_id in memory_ids)
        return tuple(
            record
            for record in found
            if record is not None and record.live and in_bounds(record, scope)
        )

    # -- the intent --------------------------------------------------------

    def save_intent(self, intent: IntentState) -> None:
        """Upsert, and keep the one-open-per-session invariant the SQL keeps.

        The fake's version of the partial unique index: an open intent arriving
        for a session that already has a different open one is refused, because
        the failure it would otherwise cause -- recall picking one of two tasks
        -- is silent and produces answers about the wrong work.
        """
        if intent.live:
            clash = next(
                (
                    other
                    for other in self.intents.values()
                    if other.live
                    and other.intent_id != intent.intent_id
                    and other.session_id == intent.session_id
                    and other.project_code == intent.project_code
                ),
                None,
            )
            if clash is not None:
                raise ValueError(
                    f"session {intent.session_id!r} already has {clash.intent_id!r} "
                    f"open; close it or switch to the new task, so recall never "
                    f"has to choose between two"
                )
        self.intents[intent.intent_id] = intent

    def open_intent(self, scope: MemoryScope) -> IntentState | None:
        return next(
            (
                intent
                for intent in self.intents.values()
                if intent.live
                and intent.session_id == scope.session_id
                and intent.project_code == scope.project_code
            ),
            None,
        )
