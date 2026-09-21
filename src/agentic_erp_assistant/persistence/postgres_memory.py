"""The durable ends of the three memory ports, over Postgres.

Three adapters in one file because they are one story: memory is the record,
the intent is the task in flight, and the audit is why either of them looks the
way it does. Splitting them across three modules would put a reader of the SQL
one file away from the answer to "and where is the decision that wrote this
row?".

Postgres is the record, and Qdrant is an index over it
------------------------------------------------------

This is the decision worth arguing with, because it is deliberately the reverse
of ADR 0010, where the vector store *is* the chunk store. A chunk has no
identity that outlives an ingest, no supersession and no uniqueness constraint,
so a copy of it in Qdrant cannot be wrong. A memory has all three. It is retired
the moment something better replaces it, exactly one intent may be open per
session, and every row here has an audit row that has to agree with it. Those
are relational obligations, and a store that cannot express them would enforce
them in application code that two processes could race past.

So the tables here are authoritative and the vector index is a way of finding
them. See :mod:`agentic_erp_assistant.memory.qdrant_index`, and ADR 0013 for
the comparison in full.

The failure rules are not the same for the three
------------------------------------------------

``write`` and ``supersede`` may raise: a memory that was not stored is a memory
that does not exist, and the consolidator has to be able to tell the audit that.
``record`` must not, for the reason
:class:`~agentic_erp_assistant.tools.audit.AuditSink` gives and one more: memory
consolidation runs after the turn has already answered the user, so an exception
would turn "we could not write down that we declined to remember something" into
a failed request.
"""

import logging
from collections.abc import Iterable, Sequence
from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agentic_erp_assistant.memory.audit import MemoryAuditRow
from agentic_erp_assistant.memory.intent import IntentState
from agentic_erp_assistant.memory.models import MemoryKind, MemoryScope, bounds
from agentic_erp_assistant.state.memory import MemoryRecord

__all__ = ["PostgresMemoryAudit", "PostgresMemoryStore"]

logger = logging.getLogger(__name__)


_MEMORY_COLUMNS = (
    "memory_id, kind, key, statement, project_code, required_scope, actor, "
    "session_id, recorded_in_run, recorded_at, confidence, supersedes, "
    "superseded_at, links"
)

_INTENT_COLUMNS = (
    "intent_id, goal, project_code, required_scope, actor, session_id, "
    "confirmed_slots, unresolved_slots, status, opened_at, updated_at, closed_at"
)


def _scope_clause(kind: MemoryKind, scope: MemoryScope) -> tuple[str, list[object]]:
    """The WHERE fragment for "records of this kind that belong to this scope".

    Built from :func:`~agentic_erp_assistant.memory.models.bounds` rather than
    written out, so the SQL and the in-memory store cannot disagree about whose
    memory a record is. A preference that leaked between actors because one of
    the two filters was edited alone is the exact bug that would follow from
    spelling this twice -- and it would be invisible until somebody noticed the
    assistant answering in the wrong language.
    """
    matched = bounds(kind, scope)
    clause = " AND ".join(f"{column} = %s" for column in matched)
    return clause, list(matched.values())


class PostgresMemoryStore:
    """Memories and intents, in two tables, with the scope in every query.

    Takes an open connection like every adapter here, so construction has no
    side effect and the composition point decides where the database is.

    Satisfies :class:`~agentic_erp_assistant.memory.store.MemoryStorePort` and
    :class:`~agentic_erp_assistant.memory.store.IntentStorePort` structurally,
    importing neither.
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    # -- records -----------------------------------------------------------

    def write(self, record: MemoryRecord) -> None:
        """Store one record, or replace the one that already has its id.

        Idempotent on the primary key, because ``memory_id`` is derived from the
        scope and the content: proposing the same fact twice is ordinary, and a
        store that raised on the second would turn that into a failed
        consolidation.

        Raises:
            psycopg.Error: The write did not happen. Raised rather than
                swallowed -- a memory that was not stored is one that does not
                exist, and the caller has to be able to tell the audit so.
        """
        self._connection.execute(
            f"""
            INSERT INTO memories ({_MEMORY_COLUMNS})
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (memory_id) DO UPDATE SET
                statement = EXCLUDED.statement,
                confidence = EXCLUDED.confidence,
                supersedes = EXCLUDED.supersedes,
                superseded_at = EXCLUDED.superseded_at,
                links = EXCLUDED.links
            """,
            (
                record.memory_id,
                record.kind,
                record.key,
                record.statement,
                record.project_code,
                record.required_scope,
                record.actor,
                record.session_id,
                record.recorded_in_run,
                record.recorded_at,
                record.confidence,
                list(record.supersedes),
                record.superseded_at,
                list(record.links),
            ),
        )

    def supersede(
        self, memory_ids: Sequence[str], *, at: datetime
    ) -> tuple[MemoryRecord, ...]:
        """Retire these records, and return the ones this call actually retired.

        ``WHERE superseded_at IS NULL`` is what makes it safe to call twice: a
        record already retired is skipped rather than having its retirement date
        moved, so a consolidator resuming after a crash finishes the job instead
        of rewriting when the assistant stopped believing something.

        Nothing is deleted. ``RETURNING`` hands back the rows so the caller can
        write a ``forget`` audit row against each -- a store returning a count
        would leave it guessing which ids to name.
        """
        if not memory_ids:
            return ()

        with self._connection.cursor(row_factory=dict_row) as cursor:
            rows = cursor.execute(
                f"""
                UPDATE memories
                SET superseded_at = %s
                WHERE memory_id = ANY(%s) AND superseded_at IS NULL
                RETURNING {_MEMORY_COLUMNS}
                """,
                (at, list(memory_ids)),
            ).fetchall()

        order = {memory_id: index for index, memory_id in enumerate(memory_ids)}
        retired = [_record(row) for row in rows]
        retired.sort(key=lambda record: order[record.memory_id])
        return tuple(retired)

    def live(
        self,
        scope: MemoryScope,
        *,
        kinds: Iterable[MemoryKind] = (),
        limit: int | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Live in-scope records, newest first.

        One query per kind rather than one query with an OR of scope clauses,
        because the bounds differ *by kind* -- a preference is bounded by actor
        and an intent by session -- and a single WHERE would have to encode that
        difference a second time. The cost is at most five small indexed lookups
        for a store this size; the alternative is the SQL and
        :func:`~agentic_erp_assistant.memory.models.bounds` drifting apart.
        """
        wanted: tuple[MemoryKind, ...] = tuple(kinds) or _ALL_KINDS
        found: list[MemoryRecord] = []
        with self._connection.cursor(row_factory=dict_row) as cursor:
            for kind in wanted:
                clause, values = _scope_clause(kind, scope)
                found.extend(
                    _record(row)
                    for row in cursor.execute(
                        f"""
                        SELECT {_MEMORY_COLUMNS} FROM memories
                        WHERE kind = %s AND superseded_at IS NULL AND {clause}
                        """,
                        (kind, *values),
                    ).fetchall()
                )

        found.sort(key=lambda record: (record.recorded_at, record.memory_id), reverse=True)
        return tuple(found if limit is None else found[:limit])

    def by_id(
        self, memory_ids: Sequence[str], scope: MemoryScope
    ) -> tuple[MemoryRecord, ...]:
        """Live, in-scope records for these ids, in the order given.

        The hydration step behind semantic recall. Both filters are re-applied
        here rather than trusted from the index, which is what makes the index an
        optimization: an id it was slow to retire, or one it should never have
        returned, simply does not come back.
        """
        if not memory_ids:
            return ()

        with self._connection.cursor(row_factory=dict_row) as cursor:
            rows = cursor.execute(
                f"""
                SELECT {_MEMORY_COLUMNS} FROM memories
                WHERE memory_id = ANY(%s) AND superseded_at IS NULL
                """,
                (list(memory_ids),),
            ).fetchall()

        # The scope filter is applied in Python rather than in SQL because it
        # depends on the row's own kind -- see `live` for the same reason.
        found = {
            record.memory_id: record
            for record in (_record(row) for row in rows)
            if _in_scope(record, scope)
        }
        return tuple(
            found[memory_id] for memory_id in memory_ids if memory_id in found
        )

    # -- the intent --------------------------------------------------------

    def save_intent(self, intent: IntentState) -> None:
        """Store or replace one task.

        Raises:
            ValueError: The session already has a different task open. The
                partial unique index is what says so, so two processes cannot
                race past the rule from opposite sides -- the same shape
                :class:`~agentic_erp_assistant.persistence.postgres_pause.PostgresPauseStore`
                uses for the pending-approval rule. Typed as ``ValueError``
                because that is what the in-memory store raises, and a caller
                should not have to catch two things for one mistake.
        """
        try:
            self._connection.execute(
                f"""
                INSERT INTO intents ({_INTENT_COLUMNS})
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (intent_id) DO UPDATE SET
                    goal = EXCLUDED.goal,
                    confirmed_slots = EXCLUDED.confirmed_slots,
                    unresolved_slots = EXCLUDED.unresolved_slots,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at,
                    closed_at = EXCLUDED.closed_at
                """,
                (
                    intent.intent_id,
                    intent.goal,
                    intent.project_code,
                    intent.required_scope,
                    intent.actor,
                    intent.session_id,
                    Jsonb(dict(intent.confirmed_slots)),
                    list(intent.unresolved_slots),
                    intent.status,
                    intent.opened_at,
                    intent.updated_at,
                    intent.closed_at,
                ),
            )
        except psycopg.errors.UniqueViolation as error:
            raise ValueError(
                f"session {intent.session_id!r} already has a task open; close "
                f"it or switch to the new one, so recall never has to choose "
                f"between two"
            ) from error

    def open_intent(self, scope: MemoryScope) -> IntentState | None:
        """The task this session is working on, or ``None``."""
        with self._connection.cursor(row_factory=dict_row) as cursor:
            row = cursor.execute(
                f"""
                SELECT {_INTENT_COLUMNS} FROM intents
                WHERE project_code = %s AND session_id = %s AND status = 'open'
                """,
                (scope.project_code, scope.session_id),
            ).fetchone()
        return None if row is None else _intent(row)


class PostgresMemoryAudit:
    """Memory decisions, one INSERT per decision, never raised past.

    Every verdict lands here, not only the ones that stored something. A run
    whose model proposed five memories and stored one leaves five rows, and the
    four rejections each name the rule -- which is the half of the record that
    shows the policy working rather than merely existing.
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def record(self, row: MemoryAuditRow) -> None:
        """Persist one row. Must not raise, and does not.

        The log line carries the identifiers the row would have, so a lost row
        is reconstructable by hand -- the same contract, and the same
        consolation, as
        :class:`~agentic_erp_assistant.persistence.postgres_audit.PostgresAuditLog`.
        """
        try:
            self._connection.execute(
                """
                INSERT INTO memory_audit
                    (occurred_at, trace_id, session_id, project_code, actor,
                     memory_id, kind, decision, rejection, reason,
                     statement_summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row.occurred_at,
                    row.trace_id,
                    row.session_id,
                    row.project_code,
                    row.actor,
                    row.memory_id,
                    row.kind,
                    row.decision,
                    row.rejection,
                    row.reason,
                    row.statement_summary,
                ),
            )
        except Exception:  # noqa: BLE001 - the port forbids raising; see above
            logger.error(
                "memory decision was not recorded: run %s, session %s, %s on "
                "%s (%s) -- reconstruct from this log line",
                row.trace_id,
                row.session_id,
                row.decision,
                row.memory_id,
                row.rejection or row.kind,
            )


_ALL_KINDS: tuple[MemoryKind, ...] = (
    "preference",
    "decision",
    "fact",
    "intent",
    "session_summary",
)
"""Every kind, for the no-narrowing case in :meth:`PostgresMemoryStore.live`.

Spelled out rather than derived with ``get_args`` so the per-kind queries run in
a stated order and a listing is reproducible between runs -- the same reason
:meth:`~agentic_erp_assistant.rag.vector_index.QdrantVectorIndex.iter_chunks`
sorts rather than trusting the store's paging order.
"""


def _in_scope(record: MemoryRecord, scope: MemoryScope) -> bool:
    """The scope test, using the record's own kind to pick the bounds."""
    return all(
        getattr(record, column) == value
        for column, value in bounds(record.kind, scope).items()
    )


def _record(row: dict) -> MemoryRecord:
    """One row as a validated record.

    Through the model rather than trusted, for the reason
    :meth:`~agentic_erp_assistant.persistence.postgres_trace.PostgresTraceStore.load_run`
    revalidates a state: a row written by an older shape of the model fails
    loudly on load instead of coming back as a silently wrong object.
    """
    return MemoryRecord(
        memory_id=row["memory_id"],
        kind=row["kind"],
        key=row["key"],
        statement=row["statement"],
        project_code=row["project_code"],
        required_scope=row["required_scope"],
        actor=row["actor"],
        session_id=row["session_id"],
        recorded_in_run=row["recorded_in_run"],
        recorded_at=row["recorded_at"],
        confidence=row["confidence"],
        supersedes=tuple(row["supersedes"] or ()),
        superseded_at=row["superseded_at"],
        links=tuple(row["links"] or ()),
    )


def _intent(row: dict) -> IntentState:
    """One row as a validated task."""
    return IntentState(
        intent_id=row["intent_id"],
        goal=row["goal"],
        project_code=row["project_code"],
        required_scope=row["required_scope"],
        actor=row["actor"],
        session_id=row["session_id"],
        confirmed_slots=row["confirmed_slots"] or {},
        unresolved_slots=tuple(row["unresolved_slots"] or ()),
        status=row["status"],
        opened_at=row["opened_at"],
        updated_at=row["updated_at"],
        closed_at=row["closed_at"],
    )
