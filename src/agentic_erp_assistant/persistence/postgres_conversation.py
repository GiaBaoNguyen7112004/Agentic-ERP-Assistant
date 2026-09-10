"""A :class:`~agentic_erp_assistant.memory.conversation.ConversationStorePort`
over Postgres.

The one table this adapter owns, ``session_turns``, holds one row per turn --
finished or still paused -- and is what makes the short-term window survive a
restart the way :mod:`~agentic_erp_assistant.persistence.postgres_pause`
already makes a pending approval survive one. ``append`` never loses the
turn's true start: ``started_at = LEAST(session_turns.started_at,
EXCLUDED.started_at)`` is the same move
:meth:`~agentic_erp_assistant.persistence.postgres_trace.PostgresTraceStore.save_run`
makes for a run resumed across a restart, so a paused turn recorded once while
waiting and again on resume still sorts where it actually began.

``promoted_in_run`` is the watermark
:mod:`agentic_erp_assistant.memory.promotion` reads through
:meth:`PostgresConversationStore.evicted`: a turn already folded into a
session summary is excluded from a later eviction, so a memory outage that
delays promotion does not fold the same turn twice.
"""

from collections.abc import Sequence

import psycopg
from psycopg.rows import dict_row

from agentic_erp_assistant.state.conversation import ConversationTurn

__all__ = ["PostgresConversationStore"]


_COLUMNS = (
    "trace_id, session_id, actor, request, response, route, failure, "
    "tool_name, approval, started_at, finished_at"
)


def _turn(row: dict) -> ConversationTurn:
    """One row as a validated turn.

    Through the model rather than trusted, for the reason
    :func:`~agentic_erp_assistant.persistence.postgres_memory._record` is: a
    row written by an older shape of the model fails loudly on load instead
    of reaching a prompt as a silently wrong object.
    """
    return ConversationTurn(
        trace_id=row["trace_id"],
        session_id=row["session_id"],
        actor=row["actor"],
        request=row["request"],
        response=row["response"],
        route=row["route"],
        failure=row["failure"],
        tool_name=row["tool_name"],
        approval=row["approval"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


class PostgresConversationStore:
    """One session's turns, in one table, with the actor bound into every query.

    Takes an open connection like every adapter here, so construction has no
    side effect and the composition point decides where the database is.
    Satisfies :class:`~agentic_erp_assistant.memory.conversation.ConversationStorePort`
    structurally, importing neither it nor anything above this layer.
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def append(self, turn: ConversationTurn) -> None:
        """Record this turn, upserting on ``trace_id``.

        Every field but ``started_at`` takes the newest value given --
        a paused turn resumed with a reply is meant to end up with that
        reply. ``started_at`` takes the earliest of the two, so a turn
        recorded once while waiting and again on resume still sorts where it
        actually began.
        """
        self._connection.execute(
            f"""
            INSERT INTO session_turns ({_COLUMNS})
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trace_id) DO UPDATE SET
                response = EXCLUDED.response,
                route = EXCLUDED.route,
                failure = EXCLUDED.failure,
                tool_name = EXCLUDED.tool_name,
                approval = EXCLUDED.approval,
                finished_at = EXCLUDED.finished_at,
                started_at = LEAST(session_turns.started_at, EXCLUDED.started_at)
            """,
            (
                turn.trace_id,
                turn.session_id,
                turn.actor,
                turn.request,
                turn.response,
                turn.route,
                turn.failure,
                turn.tool_name,
                turn.approval,
                turn.started_at,
                turn.finished_at,
            ),
        )

    def recent(
        self, session_id: str, *, actor: str, limit: int
    ) -> tuple[ConversationTurn, ...]:
        """The newest ``limit`` turns of this session for this actor, oldest first."""
        with self._connection.cursor(row_factory=dict_row) as cursor:
            rows = cursor.execute(
                f"""
                SELECT {_COLUMNS} FROM session_turns
                WHERE session_id = %s AND actor = %s
                ORDER BY started_at DESC, trace_id DESC
                LIMIT %s
                """,
                (session_id, actor, limit),
            ).fetchall()
        return tuple(_turn(row) for row in reversed(rows))

    def evicted(
        self, session_id: str, *, actor: str, keep: int, limit: int
    ) -> tuple[ConversationTurn, ...]:
        """Settled turns older than the newest ``keep``, not yet promoted, oldest first.

        The newest ``keep`` turns are found first (the same window ``recent``
        would return) and excluded by id, rather than by an offset: an offset
        would assume the two queries agree on ordering ties the same way,
        which naming the ids outright does not require.
        """
        with self._connection.cursor(row_factory=dict_row) as cursor:
            rows = cursor.execute(
                f"""
                SELECT {_COLUMNS} FROM session_turns
                WHERE session_id = %s AND actor = %s
                    AND promoted_in_run IS NULL
                    AND NOT (route = 'request_approval' AND approval = 'pending')
                    AND trace_id NOT IN (
                        SELECT trace_id FROM session_turns
                        WHERE session_id = %s AND actor = %s
                        ORDER BY started_at DESC, trace_id DESC
                        LIMIT %s
                    )
                ORDER BY started_at ASC, trace_id ASC
                LIMIT %s
                """,
                (session_id, actor, session_id, actor, max(keep, 0), limit),
            ).fetchall()
        return tuple(_turn(row) for row in rows)

    def mark_promoted(self, trace_ids: Sequence[str], *, run: str) -> None:
        """Record that these turns were folded into a session summary in ``run``.

        ``trace_id = ANY(%s)`` with an empty list matches nothing, so a caller
        need not special-case an empty sequence.
        """
        if not trace_ids:
            return
        self._connection.execute(
            """
            UPDATE session_turns SET promoted_in_run = %s
            WHERE trace_id = ANY(%s) AND promoted_in_run IS NULL
            """,
            (run, list(trace_ids)),
        )
