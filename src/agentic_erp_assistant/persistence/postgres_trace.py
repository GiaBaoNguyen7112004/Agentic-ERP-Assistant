"""A :class:`~agentic_erp_assistant.trace.ports.TraceStore` over Postgres.

The trace store is the evidence rule's landing place: every request emits a
trace, and this is where the trace goes so the rule stays true after the
process ends. Three decisions shape the adapter:

* **The run is saved once, at the end, in one transaction.** ``save_run``
  writes the run row and every event in a single
  ``with connection.transaction()`` block, so a run's record is all there or
  not there -- never half. Events are appended in three places across the
  engine, and a store hook called per append would make the store a fifth
  collaborator of a graph built on four. The accepted cost: a hard crash
  mid-run loses that run's trace, which is no worse than the in-memory state
  of the world before this package existed.

* **Re-saving is idempotent.** The event rows upsert with ``ON CONFLICT
  (trace_id, seq) DO NOTHING``, where ``seq`` is the event's position in the
  state's log, so a resumed run's second save re-writes what is already
  there and adds what the continuation produced. The run row itself
  updates, and keeps the earliest ``started_at`` -- a run resumed tomorrow
  still reads as having started when it actually started, not when its
  second segment happened to begin.

* **``record_model_call`` must not raise.** The port says so and the adapter
  obeys: a telemetry INSERT that fails is one ``logger.error``, and the turn
  that spent the tokens goes on. Cost evidence degrades loudly; it does not
  take an answered question down with it.
"""

import logging
from typing import TYPE_CHECKING

import psycopg
from psycopg.types.json import Jsonb

from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.trace.records import RunRecord

if TYPE_CHECKING:  # pragma: no cover -- import guards are the point
    from agentic_erp_assistant.llm.telemetry import ModelCallRecord

__all__ = ["PostgresTraceStore"]

logger = logging.getLogger(__name__)


class PostgresTraceStore:
    """Runs, their events, and their model-call telemetry, in three tables.

    Takes an open connection rather than opening one, so construction never
    has a side effect a test has to undo, and the same class runs against the
    compose database in development and a hosted one in a deployment. See
    :func:`~agentic_erp_assistant.persistence.connection.connect` for where
    connections come from.
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    # -- the run and its events --------------------------------------------

    def save_run(self, record: RunRecord) -> None:
        """Write the run row and its whole event log, atomically.

        The upsert semantics matter more than the write: one trace id is one
        run, however many segments it was saved in, and the second save of a
        resumed run must extend the record rather than fork it.

        Raises:
            psycopg.OperationalError: The database became unreachable. Raised
                here rather than swallowed because ``save_run`` is the one
                store call allowed to fail: the run already happened, and a
                caller who ignores that it was never recorded loses evidence,
                which is the one thing this package exists to keep.
        """
        state = record.state
        with self._connection.transaction():
            self._connection.execute(
                """
                INSERT INTO runs
                    (trace_id, actor, request, outcome, started_at, finished_at,
                     state)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trace_id) DO UPDATE SET
                    actor = EXCLUDED.actor,
                    request = EXCLUDED.request,
                    outcome = EXCLUDED.outcome,
                    started_at = LEAST(runs.started_at, EXCLUDED.started_at),
                    finished_at = EXCLUDED.finished_at,
                    state = EXCLUDED.state
                """,
                (
                    record.trace_id,
                    state.actor,
                    state.request,
                    record.outcome,
                    record.started_at,
                    record.finished_at,
                    Jsonb(state.model_dump(mode="json")),
                ),
            )
            self._connection.cursor().executemany(
                """
                INSERT INTO trace_events (trace_id, seq, node, kind, detail)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (trace_id, seq) DO NOTHING
                """,
                [
                    (
                        record.trace_id,
                        seq,
                        event.node,
                        event.kind,
                        event.detail,
                    )
                    for seq, event in enumerate(state.events)
                ],
            )

    def load_run(self, trace_id: str) -> AgentState | None:
        """The run's final state, revalidated, or ``None`` if it never ran.

        The stored document goes through ``AgentState.model_validate`` rather
        than being trusted: the state carries its own ``state_version``, so a
        run saved by an older shape of the model refuses to load as a loud
        error instead of coming back as a silently wrong object.
        """
        row = self._connection.execute(
            "SELECT state FROM runs WHERE trace_id = %s", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        return AgentState.model_validate(row[0])

    # -- the model calls ----------------------------------------------------

    def record_model_call(self, trace_id: str, record: "ModelCallRecord") -> None:
        """File one call's cost under its run. Must not raise, and does not.

        A telemetry failure is logged and eaten: the port's rule, restated
        where it is kept. The one thing a cost record must never be able to
        do is fail the request that spent the money.
        """
        try:
            self._connection.execute(
                """
                INSERT INTO model_calls
                    (trace_id, model, outcome, estimated_input_tokens,
                     input_tokens, output_tokens, cost_usd, latency_seconds,
                     attempts, occurred_at, detail)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    trace_id,
                    record.model,
                    record.outcome,
                    record.estimated_input_tokens,
                    record.input_tokens,
                    record.output_tokens,
                    record.cost_usd,
                    record.latency_seconds,
                    record.attempts,
                    record.occurred_at,
                    record.detail,
                ),
            )
        except Exception:  # noqa: BLE001 - the port forbids raising, see above
            logger.exception(
                "model call for run %s was not recorded; cost evidence for "
                "this run is incomplete",
                trace_id,
            )