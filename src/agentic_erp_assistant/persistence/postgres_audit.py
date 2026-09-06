"""An :class:`~agentic_erp_assistant.tools.audit.AuditSink` over Postgres.

The one store that inserts immediately, per call, rather than at run end.
An audit row is the record that a write a human approved actually happened,
and it must not depend on the run finishing to exist -- a turn that crashes
a step after an approved write has still executed that write, and the audit
trail has to say so even though nothing else about the run was recorded.

Which makes the failure rule the sharpest one in this package. The sink sits
on the path of a write that a human has already approved: raising here turns
"could not write the record" into "the approved work did not happen", which
is the wrong failure and the harder one to explain. So an INSERT that fails
is one ``logger.error`` -- loud, with everything needed to reconstruct the
row by hand -- and the write goes on. A missing audit row is recoverable
evidence loss; a missing approved write is not.
"""

import logging

import psycopg

from agentic_erp_assistant.tools.models import AuditRow

__all__ = ["PostgresAuditLog"]

logger = logging.getLogger(__name__)


class PostgresAuditLog:
    """Audit rows, one INSERT per call, never raised past.

    Takes an open connection, like every adapter here, so the class is a
    cursor's worth of SQL and nothing else -- no construction side effects,
    nothing to close.
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def record(self, row: AuditRow) -> None:
        """Persist one row. Must not raise, and does not.

        See the module docstring for why. The row's own fields are the
        error's context: the message carries the trace id and the tool, so a
        reconstructed row from logs names the run it belongs to.
        """
        try:
            self._connection.execute(
                """
                INSERT INTO audit_rows
                    (occurred_at, trace_id, actor, tool_name, arguments_summary,
                     approval, status, source_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row.occurred_at,
                    row.trace_id,
                    row.actor,
                    row.tool_name,
                    row.arguments_summary,
                    row.approval,
                    row.status,
                    list(row.source_ids),
                ),
            )
        except Exception:  # noqa: BLE001 - the port forbids raising; see above
            logger.error(
                "audit row was not recorded: run %s, %s for %s -- the "
                "approved write happened and its record did not; reconstruct "
                "from this log line",
                row.trace_id,
                row.tool_name,
                row.actor,
            )