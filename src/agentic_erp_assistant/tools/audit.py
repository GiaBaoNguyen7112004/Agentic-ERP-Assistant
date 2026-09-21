"""Where the record of a gated call goes after the gateway writes it.

A protocol and one in-memory implementation, for the reason
:mod:`agentic_erp_assistant.llm.telemetry` has the same pair: the gateway must
be able to write an audit row without knowing whether the other end is a list
in a test, a JSONL file, or a compliance database. The seam is what lets the
destination change without the execution path changing with it.
"""

from typing import Protocol, runtime_checkable

from agentic_erp_assistant.tools.models import AuditRow

__all__ = ["AuditSink", "InMemoryAuditLog"]


@runtime_checkable
class AuditSink(Protocol):
    """Somewhere an audit row can be written and not lost."""

    def record(self, row: AuditRow) -> None:
        """Persist one row.

        Must not raise for a row it dislikes. This is called on the path of a
        write that a human has already approved, and an audit sink that can
        abort the call it is recording turns "we could not write the record"
        into "the approved work did not happen" -- which is the wrong failure
        and the harder one to explain.
        """
        ...


class InMemoryAuditLog:
    """A list, and the default. Enough for tests and for a single process.

    Not the answer for a real deployment: rows disappear with the process, and
    an audit trail that does not survive a restart is not an audit trail. It is
    the default because a gateway with no sink at all would silently write
    nothing, which is worse -- this at least holds the rows where a test, or a
    developer, can see them.
    """

    def __init__(self) -> None:
        self.rows: list[AuditRow] = []

    def record(self, row: AuditRow) -> None:
        self.rows.append(row)
