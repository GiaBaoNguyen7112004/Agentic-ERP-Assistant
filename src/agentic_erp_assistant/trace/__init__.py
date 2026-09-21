"""The run store: what a reviewer reads, and the pause queue an approver
answers.

CLAUDE.md's evidence rule says every request emits a trace, and until this
package existed that meant "every request *carried* one" -- the event log rode
on the state, which lived exactly as long as whoever was holding it. This
package is the other half: the place those records go so they survive the
process that produced them, behind two ports
(:class:`~agentic_erp_assistant.trace.ports.TraceStore` and
:class:`~agentic_erp_assistant.trace.ports.PauseStore`) with in-memory
defaults here and Postgres adapters in ``persistence/``.

The import rule for this package: it may depend on ``state`` and on nothing
that names a store backend. The engine's orchestrator imports these ports; a
database is named only in ``persistence/``, for the same reason a provider is
named only in ``llm/adapters/``.
"""

from agentic_erp_assistant.trace.memory import (
    InMemoryPauseStore,
    InMemoryTraceStore,
)
from agentic_erp_assistant.trace.ports import (
    PauseAlreadyPending,
    PauseStore,
    TraceStore,
)
from agentic_erp_assistant.trace.records import RunOutcome, RunRecord
from agentic_erp_assistant.trace.run_telemetry import RunTelemetry

__all__ = [
    "InMemoryPauseStore",
    "InMemoryTraceStore",
    "PauseAlreadyPending",
    "PauseStore",
    "RunOutcome",
    "RunRecord",
    "RunTelemetry",
    "TraceStore",
]