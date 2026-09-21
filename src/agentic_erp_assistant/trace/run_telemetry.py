"""The bridge between what a model call cost and the run that spent it.

``llm.telemetry`` declares its sink as a protocol and names no owner beyond
"the trace store that ``trace/`` will own" -- this module is that owner's
smallest piece. The problem it exists for: the LLM gateway sits behind
``plan(state)`` and ``answer(question, evidence)`` and never sees a state, so
it cannot stamp a trace id onto the records it emits. Rather than widen two
port signatures for a field one layer needs, the id is bound here, where the
run is known: the orchestrator builds one of these per run and hands it to the
gateway as the sink.

Satisfies :class:`~agentic_erp_assistant.llm.telemetry.TelemetrySink`
structurally, without importing it -- the same move the tool gateway makes
against ``engine/ports.py``, for the same reason: an import would point this
package at the LLM layer, and the engine's orchestrator imports this
package.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic_erp_assistant.trace.ports import TraceStore

if TYPE_CHECKING:  # pragma: no cover -- import guards are the point
    from agentic_erp_assistant.llm.telemetry import ModelCallRecord

__all__ = ["RunTelemetry"]


@dataclass
class RunTelemetry:
    """One run's telemetry sink: stamps the id, forwards the record."""

    trace_id: str
    """The run every record forwarded through this sink belongs to."""

    store: TraceStore
    """Where the stamped record goes."""

    def record(self, record: "ModelCallRecord") -> None:
        """Forward one record with this run's id.

        Must not raise, per both ports it stands between: telemetry never
        fails a request, and the store underneath has promised the same.
        """
        self.store.record_model_call(self.trace_id, record)