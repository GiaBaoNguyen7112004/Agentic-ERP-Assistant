"""The in-memory stores: the zero-config defaults, and the fakes.

They exist for the same two reasons
:class:`~agentic_erp_assistant.tools.audit.InMemoryAuditLog` does: a store has
to exist before a real one does, and a test has to be able to assert on one
without a database. They are not the answer for a real deployment -- restart
the process and the pending approvals are gone, which is the exact gap the
Postgres adapters close -- but they keep the same contracts, including the
one that matters: :meth:`InMemoryPauseStore.claim` settles once, so the fake
proves the same property the SQL ``UPDATE ... WHERE status='pending'``
proves, and a test against the fake is a test against the rule.
"""

from typing import TYPE_CHECKING

from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.trace.ports import PauseAlreadyPending
from agentic_erp_assistant.trace.records import RunRecord

if TYPE_CHECKING:  # pragma: no cover -- import guards are the point
    from agentic_erp_assistant.llm.telemetry import ModelCallRecord

__all__ = ["InMemoryPauseStore", "InMemoryTraceStore"]


def _is_paused(state: AgentState) -> bool:
    """The two fields ``engine.is_paused`` reads.

    Inlined here so this module never imports the engine: the engine's
    orchestrator imports this package, and the dependency arrow only points
    one way. If the pause condition ever grows a third field, the engine's
    version and this one have to move together -- the tests catch that by
    saving an unpaused state and expecting the refusal.
    """
    return state.route == "request_approval" and state.approval == "pending"


class InMemoryTraceStore:
    """A dict per table, for tests and for a single process."""

    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.model_calls: dict[str, list["ModelCallRecord"]] = {}

    def save_run(self, record: RunRecord) -> None:
        self.runs[record.trace_id] = record

    def load_run(self, trace_id: str) -> AgentState | None:
        record = self.runs.get(trace_id)
        return None if record is None else record.state

    def record_model_call(self, trace_id: str, record: "ModelCallRecord") -> None:
        """Must not raise, per the port -- and a list cannot."""
        self.model_calls.setdefault(trace_id, []).append(record)


class InMemoryPauseStore:
    """A pending dict and a settled dict, with the claim as the only move
    between them."""

    def __init__(self) -> None:
        self._pending: dict[str, AgentState] = {}
        self._settled: dict[str, AgentState] = {}

    def save(self, state: AgentState) -> None:
        if not _is_paused(state):
            raise ValueError(
                f"route={state.route!r} approval={state.approval!r}: only a "
                f"paused state may be filed as waiting on a human"
            )
        if state.trace_id in self._pending:
            raise PauseAlreadyPending(
                f"run {state.trace_id!r} already has a pause waiting; a "
                f"second would let an approver answer a call they were never "
                f"shown"
            )
        self._pending[state.trace_id] = state

    def pending(self, trace_id: str) -> AgentState | None:
        return self._pending.get(trace_id)

    def claim(self, trace_id: str, *, approved: bool) -> AgentState | None:
        """Compare-and-set on the pending dict.

        The fake's version of the SQL ``UPDATE ... WHERE status='pending'
        RETURNING``: only one caller finds the state still in the dict, and
        every caller after that gets ``None`` -- whatever ``approved`` they
        brought.
        """
        state = self._pending.pop(trace_id, None)
        if state is not None:
            self._settled[trace_id] = state
        return state