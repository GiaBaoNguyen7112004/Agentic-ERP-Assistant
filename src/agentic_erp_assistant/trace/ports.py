"""Where runs, pauses and model calls go so they survive the process.

Two protocols, because there are two lifetimes to protect. A **run** is
evidence: once the engine gives a state back, the store holds the copy a
reviewer reads, and nothing about a restart may lose it. A **pause** is a
promise: the engine returned a turn that is waiting on a human, and the store
is what lets the human answer hours later, in another process, without the
graph reconstructing anything.

Declared here and not in the engine's ports module because the engine never
needs them: the graph runs states, and the *orchestrator* -- the composition
point that will one day be the web layer's first call -- is the one that
persists. Ports stay next to the layer that would be swapped if the backing
store changed, which is what ``runtime_checkable`` is for: the Postgres
adapter satisfies these structurally and never imports this file.

The ``ModelCallRecord`` import is under ``TYPE_CHECKING`` for the same reason
``engine/ports.py`` keeps its LLM types there: the engine's orchestrator
imports this module, and a runtime import of the LLM package would drag a
tokenizer and a pricing table into every node import -- the exact coupling the
ports exist to prevent.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentic_erp_assistant.state.agent_state import AgentState

if TYPE_CHECKING:  # pragma: no cover -- import guards are the point
    from agentic_erp_assistant.llm.telemetry import ModelCallRecord

from agentic_erp_assistant.trace.records import RunRecord

__all__ = ["PauseAlreadyPending", "PauseStore", "TraceStore"]


class PauseAlreadyPending(RuntimeError):
    """A second pause was saved for a run that already has one waiting.

    Impossible in the orchestrator's flow -- resuming settles the old pause
    before the run can pause again -- so this being raised means a caller is
    writing pauses out of order. Raised rather than overwritten because the
    pending row is the thing an approver reads; replacing it silently would
    let a human approve a call that was never shown to them while the real
    one waited forever.
    """


@runtime_checkable
class TraceStore(Protocol):
    """The run store: what a reviewer reads after the fact.

    One row per run plus the model calls that run spent. ``save_run`` is
    idempotent on ``trace_id`` -- a resumed run is saved again with more
    events, and the second write must extend the record, not fork it.
    """

    def save_run(self, record: RunRecord) -> None:
        """File one run. May raise: a run that cannot be persisted is a run
        that did not happen, and the orchestrator treats that as a failure of
        the request, not a note in a log."""
        ...

    def load_run(self, trace_id: str) -> AgentState | None:
        """The final state of one run, or ``None`` when no such run was
        filed. A miss is a reportable answer, not an exception -- the caller
        cannot tell "never happened" from "happened elsewhere", and only one of
        them is an error."""
        ...

    def record_model_call(self, trace_id: str, record: "ModelCallRecord") -> None:
        """Note what one model call cost.

        Must not raise: telemetry never fails a request. A turn that worked
        is not unwound because its cost could not be written down -- the same
        contract :class:`~agentic_erp_assistant.tools.audit.AuditSink` keeps
        for the same reason, and the one place the two stores differ:
        ``save_run`` is evidence the request happened, this is evidence about
        its price, and only the first is load-bearing.
        """
        ...


@runtime_checkable
class PauseStore(Protocol):
    """The pending-approval queue, as a store.

    A paused state is complete and serializable by design, so storing one is
    storing the turn itself. The one rule that matters is
    :meth:`claim`: settling a pause is atomic, and the caller that wins the
    claim is the only caller that may resume the run. An approval is a
    decision about one call, and it can be recorded once -- a store that
    executed an approved write twice would be the one failure in this system
    that no trace can talk its way out of.
    """

    def save(self, state: AgentState) -> None:
        """File a paused state.

        Only a paused state -- one whose route is ``request_approval`` and
        whose approval is ``pending``; anything else is a caller bug, and
        saving it would queue a turn nobody was ever asked to approve. Raises
        :class:`PauseAlreadyPending` when a pending pause already exists for
        the run.
        """
        ...

    def pending(self, trace_id: str) -> AgentState | None:
        """The pause waiting on a human for one run, or ``None``.

        What an approver's screen is built from: the tool, the arguments, the
        actor, all read off the state the engine paused.
        """
        ...

    def claim(self, trace_id: str, *, approved: bool) -> AgentState | None:
        """Settle the pending pause and hand back the paused state.

        Atomic: the decision is recorded and the pause leaves the queue in one
        step. Returns ``None`` when nothing was pending -- which covers both
        "already settled" and "never existed", because to a caller arriving
        late they mean the same thing: the decision is not theirs to make
        anymore. The caller that receives a state is the only caller whose
        ``approved`` counts.
        """
        ...