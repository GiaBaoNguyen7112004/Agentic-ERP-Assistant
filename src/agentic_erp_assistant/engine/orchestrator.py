"""The composition point: run a turn, then file what the run produced.

The engine -- :class:`~agentic_erp_assistant.engine.workflow.WorkflowRuntime`
-- is deliberately single-minded. It applies nodes until a turn ends or pauses
and hands back a state, with no opinion about where that state goes next. But
a paused state held only in a variable dies with the process that produced
it, and CLAUDE.md's evidence rule asks for a trace that survives the request.
This module is the seam between those two facts: it drives the engine and
files the results with the stores behind
:mod:`agentic_erp_assistant.trace.ports` -- the run record either way, and
the pause when there is one.

It is the surface the future web layer calls: ``handle`` for a new request,
``resume`` for an approver's answer, and nothing else. "Orchestrator" rather
than "service": it drives the engine and the evidence stores together, and
the word says that; a service could be anything.

Import direction: this module imports the engine and the trace ports, and
nothing imports it back. The engine stays pure -- it never learns where its
states are filed -- which is what lets the same engine run in tests against
the in-memory stores and in production against Postgres without a change to
either side.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from agentic_erp_assistant.engine.workflow import WorkflowRuntime, is_paused
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.trace.ports import PauseStore, TraceStore
from agentic_erp_assistant.trace.records import RunOutcome, RunRecord

__all__ = ["ApprovalAlreadySettled", "RunOrchestrator"]

logger = logging.getLogger(__name__)


class ApprovalAlreadySettled(RuntimeError):
    """A decision arrived for an approval that was no longer waiting.

    The pause store settles once: whoever claims the pending row first wins,
    and every caller after that gets nothing -- whoever they are, whatever
    answer they brought. This is the typed shape of getting nothing. It is
    raised rather than returned because a caller asking twice is disagreeing
    with the record, and quietly accepting the second answer would audit an
    approval nobody was shown.
    """


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class RunOrchestrator:
    """Runs turns through the engine and persists every outcome.

    Attributes:
        runtime: The engine doing the work. Untouched by this class -- the
            orchestrator decides where results are filed, never how they are
            produced.
        traces: Where the run record goes, whatever the outcome. A run that
            ended, a run that paused, and a run resumed from a pause are all
            one record per trace id.
        pauses: Where a paused state waits for its human, and where a
            decision is settled exactly once.
        now: The clock, injected so a test can make time deterministic and
            so the orchestrator itself never calls ``datetime.now`` at a
            distance.

    The ordering inside both methods is the load-bearing part and reads the
    same in both: the engine runs first, the run record is filed second, and
    only then is a pause filed. A pause therefore never exists without its
    run record beside it -- an approver opening the queue always has the
    trace to read.
    """

    runtime: WorkflowRuntime
    traces: TraceStore
    pauses: PauseStore
    now: Callable[[], datetime] = _utc_now

    def handle(self, state: AgentState) -> AgentState:
        """Run a turn to its end and file it.

        Args:
            state: Where the turn starts. Usually unrouted; the engine takes
                it from there.

        Returns:
            The final state, exactly as ``runtime.run`` produced it: terminal,
            or paused on an approval.
        """
        started = self.now()
        final = self.runtime.run(state)
        self.traces.save_run(self._record(started, final))
        if is_paused(final):
            self.pauses.save(final)
        return final

    def resume(self, trace_id: str, *, approved: bool) -> AgentState:
        """Settle one approval and carry the turn on from it.

        The claim happens before anything runs, and that order is the
        guarantee. :meth:`~agentic_erp_assistant.trace.ports.PauseStore.claim`
        settles the pending row atomically -- only one caller in the world
        gets the state back -- so the write an approval authorizes happens at
        most once no matter how many times the decision is submitted, how
        many processes submit it, or which of them wins the race. A caller
        who loses the claim learns it here, before the engine is touched.

        Args:
            trace_id: The paused run being answered.
            approved: What the human said.

        Returns:
            The turn's new final state: terminal, or paused again on a
            different call -- which is filed as the run's next wait, the same
            as the first one.

        Raises:
            ApprovalAlreadySettled: The pause was not waiting anymore:
                already claimed, already denied, or never filed.
        """
        paused = self.pauses.claim(trace_id, approved=approved)
        if paused is None:
            raise ApprovalAlreadySettled(
                f"run {trace_id!r} has no pause waiting on a human; a decision "
                f"may be recorded once, and this one was"
            )

        started = self.now()
        final = self.runtime.resume_approval(paused, approved=approved)
        self.traces.save_run(self._record(started, final))
        if is_paused(final):
            self.pauses.save(final)
        return final

    def _record(self, started: datetime, final: AgentState) -> RunRecord:
        """The run record for a finished segment, in the shape the store takes.

        One record per save, describing the segment that just ran: ``handle``
        records the turn, ``resume`` records the continuation, and the store
        keeps them as one run by trace id. The outcome is read from the state
        rather than passed in, so this module cannot file a paused state as
        terminal or the reverse -- :class:`RunRecord` would refuse it, and
        the same read lives in its validator for the same reason.
        """
        outcome: RunOutcome = "paused" if is_paused(final) else "terminal"
        return RunRecord(
            trace_id=final.trace_id,
            outcome=outcome,
            started_at=started,
            finished_at=self.now(),
            state=final,
        )