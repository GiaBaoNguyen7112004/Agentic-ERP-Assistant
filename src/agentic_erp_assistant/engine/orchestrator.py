"""The composition point: recall, run a turn, consolidate, then file it.

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

Memory happens here, on both sides of the run
---------------------------------------------

Recall fills :attr:`~agentic_erp_assistant.state.agent_state.AgentState.memories`
before the engine is touched; consolidation asks what the finished turn was
worth after it hands the state back. Neither is a node, and that is the
decision ADR 0011 records. Three reasons, and the second is the load-bearing
one:

* it keeps the engine's external surface at four ports, so a graph test still
  builds with four fakes;
* memory work is not the turn's work, so it must not spend the step budget --
  a turn that recalled twice and re-planned once would hit ``max_steps``
  because of bookkeeping;
* consolidation runs only on a *terminal* state. A paused turn is one whose
  central question -- will a human approve this? -- has no answer yet, and
  storing facts from it would remember a decision nobody has made.

And it can never fail a request. Recall that raises produces a turn with no
background; consolidation that raises produces a turn nobody learned from. Both
are logged and both leave the answer standing, because the alternative is a
question that was answered correctly and reported as an error.

Import direction: this module imports the engine and the trace ports, and
nothing imports it back. The engine stays pure -- it never learns where its
states are filed -- which is what lets the same engine run in tests against
the in-memory stores and in production against Postgres without a change to
either side.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentic_erp_assistant.engine.workflow import WorkflowRuntime, is_paused
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.events import EVENT_DETAIL_MAX_CHARS, TraceEvent
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.trace.ports import PauseStore, TraceStore
from agentic_erp_assistant.trace.records import RunOutcome, RunRecord

if TYPE_CHECKING:  # pragma: no cover - a name in a signature, not a dependency
    from agentic_erp_assistant.memory.models import MemoryDecision

__all__ = ["ApprovalAlreadySettled", "RunOrchestrator", "TurnMemoryPort"]

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


@runtime_checkable
class TurnMemoryPort(Protocol):
    """What the orchestrator assumes about memory, bound to one turn's scope.

    Declared here rather than in :mod:`agentic_erp_assistant.engine.ports`, and
    that is the point of the placement: the graph does not depend on memory at
    all. That module's promise is that the workflow's entire external surface is
    four protocols on one screen, and adding a fifth would break it for a
    collaborator only the composition point has.

    An implementation is already bound to an actor, a project and a session --
    see
    :meth:`~agentic_erp_assistant.memory.service.MemoryService.for_scope`. The
    orchestrator never passes a scope, because a scope it could pass is one it
    could pass wrongly.
    """

    def recall(self, state: AgentState) -> Sequence[MemoryRecord]:
        """What this turn should be shown. May be empty, and usually is."""
        ...

    def consolidate(
        self, state: AgentState, *, evicted: Sequence[ConversationTurn] = ()
    ) -> Sequence["MemoryDecision"]:
        """What the finished turn was worth, decided and written down.

        Args:
            state: The turn, terminal.
            evicted: Turns the short-term window no longer has room for, once
                this one joins it. Folded into the session summary when
                non-empty; see
                :mod:`agentic_erp_assistant.memory.promotion`. The orchestrator
                supplies this from
                :meth:`~agentic_erp_assistant.memory.conversation.ConversationMemory.evicted`,
                not from ``state``.

        Returns every decision, refusals included: a run that refused four
        proposals and kept one is a run where the policy worked, and it must not
        read the same as one that stored five.
        """
        ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _event(kind, detail: str) -> TraceEvent:
    return TraceEvent(
        node="memory", kind=kind, detail=detail[:EVENT_DETAIL_MAX_CHARS]
    )


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
        memory: What this turn remembers and learns, already bound to a scope.
            ``None`` is a complete configuration and not a degraded one: a
            replay, an evaluation case and a one-shot script have no
            conversation to remember anything for.
        now: The clock, injected so a test can make time deterministic and
            so the orchestrator itself never calls ``datetime.now`` at a
            distance.

    The ordering inside both methods is the load-bearing part and reads the
    same in both: recall first, the engine second, the run record third, and
    only then a pause. A pause therefore never exists without its run record
    beside it -- an approver opening the queue always has the trace to read --
    and consolidation happens before the record is filed, so the events it
    produces are in the trace rather than in the next one.
    """

    runtime: WorkflowRuntime
    traces: TraceStore
    pauses: PauseStore
    memory: TurnMemoryPort | None = None
    now: Callable[[], datetime] = _utc_now

    def handle(self, state: AgentState) -> AgentState:
        """Run a turn to its end and file it.

        Args:
            state: Where the turn starts. Usually unrouted; the engine takes
                it from there. Its ``memories`` are filled here rather than by
                the caller -- recall is this class's job, and a caller that
                supplied them would be a second place recall could happen.

        Returns:
            The final state, exactly as ``runtime.run`` produced it: terminal,
            or paused on an approval, with the memory events appended.
        """
        started = self.now()
        final = self.runtime.run(self._recalled(state))
        final = self._consolidated(final)
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

        Recall does not run again. The paused state already carries what this
        turn was shown, and re-recalling would mean an approver's decision was
        made against one set of background and executed against another.

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
        final = self._consolidated(final)
        self.traces.save_run(self._record(started, final))
        if is_paused(final):
            self.pauses.save(final)
        return final

    # -- memory, on both sides of the run ----------------------------------

    def _recalled(self, state: AgentState) -> AgentState:
        """The starting state with its background filled in.

        Never raises. A memory layer that is down produces a turn with no
        background, which answers the question slightly worse -- and raising
        would produce no answer at all, which is worse than that.
        """
        if self.memory is None:
            return state

        try:
            recalled = tuple(self.memory.recall(state))
        except Exception as error:  # noqa: BLE001 - a worse answer, not no answer
            logger.warning(
                "recall failed for run %s (%s: %s); the turn continues with no "
                "background",
                state.trace_id,
                type(error).__name__,
                error,
            )
            return state

        if not recalled:
            return state
        return state.evolve(
            memories=recalled,
            events=state.events
            + (_event("memory_recalled", f"{len(recalled)} memor(y|ies) recalled"),),
        )

    def _consolidated(self, final: AgentState) -> AgentState:
        """The final state with what was learned from it recorded.

        Skipped for a paused turn: its central question has no answer yet, and
        storing facts from it would remember a decision nobody has made. Never
        raises, for the reason
        :meth:`~agentic_erp_assistant.trace.ports.TraceStore.record_model_call`
        does not -- the turn already answered the user, and losing what it might
        have taught is not a reason to unwind that.
        """
        if self.memory is None or is_paused(final):
            return final

        try:
            decisions = self.memory.consolidate(final)
        except Exception as error:  # noqa: BLE001 - see above
            logger.exception(
                "consolidation failed for run %s (%s); the turn stands and "
                "nothing was learned from it",
                final.trace_id,
                type(error).__name__,
            )
            return final

        stored = [decision for decision in decisions if decision.stores]
        refused = [decision for decision in decisions if not decision.stores]

        events = final.events
        if stored:
            events = events + (
                _event(
                    "memory_written",
                    ", ".join(
                        f"{decision.decision} ({decision.reason})"
                        for decision in stored
                    ),
                ),
            )
        for decision in refused:
            events = events + (
                _event("memory_rejected", f"{decision.rejection}: {decision.reason}"),
            )

        return final if events is final.events else final.evolve(events=events)

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
