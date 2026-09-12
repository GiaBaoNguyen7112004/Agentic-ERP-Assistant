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

The short-term window -- the session's recent turns, ADR 0014 -- runs in the
same two places on the same grounds: recall before the engine
(:attr:`~agentic_erp_assistant.state.agent_state.AgentState.history`), and
recording after the run is filed, so a history row never names a run the trace
store does not have. Its third act, promotion, rides on consolidation: the
turns this one pushed out of the window are folded into the session summary
before the record is filed, so the ``history_promoted`` event lands in the
trace of the turn that caused the eviction.

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
from agentic_erp_assistant.state.reply_contract import ReplyContract
from agentic_erp_assistant.trace.ports import PauseStore, TraceStore
from agentic_erp_assistant.trace.records import RunOutcome, RunRecord

if TYPE_CHECKING:  # pragma: no cover - a name in a signature, not a dependency
    from agentic_erp_assistant.memory.models import MemoryDecision

__all__ = [
    "ApprovalAlreadySettled",
    "ContractDeclarerPort",
    "RunOrchestrator",
    "SessionHistoryPort",
    "TurnMemoryPort",
]

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


@runtime_checkable
class SessionHistoryPort(Protocol):
    """What the orchestrator assumes about the session's short-term window.

    Declared next to its one consumer, for the same reason
    :class:`TurnMemoryPort` is: the graph does not depend on the window at
    all, and
    :class:`~agentic_erp_assistant.memory.conversation.ConversationMemory`
    satisfies this structurally without importing it.

    The window is the other half of memory -- this principal's recent turns,
    verbatim and bounded, where durable memory is judged and paraphrased --
    and it lives by the same never-fail discipline: a store that is down
    costs the turn its context and its summary, never its answer, and the
    turns a broken store fails to promote are still in the store, retried on
    a later turn.
    """

    def recall(self, state: AgentState) -> Sequence[ConversationTurn]:
        """What this turn should be shown of its session's recent turns."""
        ...

    def record(self, state: AgentState, *, started_at: datetime) -> None:
        """File this finished turn into its session's window.

        A paused turn is recorded here too, with no reply, so the session's
        next request is shown the wait rather than the write.
        """
        ...

    def evicted(self, state: AgentState) -> Sequence[ConversationTurn]:
        """Turns this turn's arrival pushed out of the window, not yet promoted."""
        ...

    def promoted(self, turns: Sequence[ConversationTurn], *, run: str) -> None:
        """Mark these turns as folded into the session summary in ``run``."""
        ...


@runtime_checkable
class ContractDeclarerPort(Protocol):
    """What the orchestrator assumes about whatever declares a reply contract.

    Declared here rather than in :mod:`agentic_erp_assistant.engine.ports`,
    for the same reason :class:`TurnMemoryPort` is: the graph does not
    depend on a reply contract at all -- ``engine/nodes.py::think`` reads
    :attr:`~agentic_erp_assistant.state.agent_state.AgentState.contract`
    directly off the state, the same way it reads ``evidence`` or
    ``observations``, never through a port. That module's promise is that
    the workflow's entire external surface is four protocols on one screen,
    and adding a fifth would break it for a collaborator only the
    composition point has.

    Satisfied by
    :meth:`~agentic_erp_assistant.reasoning.planner.Planner.declare`
    structurally, without importing it.
    """

    def declare(self, state: AgentState) -> ReplyContract:
        """Say what a complete reply to ``state.request`` must rest on.

        Never raises for an unreadable declaration -- see
        :meth:`~agentic_erp_assistant.reasoning.planner.Planner.declare` for
        what "unreadable" absorbs and what it does not. A raised exception
        here means the call itself never produced a reply (network, auth,
        retries exhausted, budget), and :meth:`RunOrchestrator._declared` is
        where that is turned into "this turn continues unchecked" rather
        than a failed request.
        """
        ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _event(node: str, kind, detail: str) -> TraceEvent:
    return TraceEvent(node=node, kind=kind, detail=detail[:EVENT_DETAIL_MAX_CHARS])


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
        conversation: The short-term window, also bound to one session and
            actor. ``None`` is the same complete configuration: a turn with
            no session has no recent past to be shown, and one-shot scripts
            are shown nothing either. The two ports are independent -- the
            window can be present while durable memory is absent, and then
            turns are shown and recorded but nothing is promoted.
        declarer: What declares this turn's reply contract (ADR 0021).
            ``None`` is a complete configuration, the same as ``memory`` and
            ``conversation``: a turn with no declarer runs exactly as every
            turn did before this port existed --
            :attr:`~agentic_erp_assistant.state.agent_state.AgentState.contract`
            stays ``None``, and every completeness check in
            :mod:`agentic_erp_assistant.reasoning.completeness` reads that as
            "nothing to hold this turn to".
        now: The clock, injected so a test can make time deterministic and
            so the orchestrator itself never calls ``datetime.now`` at a
            distance.

    The ordering inside both methods is the load-bearing part and reads the
    same in both: short-term recall first, long-term recall second, the
    reply contract declared third, the engine fourth, consolidation -- and
    the promotion of evicted turns -- fifth, the run record sixth, only then
    a pause, and only then the finished turn joins the window. A pause
    therefore never exists without its run record beside it -- an approver
    opening the queue always has the trace to read -- and consolidation
    happens before the record is filed, so the events it produces are in the
    trace rather than in the next one. The window record goes last for the
    matching reason: the trace store's contract is that an unsaved run "did
    not happen", so a history row is only ever written about a run the
    trace already holds. The contract is declared after both recalls and
    before the engine for the same reason recall runs before the engine at
    all: it is background the turn is shown (or, here, held to) once, not a
    re-plan's business to revisit mid-turn -- and after recall specifically
    because it costs a model call of its own and gains nothing from running
    first.
    """

    runtime: WorkflowRuntime
    traces: TraceStore
    pauses: PauseStore
    memory: TurnMemoryPort | None = None
    conversation: SessionHistoryPort | None = None
    declarer: ContractDeclarerPort | None = None
    now: Callable[[], datetime] = _utc_now

    def handle(self, state: AgentState) -> AgentState:
        """Run a turn to its end and file it.

        Args:
            state: Where the turn starts. Usually unrouted; the engine takes
                it from there. Its ``history``, ``memories`` and ``contract``
                are filled here rather than by the caller -- recall and
                declaration are this class's job, and a caller that supplied
                any of them would be a second place that could happen.

        Returns:
            The final state, exactly as ``runtime.run`` produced it: terminal,
            or paused on an approval, with the memory, history and
            declaration events appended.
        """
        started = self.now()
        state = self._with_history(state)
        state = self._recalled(state)
        final = self.runtime.run(self._declared(state))
        final = self._consolidated(final)
        self.traces.save_run(self._record(started, final))
        if is_paused(final):
            self.pauses.save(final)
        self._recorded(final, started)
        return final

    def resume(
        self, trace_id: str, *, approved: bool, decided_by: str | None = None
    ) -> AgentState:
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
        made against one set of background and executed against another -- the
        same reasoning keeps short-term recall off this path. The window's
        record is updated, though: the turn filed while it paused is upserted
        in place with the reply it finally gave, so the session's next request
        is shown the outcome rather than the wait.

        Args:
            trace_id: The paused run being answered.
            approved: What the human said.
            decided_by: Who said it, recorded in the pause's row and in the
                trace event. ``None`` when the caller has nobody to name.

        Returns:
            The turn's new final state: terminal, or paused again on a
            different call -- which is filed as the run's next wait, the same
            as the first one.

        Raises:
            ApprovalAlreadySettled: The pause was not waiting anymore:
                already claimed, already denied, or never filed.
        """
        paused = self.pauses.claim(trace_id, approved=approved, decided_by=decided_by)
        if paused is None:
            raise ApprovalAlreadySettled(
                f"run {trace_id!r} has no pause waiting on a human; a decision "
                f"may be recorded once, and this one was"
            )

        started = self.now()
        final = self.runtime.resume_approval(
            paused, approved=approved, decided_by=decided_by
        )
        final = self._consolidated(final)
        self.traces.save_run(self._record(started, final))
        if is_paused(final):
            self.pauses.save(final)
        self._recorded(final, started)
        return final

    # -- memory, on both sides of the run ----------------------------------

    def _with_history(self, state: AgentState) -> AgentState:
        """The starting state with the session's recent turns attached.

        Never raises, for the same reason :meth:`_recalled` does not: a
        short-term store that is down costs this turn its context, and raising
        would cost the turn its answer. Nothing is recalled for a turn with no
        session -- a one-shot run has no recent past to be shown, the same
        rule the durable recall applies.

        ``resume`` deliberately skips this method entirely: the paused state
        already carries what it was shown, and re-recalling would mean the
        approver's decision was judged against one window and executed
        against another.
        """
        if self.conversation is None or state.session_id is None:
            return state

        try:
            recalled = tuple(self.conversation.recall(state))
        except Exception as error:  # noqa: BLE001 - a worse answer, not no answer
            logger.warning(
                "short-term recall failed for run %s (%s: %s); the turn "
                "continues with no history",
                state.trace_id,
                type(error).__name__,
                error,
            )
            return state

        if not recalled:
            return state
        return state.evolve(
            history=recalled,
            events=state.events
            + (
                _event(
                    "history", "history_recalled", f"{len(recalled)} prior turn(s)"
                ),
            ),
        )

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
            + (
                _event(
                    "memory", "memory_recalled", f"{len(recalled)} memor(y|ies) recalled"
                ),
            ),
        )

    def _declared(self, state: AgentState) -> AgentState:
        """The starting state with its reply contract attached (ADR 0021).

        Never raises. A declarer that is down, or whose call never returned
        (network, auth, retries exhausted, budget), leaves
        :attr:`~agentic_erp_assistant.state.agent_state.AgentState.contract`
        at ``None`` -- "this turn was never checked", exactly the value a
        replay or a hand-built state already carries. An *unreadable answer*
        from a call that did succeed is not this method's concern:
        :meth:`~agentic_erp_assistant.reasoning.planner.Planner.declare`
        already turns that into
        :data:`~agentic_erp_assistant.state.reply_contract.EMPTY_CONTRACT`,
        a real declaration of nothing needed, before it ever gets here.
        """
        if self.declarer is None:
            return state

        try:
            contract = self.declarer.declare(state)
        except Exception as error:  # noqa: BLE001 - a worse answer, not no answer
            logger.warning(
                "reply-contract declaration failed for run %s (%s: %s); the "
                "turn continues unchecked",
                state.trace_id,
                type(error).__name__,
                error,
            )
            return state

        detail = (
            f"needs={','.join(sorted(contract.needs))} "
            f"query={contract.document_query!r}"
            if contract.needs
            else "needs=(none)"
        )
        return state.evolve(
            contract=contract,
            events=state.events + (_event("contract", "contract_declared", detail),),
        )

    def _consolidated(self, final: AgentState) -> AgentState:
        """The final state with what was learned from it recorded.

        Skipped for a paused turn: its central question has no answer yet, and
        storing facts from it would remember a decision nobody has made. Never
        raises, for the reason
        :meth:`~agentic_erp_assistant.trace.ports.TraceStore.record_model_call`
        does not -- the turn already answered the user, and losing what it might
        have taught is not a reason to unwind that.

        The turns the short-term window evicted are folded in here, through
        the same consolidation call, and marked promoted only once
        consolidation has returned: a consolidation that raises leaves the
        turns unpromoted, still in the store, to be retried on the session's
        next turn rather than lost. That is also the price of
        ``conversation`` without ``memory`` -- promotion lives in
        consolidation, so turns accumulate unpromoted until a turn runs with
        the memory layer present.
        """
        if self.memory is None or is_paused(final):
            return final

        evicted = self._evicted(final)
        try:
            decisions = self.memory.consolidate(final, evicted=evicted)
        except Exception as error:  # noqa: BLE001 - see above
            logger.exception(
                "consolidation failed for run %s (%s); the turn stands and "
                "nothing was learned from it",
                final.trace_id,
                type(error).__name__,
            )
            return final

        if evicted:
            self._promoted(evicted, final)

        stored = [decision for decision in decisions if decision.stores]
        refused = [decision for decision in decisions if not decision.stores]

        events = final.events
        if evicted:
            events = events + (
                _event(
                    "history",
                    "history_promoted",
                    f"{len(evicted)} turn(s) folded into the session summary",
                ),
            )
        if stored:
            events = events + (
                _event(
                    "memory",
                    "memory_written",
                    ", ".join(
                        f"{decision.decision} ({decision.reason})"
                        for decision in stored
                    ),
                ),
            )
        for decision in refused:
            events = events + (
                _event("memory", "memory_rejected", f"{decision.rejection}: {decision.reason}"),
            )

        return final if events is final.events else final.evolve(events=events)

    def _evicted(self, final: AgentState) -> Sequence[ConversationTurn]:
        """What this turn's arrival pushed out of the window, not yet promoted.

        Never raises: a short-term store that is down yields no eviction this
        turn, and the turns it would have named are still in the store to be
        folded in on a later turn. ``()`` for a turn with no session, before
        the store is asked -- a question with a structural answer is not
        worth an I/O round trip.
        """
        if self.conversation is None or final.session_id is None:
            return ()
        try:
            return tuple(self.conversation.evicted(final))
        except Exception as error:  # noqa: BLE001 - retried on a later turn
            logger.warning(
                "asking the window for evicted turns failed for run %s (%s: "
                "%s); they stay unpromoted and are retried",
                final.trace_id,
                type(error).__name__,
                error,
            )
            return ()

    def _promoted(self, turns: Sequence[ConversationTurn], final: AgentState) -> None:
        """Mark the evicted turns as folded into the session summary.

        Never raises, and a failure here loses nothing durable: the turns
        were already folded into the summary, and promotion is idempotent,
        so the next turn's re-fold supersedes the same summary harmlessly.
        What the watermark being stale does mean is that the turns stay
        visible to a later eviction and are folded again -- one redundant
        summary row and one more audit line, not a wrong one.
        """
        if self.conversation is None:  # pragma: no cover - _consolidated checked
            return
        try:
            self.conversation.promoted(turns, run=final.trace_id)
        except Exception as error:  # noqa: BLE001 - a re-fold supersedes harmlessly
            logger.warning(
                "marking %d promoted turn(s) failed for run %s (%s: %s); they "
                "will be folded again on the session's next turn",
                len(turns),
                final.trace_id,
                type(error).__name__,
                error,
            )

    def _recorded(self, final: AgentState, started: datetime) -> None:
        """File the finished turn into its session's window.

        Runs after the run record is filed and after a pause is saved: the
        trace store's contract is that an unsaved run "did not happen", so a
        history row is only ever written about a run the trace already holds.
        A paused turn is recorded here too, with no reply, so the session's
        next request is shown the wait rather than the write.

        Never raises, for the reason :meth:`_with_history` does not: a turn
        that failed to join the window is one the session's next turn is not
        shown, not a turn that failed. A turn with no session is skipped here
        rather than trusted to the store -- a turn outside a session is never
        history, and the question has a structural answer.
        """
        if self.conversation is None or final.session_id is None:
            return
        try:
            self.conversation.record(final, started_at=started)
        except Exception as error:  # noqa: BLE001 - one line of context lost, not the answer
            logger.warning(
                "recording the turn into the window failed for run %s (%s: %s)",
                final.trace_id,
                type(error).__name__,
                error,
            )

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
