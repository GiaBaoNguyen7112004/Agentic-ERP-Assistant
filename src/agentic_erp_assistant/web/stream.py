"""The bridge from one turn's blocking worker thread to an async SSE response.

The engine is synchronous and blocking (fact 1 in the code plan): a turn runs
node after node, each one making a blocking HTTP call, on a thread from the
default executor. :class:`TurnStream` is the one object that thread and the
event loop both touch -- every callback the composition root wires a turn's
ports to (:attr:`~agentic_erp_assistant.engine.workflow.WorkflowRuntime.
observer`, :attr:`~agentic_erp_assistant.tools.gateway.ToolGateway.on_event`,
:class:`~agentic_erp_assistant.llm.streaming.AnswerStreamSink`) calls a method
on this class from the worker thread, and every method hands its event to the
loop with :meth:`asyncio.loop.call_soon_threadsafe` rather than touching the
queue directly -- an ``asyncio.Queue`` is not thread-safe, and the one
cross-thread handoff in this whole turn happens here and nowhere else.
"""

import asyncio
import time
from collections.abc import AsyncIterator

from agentic_erp_assistant.llm.inspection import ModelRequestSnapshot, ModelResponseSnapshot
from agentic_erp_assistant.llm.telemetry import ModelCallRecord
from agentic_erp_assistant.rag.retriever import RetrievalOutcome
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.web.protocol import (
    clip_text,
    ContextEvent,
    ContractOut,
    EvidenceOut,
    HistoryTurnOut,
    MemoryOut,
    MessageOut,
    ModelCallOut,
    ModelRequestOut,
    ModelResponseOut,
    ResetEvent,
    RetrievalHitOut,
    RetrievalOut,
    ServerEvent,
    StepEvent,
    TextOut,
    TokenEvent,
    ToolOutcomeOut,
    TraceRow,
)

__all__ = ["TurnStream"]


def _history_out(turn: ConversationTurn) -> HistoryTurnOut:
    return HistoryTurnOut(
        trace_id=turn.trace_id,
        request=turn.request,
        response=turn.response,
        route=turn.route,
        failure=turn.failure,
        tool_name=turn.tool_name,
        approval=turn.approval,
        started_at=turn.started_at,
        finished_at=turn.finished_at,
    )


def _memory_out(memory: MemoryRecord) -> MemoryOut:
    return MemoryOut(
        memory_id=memory.memory_id,
        kind=memory.kind,
        key=memory.key,
        statement=memory.statement,
        confidence=memory.confidence,
        recorded_in_run=memory.recorded_in_run,
        recorded_at=memory.recorded_at,
        supersedes=memory.supersedes,
        links=memory.links,
        required_scope=memory.required_scope,
        actor=memory.actor,
        project_code=memory.project_code,
        session_id=memory.session_id,
    )


def _evidence_out(snippet: EvidenceSnippet) -> EvidenceOut:
    return EvidenceOut(
        source_id=snippet.source_id,
        locator=snippet.locator,
        tag=snippet.tag,
        text=clip_text(snippet.text),
    )


def _observation_out(outcome: ToolOutcome) -> ToolOutcomeOut:
    return ToolOutcomeOut(
        tool_name=outcome.tool_name,
        arguments_summary=outcome.arguments_summary,
        status=outcome.status,
        summary=clip_text(outcome.summary),
        source_ids=outcome.source_ids,
        error=outcome.error,
        attempts=outcome.attempts,
        retry_after_seconds=outcome.retry_after_seconds,
    )


def _request_out(snapshot: ModelRequestSnapshot) -> ModelRequestOut:
    """``llm/inspection.py`` already clipped each message's content to its
    own (smaller) bound; this rebuilds ``TextOut`` from the clipped text and
    the snapshot's separately-carried real length, rather than clipping a
    second time against a different limit."""
    return ModelRequestOut(
        kind=snapshot.kind,
        messages=tuple(
            MessageOut(role=role, content=_text_out_from_clipped(content, chars))
            for (role, content), chars in zip(
                snapshot.messages, snapshot.message_chars, strict=True
            )
        ),
        tools=snapshot.tools,
        tool_choice=snapshot.tool_choice,
        temperature=snapshot.temperature,
    )


def _response_out(snapshot: ModelResponseSnapshot) -> ModelResponseOut:
    content = (
        _text_out_from_clipped(snapshot.content, snapshot.content_chars or 0)
        if snapshot.content is not None
        else None
    )
    return ModelResponseOut(
        content=content,
        tool_name=snapshot.tool_name,
        arguments=dict(snapshot.arguments) if snapshot.arguments is not None else None,
        stop_reason=snapshot.stop_reason,
    )


def _text_out_from_clipped(clipped: str, real_chars: int) -> TextOut:
    """Build a :class:`TextOut` from text a caller already clipped
    elsewhere, and the real length it was clipped from -- never re-clips."""
    return TextOut(text=clipped, truncated=real_chars > len(clipped), chars=real_chars)


def _model_call_out(
    record: ModelCallRecord,
    request: ModelRequestSnapshot | None,
    response: ModelResponseSnapshot | None,
) -> ModelCallOut:
    return ModelCallOut(
        model=record.model,
        outcome=record.outcome,
        estimated_input_tokens=record.estimated_input_tokens,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cost_usd=record.cost_usd,
        latency_seconds=record.latency_seconds,
        attempts=record.attempts,
        occurred_at=record.occurred_at,
        detail=record.detail,
        request=_request_out(request) if request is not None else None,
        response=_response_out(response) if response is not None else None,
    )


def _retrieval_out(
    query: str, limit: int, outcome: RetrievalOutcome, minimum_similarity: float
) -> RetrievalOut:
    return RetrievalOut(
        query=query,
        limit=limit,
        hits=tuple(
            RetrievalHitOut(
                chunk_id=hit.chunk.chunk_id,
                document_id=hit.chunk.document_id,
                locator=hit.chunk.locator,
                title=hit.chunk.title,
                score=hit.score,
                ranks=dict(hit.ranks),
                scores=dict(hit.scores),
            )
            for hit in outcome.hits
        ),
        best_similarity=outcome.best_similarity,
        minimum_similarity=minimum_similarity,
        dense_candidates=outcome.dense_candidates,
        lexical_candidates=outcome.lexical_candidates,
        gated=outcome.gated,
    )


class TurnStream:
    """Implements :class:`~agentic_erp_assistant.composition.turn.
    TurnStreamLike` and adds the plumbing a request handler needs on top of
    it: :meth:`emit` for service-level events, :meth:`close`, and
    :meth:`events` to read them back.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, *, start_seq: int = 0) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[ServerEvent | None] = asyncio.Queue()
        self._emitted = start_seq
        """How many of AgentState.events have already become a TraceRow.

        Seeded from ``start_seq`` on a resumed turn: the paused state's own
        events were already streamed the first time the client saw them, and
        re-emitting them here would duplicate a row the client already has.
        """
        self._last: AgentState | None = None
        """The state :meth:`context` or the previous :meth:`step` left off
        at -- the baseline every delta field on the next
        :class:`~agentic_erp_assistant.web.protocol.StepEvent` is computed
        against, and what :meth:`trace_event` reads ``step_count`` off of to
        number a live gateway row. ``None`` only before :meth:`context` has
        ever been called, which composition wires to fire before this
        object ever sees a node -- see ``composition/turn.py``.
        """
        self._last_perf = time.perf_counter()
        """The clock :meth:`step`'s ``elapsed_ms`` measures against. Reset
        by :meth:`context` and by every :meth:`step` call, so each step
        reports the wall clock since the *previous* one (or since the
        context was attached, for the first)."""
        self._pending_model_calls: list[ModelCallOut] = []
        """Calls recorded since the last :meth:`context`/:meth:`step` drained
        this. A model call is recorded (``LLMGateway._record``) the instant
        it finishes, which can be *before* the next node's own batch of
        trace rows has flushed -- the declaration call, in particular,
        finishes before :meth:`context` is even called at all (see
        ``engine/orchestrator.py``: recall, then declare, then the engine).
        Draining on whichever of the two fires next, rather than requiring
        one specific order, is what lets both attribute correctly."""
        self._pending_retrieval: RetrievalOut | None = None
        """The one search this turn's current node made, if any, since the
        last :meth:`step` drained this."""

    def _put(self, event: ServerEvent | None) -> None:
        """Hand one event (or the closing ``None``) to the loop, from
        whichever thread is calling. The only place this class touches the
        queue across the thread boundary."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def _elapsed_ms(self) -> float:
        """Milliseconds since the last time this was called (or since
        construction), and reset the clock for the next call."""
        now = time.perf_counter()
        elapsed = (now - self._last_perf) * 1000.0
        self._last_perf = now
        return elapsed

    # -- TurnStreamLike: called from the worker thread, mid-run -------------

    def context(self, state: AgentState) -> None:
        """Called once, before the engine runs (or resumes) -- what the
        orchestrator attached to the state before handing it to the engine:
        the session's recent turns, recalled memory, and the declared reply
        contract. See :class:`~agentic_erp_assistant.web.protocol.
        ContextEvent` for why this exists beside the ``history_recalled``/
        ``memory_recalled``/``contract_declared`` rows that still stream as
        usual, folded into the first node's own batch.

        Also seeds :attr:`_last`, the baseline every later :meth:`step`
        computes its delta fields against -- on a resumed turn this is the
        paused state itself, so a resumed stream's first ``step`` correctly
        reports only what changed *since the pause*, not since the turn
        began.
        """
        contract = None
        if state.contract is not None:
            contract = ContractOut(
                needs=tuple(sorted(state.contract.needs)),
                document_query=state.contract.document_query,
            )
        model_calls = tuple(self._pending_model_calls)
        self._pending_model_calls = []
        self._put(
            ContextEvent(
                request=state.request,
                history=tuple(_history_out(turn) for turn in state.history),
                memories=tuple(_memory_out(memory) for memory in state.memories),
                contract=contract,
                model_calls=model_calls,
                state=state,
            )
        )
        self._last = state
        self._elapsed_ms()  # reset the clock; node 1's own time starts now

    def step(self, state: AgentState) -> None:
        """The engine's own observer hook -- called after every node.

        Streams every new entry of ``state.events`` since the last call (as
        :class:`~agentic_erp_assistant.web.protocol.TraceRow`, ``source``
        ``"engine"``), then a :class:`~agentic_erp_assistant.web.protocol.
        StepEvent` summarizing where the turn now stands and what this one
        step changed, against :attr:`_last`.
        """
        new_events = state.events[self._emitted :]
        node: str | None = None
        for offset, event in enumerate(new_events):
            self._put(
                TraceRow(
                    seq=self._emitted + offset,
                    node=event.node,
                    kind=event.kind,
                    detail=event.detail,
                    source="engine",
                    step=None,
                )
            )
            if event.kind == "node_entered":
                node = event.node
        self._emitted += len(new_events)

        last = self._last
        evidence = None
        if last is None or state.evidence != last.evidence:
            evidence = tuple(_evidence_out(snippet) for snippet in state.evidence)
        new_observation_count = len(state.observations) - (
            len(last.observations) if last is not None else 0
        )
        observations = (
            tuple(_observation_out(outcome) for outcome in state.observations[-new_observation_count:])
            if new_observation_count > 0
            else ()
        )

        model_calls = tuple(self._pending_model_calls)
        self._pending_model_calls = []
        retrieval = self._pending_retrieval
        self._pending_retrieval = None

        self._put(
            StepEvent(
                route=state.route,
                tool_name=state.tool_name,
                tool_arguments=dict(state.tool_arguments) if state.tool_arguments else None,
                tool_mutating=state.tool_mutating,
                approval=state.approval,
                step_count=state.step_count,
                terminal=state.terminal,
                node=node,
                elapsed_ms=self._elapsed_ms(),
                evidence=evidence,
                observations=observations,
                response=state.response,
                failure=state.failure,
                error_detail=state.error_detail,
                draft=state.draft,
                redirected_needs=tuple(sorted(state.redirected_needs)),
                retry_count=state.retry_count,
                retrieval=retrieval,
                model_calls=model_calls,
                state=state,
            )
        )
        self._last = state

    def trace_event(self, event: TraceEvent) -> None:
        """A gateway-internal event (a retry, an approval record) -- never
        stored on ``AgentState.events``, so it carries no ``seq``.

        Stamped with the node execution under way when it fired --
        ``self._last``'s ``step_count`` plus one, since a live gateway row
        is always produced *during* the node about to close on the next
        ``step_count`` (see ``engine/nodes.py``'s call sites and
        :class:`~agentic_erp_assistant.web.protocol.TraceRow`'s own
        docstring for why).
        """
        self._put(
            TraceRow(
                seq=None,
                node=event.node,
                kind=event.kind,
                detail=event.detail,
                source="tool_gateway",
                step=(self._last.step_count if self._last is not None else 0) + 1,
            )
        )

    def model_call(
        self,
        record: ModelCallRecord,
        request: ModelRequestSnapshot | None,
        response: ModelResponseSnapshot | None,
    ) -> None:
        """Satisfies :class:`~agentic_erp_assistant.llm.inspection.
        ModelCallInspector` -- bound to ``LLMGateway.inspector``
        (``composition/turn.py``) for the answering gateway, which is also
        the planner's own model, so a routing decision, an answer, and the
        reply contract's declaration call all arrive here.

        Buffered rather than put on the queue directly: a call finishes
        mid-node (or, for the declaration, before :meth:`context` has even
        been called), and only :meth:`context`/:meth:`step` know which
        event this call's snapshot belongs on.
        """
        self._pending_model_calls.append(_model_call_out(record, request, response))

    def retrieval(
        self, query: str, limit: int, outcome: RetrievalOutcome, minimum_similarity: float
    ) -> None:
        """Called by ``composition/turn.py``'s ``InspectedRetriever`` right
        after a search, with the same :class:`~agentic_erp_assistant.rag.
        retriever.RetrievalOutcome` the retriever computed for its own
        return value -- nothing here re-runs the search.

        Buffered like :meth:`model_call`, for the same reason: the search
        happens inside ``retrieve_and_answer``, mid-node, and only the
        `step()` that closes that node knows to attach it.
        """
        self._pending_retrieval = _retrieval_out(query, limit, outcome, minimum_similarity)

    def delta(self, text: str) -> None:
        """More reply text has arrived. See
        :class:`~agentic_erp_assistant.llm.streaming.AnswerStreamSink`."""
        self._put(TokenEvent(text=text))

    def reset(self) -> None:
        """A retried attempt is starting over; the client must discard the
        tokens it has shown so far for this reply."""
        self._put(ResetEvent())

    # -- the rest of the turn: called from the service, mid- or post-run ----

    def emit(self, event: ServerEvent) -> None:
        """Hand a fully-built, service-level event straight to the queue --
        ``turn_started``, ``approval_required``, ``answer``,
        ``turn_finished``, ``error``. These are decided by
        :mod:`agentic_erp_assistant.web.service`, not derived from a state
        the way the four methods above are."""
        self._put(event)

    def flush_events(self, state: AgentState) -> None:
        """Catch up on any trace rows the step-by-step observer never saw.

        Consolidation and short-term-window promotion run *after*
        :meth:`~agentic_erp_assistant.engine.workflow.WorkflowRuntime.run`
        returns, inside :class:`~agentic_erp_assistant.engine.orchestrator.
        RunOrchestrator`, which the observer is never attached to -- so a
        ``memory_written`` or ``history_promoted`` event lands on the final
        state's ``events`` with nobody having streamed it. Called once,
        after ``handle()``/``resume()`` returns, with the state they
        actually produced. Trace rows only: there is no further ``step`` to
        summarize once the run is filed.
        """
        new_events = state.events[self._emitted :]
        for offset, event in enumerate(new_events):
            self._put(
                TraceRow(
                    seq=self._emitted + offset,
                    node=event.node,
                    kind=event.kind,
                    detail=event.detail,
                    source="engine",
                    step=None,
                )
            )
        self._emitted += len(new_events)

    def close(self) -> None:
        """Signal the end of the stream. Idempotent to call more than once
        (the sentinel is just another item), though a caller should not
        need to."""
        self._put(None)

    async def events(self) -> AsyncIterator[ServerEvent]:
        """Read events back, in the order they were put, until :meth:`close`.

        The async side of the bridge: this runs on the event loop, in the
        request handler that owns the ``StreamingResponse``, and is the only
        method on this class that is ever awaited.
        """
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event
