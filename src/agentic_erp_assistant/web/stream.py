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
from collections.abc import AsyncIterator

from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.web.protocol import (
    ResetEvent,
    ServerEvent,
    StepEvent,
    TokenEvent,
    TraceRow,
)

__all__ = ["TurnStream"]


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

    def _put(self, event: ServerEvent | None) -> None:
        """Hand one event (or the closing ``None``) to the loop, from
        whichever thread is calling. The only place this class touches the
        queue across the thread boundary."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    # -- TurnStreamLike: called from the worker thread, mid-run -------------

    def step(self, state: AgentState) -> None:
        """The engine's own observer hook -- called after every node.

        Streams every new entry of ``state.events`` since the last call (as
        :class:`~agentic_erp_assistant.web.protocol.TraceRow`, ``source``
        ``"engine"``), then a :class:`~agentic_erp_assistant.web.protocol.
        StepEvent` summarizing where the turn now stands.
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
                )
            )
        self._emitted += len(new_events)

        self._put(
            StepEvent(
                route=state.route,
                tool_name=state.tool_name,
                tool_arguments=dict(state.tool_arguments) if state.tool_arguments else None,
                tool_mutating=state.tool_mutating,
                approval=state.approval,
                step_count=state.step_count,
                terminal=state.terminal,
            )
        )

    def trace_event(self, event: TraceEvent) -> None:
        """A gateway-internal event (a retry, an approval record) -- never
        stored on ``AgentState.events``, so it carries no ``seq``."""
        self._put(
            TraceRow(
                seq=None,
                node=event.node,
                kind=event.kind,
                detail=event.detail,
                source="tool_gateway",
            )
        )

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
