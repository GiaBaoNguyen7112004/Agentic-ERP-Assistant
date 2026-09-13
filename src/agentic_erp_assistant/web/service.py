"""The one place an HTTP request becomes a call into the composition root.

Every method here is a plain, synchronous function -- meant to be run on a
worker thread via ``loop.run_in_executor`` from an async route in
:mod:`agentic_erp_assistant.web.app`, the same way the engine itself runs.
Nothing in this module is async, and nothing in it imports FastAPI: the
three exceptions below are the only vocabulary the route layer needs to turn
this module's refusals into HTTP status codes.
"""

import logging
import uuid
from dataclasses import dataclass

from agentic_erp_assistant.composition.resources import AppResources
from agentic_erp_assistant.composition.turn import TurnPorts, build_turn, initial_state
from agentic_erp_assistant.composition.users import UnknownUser, User
from agentic_erp_assistant.engine.orchestrator import ApprovalAlreadySettled
from agentic_erp_assistant.engine.workflow import is_paused
from agentic_erp_assistant.persistence.postgres_pause import PostgresPauseStore
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.tools.models import ARGUMENTS_SUMMARY_MAX_CHARS
from agentic_erp_assistant.web.protocol import (
    AnswerEvent,
    ApprovalRequiredEvent,
    ErrorEvent,
    MemoryAuditOut,
    ModelCallOut,
    ModelCallTotalsOut,
    ObservationOut,
    TurnFinishedEvent,
    TurnStartedEvent,
    parse_citations,
)
from agentic_erp_assistant.web.stream import TurnStream

__all__ = [
    "ChatService",
    "Conflict",
    "Forbidden",
    "NotFound",
    "PendingDecision",
]

logger = logging.getLogger(__name__)


class Forbidden(RuntimeError):
    """The actor named is not entitled to do what was asked -- unknown, or
    lacking ``approvals.decide``. Maps to HTTP 403."""


class NotFound(RuntimeError):
    """No pause was ever recorded for this trace id. Maps to HTTP 404."""


class Conflict(RuntimeError):
    """The pause exists but has already been settled -- by this caller's
    request or another's. Maps to HTTP 409."""


def _summarize_arguments(tool_name: str, arguments: dict[str, object]) -> str:
    """One line an approver reads: ``create_risk(project_id=atlas, ...)``.

    Deliberately not imported from the gateway's own private renderer of
    the same shape (``tools.gateway._summarize``, ``persistence.
    postgres_pause._summarize``): both take a different input
    (``ToolRequest``, a state) than this one has (a route's own arguments
    dict), and a third small renderer here is cheaper than threading a
    fourth shared shape through three modules for one display line.
    """
    parts = []
    for key, value in arguments.items():
        rendered = str(value)
        if len(rendered) > 40:
            rendered = rendered[:39] + "…"
        parts.append(f"{key}={rendered}")
    line = f"{tool_name}({', '.join(parts)})"
    if len(line) > ARGUMENTS_SUMMARY_MAX_CHARS:
        line = line[: ARGUMENTS_SUMMARY_MAX_CHARS - 1] + "…"
    return line


@dataclass(frozen=True)
class PendingDecision:
    """What :meth:`ChatService.precheck_approval` resolved, ready to resume.

    Carries the *requester*, not the approver: the resumed turn continues
    under the scopes and project the original request was made with (see
    ``scripts/run_turn.py`` for the same rule, reached the same way).
    """

    trace_id: str
    requester: User
    session_id: str | None
    approver: User
    events_already_seen: int
    """``len(events)`` on the paused state, as it stood at precheck time.

    The route uses this as ``TurnStream(start_seq=...)``: the client that
    is deciding this approval already saw these rows when the turn first
    paused, and re-streaming them on resume would duplicate what its trace
    panel already has.
    """


@dataclass
class ChatService:
    """Holds nothing but the process-wide resources -- everything else is a
    parameter, so one instance serves every request."""

    resources: AppResources

    def run_turn(
        self, *, actor: str, session_id: str | None, message: str, stream: TurnStream
    ) -> None:
        """Run one fresh turn to its end (or its pause), streaming as it goes.

        Never raises: every failure, including an actor the directory does
        not know, becomes an :class:`~agentic_erp_assistant.web.protocol.
        ErrorEvent` on the stream rather than an exception the caller has to
        catch on a worker thread nobody is watching.
        """
        connection = None
        try:
            user = self.resources.users.get(actor)
            trace_id = f"run-{uuid.uuid4().hex}"
            resolved_session = session_id or f"sess-{uuid.uuid4().hex[:12]}"

            connection = self.resources.connect()
            turn = build_turn(
                self.resources,
                user=user,
                session_id=resolved_session,
                trace_id=trace_id,
                connection=connection,
                stream=stream,
            )
            stream.emit(
                TurnStartedEvent(
                    trace_id=trace_id,
                    session_id=resolved_session,
                    actor=user.actor,
                    resumed=False,
                )
            )
            state = initial_state(
                user, session_id=resolved_session, trace_id=trace_id, message=message
            )
            final = turn.orchestrator.handle(state)
            stream.flush_events(final)
            self._emit_tail(final, turn, stream)
        except UnknownUser:
            stream.emit(ErrorEvent(message=f"unknown actor {actor!r}"))
        except Exception as error:  # noqa: BLE001 - the stream's own report
            logger.exception("run_turn failed for actor %r", actor)
            stream.emit(ErrorEvent(message=f"{type(error).__name__}: {error}"))
        finally:
            if connection is not None:
                connection.close()
            stream.close()

    def precheck_approval(self, *, trace_id: str, approver_actor: str) -> PendingDecision:
        """Everything that can be decided before any stream starts.

        Run this first, synchronously, and turn its three exceptions into a
        plain JSON 403/404/409 -- the approvals route's contract (D11's
        sibling rule for approvals: a caller should not have to open an SSE
        connection to be told it was refused).

        Raises:
            Forbidden: ``approver_actor`` is not in the directory, or holds
                no ``approvals.decide``.
            NotFound: No pause was ever recorded for ``trace_id``.
            Conflict: The pause exists but is not ``pending`` anymore.
        """
        try:
            approver = self.resources.users.get(approver_actor)
        except UnknownUser:
            raise Forbidden(f"unknown actor {approver_actor!r}") from None
        if not approver.can_approve:
            raise Forbidden(
                f"actor {approver_actor!r} does not hold 'approvals.decide'"
            )

        connection = self.resources.connect()
        try:
            row = connection.execute(
                "SELECT status FROM pauses WHERE trace_id = %s", (trace_id,)
            ).fetchone()
            if row is None:
                raise NotFound(f"no pause was ever recorded for {trace_id!r}")
            status = row[0]
            if status != "pending":
                raise Conflict(
                    f"the pause for {trace_id!r} is {status!r}, not 'pending' -- "
                    f"a decision is recorded once"
                )

            pending = PostgresPauseStore(connection).pending(trace_id)
            assert pending is not None  # the status check above just proved it
            requester = self.resources.users.get(pending.actor)
            return PendingDecision(
                trace_id=trace_id,
                requester=requester,
                session_id=pending.session_id,
                approver=approver,
                events_already_seen=len(pending.events),
            )
        finally:
            connection.close()

    def resume_turn(
        self, pending: PendingDecision, *, approved: bool, stream: TurnStream
    ) -> None:
        """Resume the pause :meth:`precheck_approval` already validated.

        Assumes the precheck passed; still tolerates the pause having been
        claimed by someone else in the gap between that check and this call
        (:class:`~agentic_erp_assistant.engine.orchestrator.
        ApprovalAlreadySettled`) -- rare, and by the time it is possible
        here the stream has already started, so it is reported as an
        :class:`~agentic_erp_assistant.web.protocol.ErrorEvent` rather than
        the clean 409 the precheck gives the common case.
        """
        connection = None
        try:
            connection = self.resources.connect()
            turn = build_turn(
                self.resources,
                user=pending.requester,
                session_id=pending.session_id or "",
                trace_id=pending.trace_id,
                connection=connection,
                stream=stream,
            )
            stream.emit(
                TurnStartedEvent(
                    trace_id=pending.trace_id,
                    session_id=pending.session_id,
                    actor=pending.requester.actor,
                    resumed=True,
                )
            )
            final = turn.orchestrator.resume(
                pending.trace_id, approved=approved, decided_by=pending.approver.actor
            )
            stream.flush_events(final)
            self._emit_tail(final, turn, stream)
        except ApprovalAlreadySettled as error:
            stream.emit(ErrorEvent(message=str(error)))
        except Exception as error:  # noqa: BLE001 - the stream's own report
            logger.exception("resume_turn failed for trace %r", pending.trace_id)
            stream.emit(ErrorEvent(message=f"{type(error).__name__}: {error}"))
        finally:
            if connection is not None:
                connection.close()
            stream.close()

    # -- shared tail: the run ended, one way or another ---------------------

    def _emit_tail(self, final: AgentState, turn: TurnPorts, stream: TurnStream) -> None:
        if is_paused(final):
            stream.emit(
                ApprovalRequiredEvent(
                    trace_id=final.trace_id,
                    tool_name=final.tool_name or "",
                    arguments=dict(final.tool_arguments or {}),
                    summary=_summarize_arguments(
                        final.tool_name or "", dict(final.tool_arguments or {})
                    ),
                    actor=final.actor,
                )
            )
        else:
            text, citations = parse_citations(final.response, final)
            stream.emit(
                AnswerEvent(
                    text=text or "",
                    route=final.route,
                    failure=final.failure,
                    error_detail=final.error_detail,
                    citations=citations,
                )
            )

        records, totals = turn.queries.model_calls(final.trace_id)
        run_row = turn.queries.run(final.trace_id)
        # The run row this call just filed (handle()/resume() already called
        # traces.save_run before returning) -- see the trace store's own
        # contract that an unsaved run "did not happen"; by the time this
        # tail runs, it always has.
        assert run_row is not None
        audit_rows = turn.queries.memory_audit(final.trace_id)

        stream.emit(
            TurnFinishedEvent(
                outcome="paused" if is_paused(final) else "terminal",
                route=final.route,
                failure=final.failure,
                step_count=final.step_count,
                evidence=tuple(dict.fromkeys(s.source_id for s in final.evidence)),
                memories_recalled=len(final.memories),
                history_shown=len(final.history),
                observations=tuple(
                    ObservationOut(tool=o.tool_name, status=o.status, attempts=o.attempts)
                    for o in final.observations
                ),
                model_calls=ModelCallTotalsOut(
                    count=totals.count,
                    input_tokens=totals.input_tokens,
                    output_tokens=totals.output_tokens,
                    cost_usd=totals.cost_usd,
                    unpriced=totals.unpriced,
                    records=tuple(
                        ModelCallOut(
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
                        )
                        for record in records
                    ),
                ),
                memory_audit=tuple(
                    MemoryAuditOut(
                        occurred_at=row.occurred_at,
                        memory_id=row.memory_id,
                        kind=row.kind,
                        decision=row.decision,
                        rejection=row.rejection,
                        reason=row.reason,
                        statement_summary=row.statement_summary,
                    )
                    for row in audit_rows
                ),
                started_at=run_row.started_at,
                finished_at=run_row.finished_at,
            )
        )
