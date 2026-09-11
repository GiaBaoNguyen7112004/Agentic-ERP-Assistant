"""The read model a screen needs, as plain SQL over the evidence tables.

Deliberately not new methods on :mod:`agentic_erp_assistant.trace.ports` or
:mod:`agentic_erp_assistant.memory.store`. Those ports are write-side
contracts the orchestrator depends on -- their shape is fixed by what a turn
needs to file, not by what a screen wants to list -- and a listing query
(every pending approval, every session for an actor, the running totals on a
trace's model calls) is a screen's need. Widening a port to serve it would
mean every fake standing in for that port in an engine test grows a method
the engine never calls. This class exists so that widening never has to
happen: it reads the same tables, through its own connection, and nothing
downstream of the orchestrator imports it.

Every row is revalidated through the same typed model the write path
constructs it from (:class:`~agentic_erp_assistant.state.agent_state.AgentState`,
:class:`~agentic_erp_assistant.state.events.TraceEvent`, and so on) rather
than handed back as a raw dict -- a row written by an older shape of a model
fails loudly here too, the same reason
:meth:`~agentic_erp_assistant.persistence.postgres_trace.PostgresTraceStore.load_run`
revalidates.
"""

from dataclasses import dataclass
from datetime import datetime

import psycopg
from psycopg.rows import dict_row

from agentic_erp_assistant.llm.telemetry import ModelCallRecord
from agentic_erp_assistant.memory.audit import MemoryAuditRow
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.tools.models import AuditRow

__all__ = [
    "EvidenceQueries",
    "ModelCallTotals",
    "PendingApproval",
    "RunRow",
    "SessionSummary",
]


@dataclass(frozen=True)
class PendingApproval:
    """One row of the approval queue -- everything an approver's card shows."""

    trace_id: str
    actor: str
    tool_name: str
    arguments_summary: str
    created_at: datetime
    session_id: str | None
    """``None`` when the run has no session -- a one-shot script's pause has
    nothing for the window to have recorded yet."""


@dataclass(frozen=True)
class RunRow:
    """A run's own columns, plus its state -- revalidated, not trusted."""

    trace_id: str
    actor: str
    project_code: str | None
    """``None`` only for a run saved before ``STATE_VERSION`` 2 and never
    resaved since; every run filed under the current build carries one."""

    outcome: str
    started_at: datetime
    finished_at: datetime
    state: AgentState


@dataclass(frozen=True)
class ModelCallTotals:
    """The running cost of one trace's model calls, summed once so a screen
    does not re-derive it from the rows."""

    count: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    """The summed cost, or ``None`` when any call in the trace has no
    reviewed rate -- a partial total that silently omitted a call would read
    as complete when it is not. See :data:`unpriced`."""

    unpriced: int
    """How many calls in the trace have no reviewed rate."""


@dataclass(frozen=True)
class SessionSummary:
    """One session, as a list of them is shown: what it opened with, and
    when it was last touched."""

    session_id: str
    first_request: str
    last_started_at: datetime


@dataclass(frozen=True)
class EvidenceQueries:
    """Read-only queries over one connection, for one request.

    Built the same way every per-turn port is (see
    :mod:`agentic_erp_assistant.composition.turn`): one connection per
    request, never shared, because a connection is not safe for concurrent
    use.
    """

    connection: psycopg.Connection

    def pending_approvals(self, actor: str | None = None) -> tuple[PendingApproval, ...]:
        """Every pause still waiting on a human, oldest first.

        Joined to ``session_turns`` for the session id a queue card links
        back to -- a left join, because a pending run that never had a
        session (a one-shot script) still belongs in the queue.
        """
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT p.trace_id, p.actor, p.tool_name, p.arguments_summary,
                       p.created_at, st.session_id
                FROM pauses p
                LEFT JOIN session_turns st ON st.trace_id = p.trace_id
                WHERE p.status = 'pending'
                  AND (%(actor)s::text IS NULL OR p.actor = %(actor)s)
                ORDER BY p.created_at
                """,
                {"actor": actor},
            )
            rows = cursor.fetchall()
        return tuple(
            PendingApproval(
                trace_id=row["trace_id"],
                actor=row["actor"],
                tool_name=row["tool_name"],
                arguments_summary=row["arguments_summary"],
                created_at=row["created_at"],
                session_id=row["session_id"],
            )
            for row in rows
        )

    def run(self, trace_id: str) -> RunRow | None:
        """One run's columns and its revalidated state, or ``None``."""
        row = self.connection.execute(
            "SELECT trace_id, actor, project_code, outcome, started_at, "
            "finished_at, state FROM runs WHERE trace_id = %s",
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        trace_id_, actor, project_code, outcome, started_at, finished_at, state = row
        return RunRow(
            trace_id=trace_id_,
            actor=actor,
            project_code=project_code,
            outcome=outcome,
            started_at=started_at,
            finished_at=finished_at,
            state=AgentState.model_validate(state),
        )

    def events(self, trace_id: str) -> tuple[TraceEvent, ...]:
        """The trace, in order."""
        rows = self.connection.execute(
            "SELECT node, kind, detail FROM trace_events "
            "WHERE trace_id = %s ORDER BY seq",
            (trace_id,),
        ).fetchall()
        return tuple(
            TraceEvent(node=node, kind=kind, detail=detail)
            for node, kind, detail in rows
        )

    def audit_rows(self, trace_id: str) -> tuple[AuditRow, ...]:
        """Every gated call this run made, in the order they were recorded."""
        rows = self.connection.execute(
            "SELECT trace_id, occurred_at, actor, tool_name, arguments_summary, "
            "approval, status, source_ids FROM audit_rows "
            "WHERE trace_id = %s ORDER BY id",
            (trace_id,),
        ).fetchall()
        return tuple(
            AuditRow(
                trace_id=row[0],
                occurred_at=row[1],
                actor=row[2],
                tool_name=row[3],
                arguments_summary=row[4],
                approval=row[5],
                status=row[6],
                source_ids=tuple(row[7] or ()),
            )
            for row in rows
        )

    def model_calls(self, trace_id: str) -> tuple[tuple[ModelCallRecord, ...], ModelCallTotals]:
        """Every model call this run made, and the running totals over them."""
        rows = self.connection.execute(
            "SELECT model, outcome, estimated_input_tokens, input_tokens, "
            "output_tokens, cost_usd, latency_seconds, attempts, occurred_at, "
            "detail FROM model_calls WHERE trace_id = %s ORDER BY id",
            (trace_id,),
        ).fetchall()
        records = tuple(
            ModelCallRecord(
                model=row[0],
                outcome=row[1],
                estimated_input_tokens=row[2],
                input_tokens=row[3],
                output_tokens=row[4],
                cost_usd=row[5],
                latency_seconds=row[6],
                attempts=row[7],
                occurred_at=row[8],
                detail=row[9],
            )
            for row in rows
        )
        unpriced = sum(1 for record in records if record.cost_usd is None)
        totals = ModelCallTotals(
            count=len(records),
            input_tokens=sum(record.input_tokens for record in records),
            output_tokens=sum(record.output_tokens for record in records),
            cost_usd=(
                sum(record.cost_usd for record in records if record.cost_usd is not None)
                if unpriced == 0
                else None
            ),
            unpriced=unpriced,
        )
        return records, totals

    def memory_audit(self, trace_id: str) -> tuple[MemoryAuditRow, ...]:
        """Every memory decision made while filing this run."""
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT occurred_at, trace_id, session_id, project_code, actor, "
                "memory_id, kind, decision, rejection, reason, statement_summary "
                "FROM memory_audit WHERE trace_id = %s ORDER BY id",
                (trace_id,),
            )
            rows = cursor.fetchall()
        return tuple(MemoryAuditRow.model_validate(row) for row in rows)

    def memories(self, *, actor: str, project_code: str, session_id: str) -> tuple[MemoryRecord, ...]:
        """Every live memory in this scope, most recently recorded first."""
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT memory_id, kind, key, statement, project_code, "
                "required_scope, actor, session_id, recorded_in_run, "
                "recorded_at, confidence, supersedes, superseded_at, links "
                "FROM memories "
                "WHERE actor = %s AND project_code = %s AND session_id = %s "
                "AND superseded_at IS NULL "
                "ORDER BY recorded_at DESC",
                (actor, project_code, session_id),
            )
            rows = cursor.fetchall()
        return tuple(MemoryRecord.model_validate(row) for row in rows)

    def sessions(self, actor: str) -> tuple[SessionSummary, ...]:
        """Every session this actor has, most recently active first."""
        rows = self.connection.execute(
            """
            SELECT session_id,
                   (array_agg(request ORDER BY started_at ASC))[1] AS first_request,
                   max(started_at) AS last_started_at
            FROM session_turns
            WHERE actor = %s
            GROUP BY session_id
            ORDER BY last_started_at DESC
            """,
            (actor,),
        ).fetchall()
        return tuple(
            SessionSummary(session_id=row[0], first_request=row[1], last_started_at=row[2])
            for row in rows
        )

    def session_turns(self, session_id: str, actor: str) -> tuple[ConversationTurn, ...]:
        """One session's turns, oldest first -- what a chat history replays."""
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT trace_id, session_id, actor, request, response, route, "
                "failure, tool_name, approval, started_at, finished_at "
                "FROM session_turns "
                "WHERE session_id = %s AND actor = %s "
                "ORDER BY started_at",
                (session_id, actor),
            )
            rows = cursor.fetchall()
        return tuple(ConversationTurn.model_validate(row) for row in rows)
