"""A :class:`~agentic_erp_assistant.trace.ports.PauseStore` over Postgres.

The table is the point of the whole pause story: a pending approval that
survives the process means an approver can answer an hour and a restart later,
and the database -- not a variable in a web handler -- is what makes that
true. Two properties do the work:

* **One pending row per run, as a database guarantee.** A partial unique
  index on ``pauses (trace_id) WHERE status = 'pending'`` means the second
  save of a paused run is refused by the database no matter which process
  attempted it -- surfaced as :class:`~agentic_erp_assistant.trace.ports.
  PauseAlreadyPending`, the same typed error the in-memory fake raises for
  the same mistake.

* **Settling is one atomic UPDATE, and it happens before anything runs.**
  ``claim`` is ``UPDATE ... WHERE status = 'pending' RETURNING state``: the
  row flips to resolved and the caller learns the state in one statement, so
  exactly one caller in the world can win -- whichever of them the database
  lets through. The winner executes the approved write; the losers get
  ``None`` and never touch the engine. Execute-then-claim would let every
  winner run the write, and one approved write executed twice is the failure
  this shape exists to make impossible.

The claim's decision is recorded in the row it settles: ``decision`` and
``decided_at`` land in the same UPDATE, so the queue's history answers
"what did the human actually say?" without the trace ever being loaded.
"""

import logging

import psycopg
from psycopg.types.json import Jsonb

from agentic_erp_assistant.engine.workflow import is_paused
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.tools.models import ARGUMENTS_SUMMARY_MAX_CHARS
from agentic_erp_assistant.trace.ports import PauseAlreadyPending

__all__ = ["PostgresPauseStore"]

logger = logging.getLogger(__name__)

_VALUE_MAX_CHARS = 40
"""Per-value cap in the summary line. The same cap the tool gateway's
renderer uses, for the same reason: a value that can hold a payload can hold
a credential, and no summary is worth that."""


def _summarize(state: AgentState) -> str:
    """The paused call as one line an approver reads in a queue listing.

    Mirrors the tool gateway's ``_summarize`` rather than importing it: that
    function renders a :class:`~agentic_erp_assistant.state.tool_request.
    ToolRequest`, and the function that renders a paused state is this one.
    Two renderers of the same facts is the cost; the line both produce is
    presentation, not evidence -- the state's jsonb column is the record.
    """
    parts = []
    for key, value in (state.tool_arguments or {}).items():
        rendered = str(value)
        if len(rendered) > _VALUE_MAX_CHARS:
            rendered = rendered[: _VALUE_MAX_CHARS - 1] + "…"
        parts.append(f"{key}={rendered}")

    line = f"{state.tool_name or 'call'}({', '.join(parts)})"
    if len(line) > ARGUMENTS_SUMMARY_MAX_CHARS:
        line = line[: ARGUMENTS_SUMMARY_MAX_CHARS - 1] + "…"
    return line


class PostgresPauseStore:
    """Pending approvals in one table, settled exactly once.

    The paused predicate comes from the engine here, where
    :mod:`agentic_erp_assistant.trace.memory` inlines its own copy. Both are
    honest to their situation: the trace package cannot import the engine
    because the engine's orchestrator imports the trace, and this package is
    a leaf that nothing imports -- so it takes the one definition rather than
    growing a third.
    """

    def __init__(self, connection: psycopg.Connection) -> None:
        self._connection = connection

    def save(self, state: AgentState) -> None:
        """File one paused state as the run's pending decision.

        Raises:
            ValueError: The state is not waiting on anybody -- filing it
                would queue an approval nobody was asked for.
            PauseAlreadyPending: The run already has a pause waiting; the
                unique partial index is what says so, which means two
                processes cannot race past the rule from opposite sides.
        """
        if not is_paused(state):
            raise ValueError(
                f"route={state.route!r} approval={state.approval!r}: only a "
                f"paused state may be filed as waiting on a human"
            )
        try:
            self._connection.execute(
                """
                INSERT INTO pauses (trace_id, actor, tool_name, arguments_summary,
                                    state)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    state.trace_id,
                    state.actor,
                    state.tool_name or "",
                    _summarize(state),
                    Jsonb(state.model_dump(mode="json")),
                ),
            )
        except psycopg.errors.UniqueViolation as error:
            raise PauseAlreadyPending(
                f"run {state.trace_id!r} already has a pause waiting; a "
                f"second would let an approver answer a call they were never "
                f"shown"
            ) from error

    def pending(self, trace_id: str) -> AgentState | None:
        """The run's waiting state, or ``None`` when nobody is owed a decision."""
        row = self._connection.execute(
            "SELECT state FROM pauses WHERE trace_id = %s AND status = 'pending'",
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        return AgentState.model_validate(row[0])

    def claim(
        self, trace_id: str, *, approved: bool, decided_by: str | None = None
    ) -> AgentState | None:
        """Settle the pending decision and hand back the state, or ``None``.

        One statement, so atomic by construction: the row is resolved and
        the state returned to exactly the caller whose UPDATE matched. Every
        later caller -- whichever answer they brought -- finds no row and
        gets ``None``.
        """
        decision = "approved" if approved else "denied"
        row = self._connection.execute(
            """
            UPDATE pauses
            SET status = 'resolved', decision = %s, decided_at = now(),
                decided_by = %s
            WHERE trace_id = %s AND status = 'pending'
            RETURNING state
            """,
            (decision, decided_by, trace_id),
        ).fetchone()
        if row is None:
            return None
        logger.info("pause for run %s resolved as %s", trace_id, decision)
        return AgentState.model_validate(row[0])