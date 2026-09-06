"""The evidence schema: five tables, as hand-written DDL.

Design rules, so a reviewer can argue with each of them:

* **Plain SQL, no ORM, no migration framework.** The repo's rule is that a
  plain SDK is enough wherever one is enough, and five tables with no join
  inheritance are well under that line. The cost -- schema changes are made
  here and re-applied by ``scripts/init_postgres.py`` -- buys a schema a
  reviewer reads in one screen, the same trade
  :mod:`agentic_erp_assistant.engine.transitions` makes with its table.

* **``CREATE TABLE IF NOT EXISTS``, applied by an explicit script.** Nothing
  migrates on connect: a schema that creates itself on first request is a
  schema that creates itself under load, and an init run is a deliberate act
  an operator can see. When a change is incompatible with stored data, that
  is a decision to record in an ADR and apply by hand, not something a
  library guesses at.

* **CHECK constraints derive from the typed vocabulary, not from a second
  spelling of it.** The ``IN`` lists below are built with ``get_args`` from
  the Literals the state models already close: a value added to
  :data:`~agentic_erp_assistant.state.events.EventKind` lands in the database
  constraint without anyone remembering to copy it, and a report grouping by
  these columns can never meet a category nobody declared. The mirror risk of
  a hand-maintained SQL list -- the Literal grows, the CHECK quietly rejects
  what the type system accepts -- is exactly the drift this removes.

* **JSONB for the state, columns for everything a report filters on.** The
  whole final :class:`~agentic_erp_assistant.state.agent_state.AgentState` is
  one opaque document, revalidated through the model on read, so what loads
  back is what was saved or a loud refusal -- never a silently half-parsed
  state. But ``outcome``, ``actor``, ``kind``, ``trace_id`` are real columns,
  because a reviewer groups and filters by them, and grouping by a jsonb
  path is a query nobody wants to write or review.
"""

from collections.abc import Sequence
from typing import get_args

from agentic_erp_assistant.llm.telemetry import Outcome as ModelCallOutcome
from agentic_erp_assistant.state.agent_state import ApprovalDecision
from agentic_erp_assistant.state.events import EventKind
from agentic_erp_assistant.state.tool_outcome import ToolStatus

__all__ = [
    "APPROVALS",
    "EVENT_KINDS",
    "MODEL_CALL_OUTCOMES",
    "RUN_OUTCOMES",
    "SCHEMA_STATEMENTS",
    "TOOL_STATUSES",
    "apply_schema",
    "tables_in",
]


def _sql_list(values: Sequence[str]) -> str:
    """A CHECK constraint's IN list, built from the closed set it mirrors."""
    return ", ".join(f"'{value}'" for value in values)


APPROVALS: tuple[str, ...] = get_args(ApprovalDecision)
EVENT_KINDS: tuple[str, ...] = get_args(EventKind)
MODEL_CALL_OUTCOMES: tuple[str, ...] = get_args(ModelCallOutcome)
RUN_OUTCOMES: tuple[str, ...] = ("terminal", "paused")
"""The :data:`~agentic_erp_assistant.trace.records.RunOutcome` set. Spelled
here rather than imported because the records module is the one place that
names it; this is the database's own copy of the same two words."""
TOOL_STATUSES: tuple[str, ...] = get_args(ToolStatus)

_SCHEMA_TEMPLATE = f"""
CREATE TABLE IF NOT EXISTS runs (
    trace_id text PRIMARY KEY,
    actor text NOT NULL,
    request text NOT NULL,
    outcome text NOT NULL
        CHECK (outcome IN ({_sql_list(RUN_OUTCOMES)})),
    started_at timestamptz NOT NULL,
    finished_at timestamptz NOT NULL,
    state jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS trace_events (
    trace_id text NOT NULL REFERENCES runs (trace_id) ON DELETE CASCADE,
    seq integer NOT NULL,
    node text NOT NULL,
    kind text NOT NULL
        CHECK (kind IN ({_sql_list(EVENT_KINDS)})),
    detail text NOT NULL DEFAULT '',
    PRIMARY KEY (trace_id, seq)
);

CREATE TABLE IF NOT EXISTS audit_rows (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    occurred_at timestamptz NOT NULL,
    trace_id text NOT NULL,
    actor text NOT NULL,
    tool_name text NOT NULL,
    arguments_summary text NOT NULL,
    approval text NOT NULL
        CHECK (approval IN ({_sql_list(APPROVALS)})),
    status text NOT NULL
        CHECK (status IN ({_sql_list(TOOL_STATUSES)})),
    source_ids text[] NOT NULL DEFAULT '{{}}'
);

CREATE INDEX IF NOT EXISTS audit_rows_by_run ON audit_rows (trace_id);
CREATE INDEX IF NOT EXISTS audit_rows_by_time ON audit_rows (occurred_at);

CREATE TABLE IF NOT EXISTS model_calls (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trace_id text NOT NULL,
    model text NOT NULL,
    outcome text NOT NULL
        CHECK (outcome IN ({_sql_list(MODEL_CALL_OUTCOMES)})),
    estimated_input_tokens integer NOT NULL,
    input_tokens integer NOT NULL,
    output_tokens integer NOT NULL,
    cost_usd double precision,
    latency_seconds double precision NOT NULL,
    attempts integer NOT NULL,
    occurred_at timestamptz NOT NULL,
    detail text
);

CREATE INDEX IF NOT EXISTS model_calls_by_run ON model_calls (trace_id);

CREATE TABLE IF NOT EXISTS pauses (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trace_id text NOT NULL,
    actor text NOT NULL,
    tool_name text NOT NULL,
    arguments_summary text NOT NULL,
    state jsonb NOT NULL,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'resolved')),
    created_at timestamptz NOT NULL DEFAULT now(),
    decision text,
    decided_at timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS pauses_one_pending_per_run
    ON pauses (trace_id) WHERE status = 'pending';
"""

SCHEMA_STATEMENTS: tuple[str, ...] = tuple(
    statement.strip() for statement in _SCHEMA_TEMPLATE.split(";") if statement.strip()
)
"""One entry per statement, so an init script can report what it applied.

Split on ``;`` rather than shipped as one blob because each statement is
applied with its own cursor call and named in the output -- an operator
running ``init_postgres`` sees the list, and a failed statement names itself
instead of leaving "somewhere in the schema" behind.
"""


def tables_in(statement: str) -> Sequence[str]:
    """The table names a statement creates, for reporting.

    ``IF NOT EXISTS`` means an init run is idempotent, but the operator should
    still learn which tables the schema covers -- this turns a statement back
    into the names it would have created. The ``IF NOT EXISTS`` prefix is
    stepped over rather than filtered token by token: ``IF`` is a valid
    identifier, so a filter that trusts spelling would report a table named
    "IF".
    """
    lowered = statement.lower()
    if "create table" not in lowered:
        return ()
    remainder = statement[lowered.index("create table") + len("create table") :]
    remainder = remainder.strip()
    if remainder.upper().startswith("IF NOT EXISTS"):
        remainder = remainder[len("IF NOT EXISTS") :].strip()
    words = remainder.split("(")[0].split()
    return (words[0],) if words else ()


def apply_schema(cursor) -> list[str]:
    """Apply every statement, and return the tables that exist afterwards.

    Args:
        cursor: An open psycopg cursor. DDL in Postgres is transactional, so
            the caller can wrap this in one transaction and get all-or-nothing
            schema creation.

    Returns:
        The names of the tables present after the run, whether they were
        created now or already existed.
    """
    for statement in SCHEMA_STATEMENTS:
        cursor.execute(statement)

    cursor.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY table_name"
    )
    return [row[0] for row in cursor.fetchall()]