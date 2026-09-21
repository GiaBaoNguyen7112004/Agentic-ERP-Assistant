"""The evidence schema: nine tables, as hand-written DDL.

Design rules, so a reviewer can argue with each of them:

* **Plain SQL, no ORM, no migration framework.** The repo's rule is that a
  plain SDK is enough wherever one is enough, and nine tables with no join
  inheritance are well under that line. The cost -- schema changes are made
  here and re-applied by ``scripts/init_postgres.py`` -- buys a schema a
  reviewer reads in one screen, the same trade
  :mod:`agentic_erp_assistant.engine.transitions` makes with its table.

* **Invariants that must hold across processes are database constraints.**
  Two of them: ``intents_one_open_per_session``, so a session cannot end up
  with two tasks in flight and recall never has to choose between them; and
  ``memory_audit_rejection_matches_decision``, so a row cannot claim it stored
  something while naming the rule that refused it. Both are already enforced
  in Python, and both are here as well because the Python check protects one
  process and the index protects the database. The pause table's partial
  unique index is the same move for the same reason.

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
from agentic_erp_assistant.memory.intent import IntentStatus
from agentic_erp_assistant.memory.models import (
    MemoryDecisionKind,
    RejectionReason,
)
from agentic_erp_assistant.reasoning.decision import DecisionRoute, FailureMode
from agentic_erp_assistant.state.agent_state import ApprovalDecision
from agentic_erp_assistant.state.events import EventKind
from agentic_erp_assistant.state.memory import MemoryKind
from agentic_erp_assistant.state.tool_outcome import ToolStatus

__all__ = [
    "APPROVALS",
    "DECISION_ROUTES",
    "EVENT_KINDS",
    "FAILURE_MODES",
    "INTENT_STATUSES",
    "MEMORY_DECISIONS",
    "MEMORY_KINDS",
    "MODEL_CALL_OUTCOMES",
    "REJECTION_REASONS",
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
MEMORY_KINDS: tuple[str, ...] = get_args(MemoryKind)
MEMORY_DECISIONS: tuple[str, ...] = get_args(MemoryDecisionKind)
REJECTION_REASONS: tuple[str, ...] = get_args(RejectionReason)
INTENT_STATUSES: tuple[str, ...] = get_args(IntentStatus)
DECISION_ROUTES: tuple[str, ...] = get_args(DecisionRoute)
FAILURE_MODES: tuple[str, ...] = get_args(FailureMode)

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

-- Added after the table first shipped, alongside AgentState.project_code
-- (STATE_VERSION 2). Idempotent, like the pauses.decided_by column above.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS project_code text;

CREATE TABLE IF NOT EXISTS trace_events (
    trace_id text NOT NULL REFERENCES runs (trace_id) ON DELETE CASCADE,
    seq integer NOT NULL,
    node text NOT NULL,
    kind text NOT NULL
        CONSTRAINT trace_events_kind_check CHECK (kind IN ({_sql_list(EVENT_KINDS)})),
    detail text NOT NULL DEFAULT '',
    PRIMARY KEY (trace_id, seq)
);

-- EventKind is a closed set that has grown before (history_recalled,
-- history_promoted) and will again. A guarded table creation never touches
-- an existing constraint, so on a database initialised before a growth the
-- old CHECK would silently keep rejecting the new members forever -- and
-- trace_events.save_run is the one store call an unsaved run cannot survive.
-- Naming the constraint above and re-applying it here on every init run is
-- what makes that growth a schema update instead of a silent split between
-- "databases initialised before" and "after".
ALTER TABLE trace_events DROP CONSTRAINT IF EXISTS trace_events_kind_check;
ALTER TABLE trace_events ADD CONSTRAINT trace_events_kind_check
    CHECK (kind IN ({_sql_list(EVENT_KINDS)}));

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

-- Added after the table first shipped: who claimed the pause, beside what
-- they decided. Idempotent, like the constraint re-apply above, so a
-- database initialised before this column gains it on the next init run.
ALTER TABLE pauses ADD COLUMN IF NOT EXISTS decided_by text;

CREATE TABLE IF NOT EXISTS memories (
    memory_id text PRIMARY KEY,
    kind text NOT NULL
        CHECK (kind IN ({_sql_list(MEMORY_KINDS)})),
    key text NOT NULL,
    statement text NOT NULL,
    project_code text NOT NULL,
    required_scope text NOT NULL,
    actor text NOT NULL,
    session_id text NOT NULL,
    recorded_in_run text NOT NULL,
    recorded_at timestamptz NOT NULL,
    confidence double precision NOT NULL
        CHECK (confidence >= 0 AND confidence <= 1),
    supersedes text[] NOT NULL DEFAULT '{{}}',
    superseded_at timestamptz,
    links text[] NOT NULL DEFAULT '{{}}'
);

CREATE INDEX IF NOT EXISTS memories_live_by_scope
    ON memories (project_code, actor, session_id, kind)
    WHERE superseded_at IS NULL;

CREATE TABLE IF NOT EXISTS intents (
    intent_id text PRIMARY KEY,
    goal text NOT NULL,
    project_code text NOT NULL,
    required_scope text NOT NULL,
    actor text NOT NULL,
    session_id text NOT NULL,
    confirmed_slots jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    unresolved_slots text[] NOT NULL DEFAULT '{{}}',
    status text NOT NULL
        CHECK (status IN ({_sql_list(INTENT_STATUSES)})),
    opened_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    closed_at timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS intents_one_open_per_session
    ON intents (project_code, session_id) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS memory_audit (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    occurred_at timestamptz NOT NULL,
    trace_id text NOT NULL,
    session_id text NOT NULL,
    project_code text NOT NULL,
    actor text NOT NULL,
    memory_id text NOT NULL,
    kind text NOT NULL
        CHECK (kind IN ({_sql_list(MEMORY_KINDS)})),
    decision text NOT NULL
        CHECK (decision IN ({_sql_list(MEMORY_DECISIONS)})),
    rejection text
        CHECK (rejection IS NULL OR rejection IN ({_sql_list(REJECTION_REASONS)})),
    reason text NOT NULL DEFAULT '',
    statement_summary text NOT NULL,
    CONSTRAINT memory_audit_rejection_matches_decision
        CHECK ((decision = 'reject') = (rejection IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS memory_audit_by_run ON memory_audit (trace_id);
CREATE INDEX IF NOT EXISTS memory_audit_by_session ON memory_audit (session_id);
CREATE INDEX IF NOT EXISTS memory_audit_by_decision ON memory_audit (decision);

-- RejectionReason grew on a database that already existed (not_established,
-- 2026-09, the memory refactor). Same reasoning as trace_events_kind_check
-- above: the inline CHECK names the constraint implicitly
-- (memory_audit_rejection_check), a guarded table creation never touches it,
-- and memory consolidation logs an audit-insert failure rather than raising --
-- exactly the combination that would let the old CHECK keep refusing the new
-- reason forever while every turn looked fine.
ALTER TABLE memory_audit DROP CONSTRAINT IF EXISTS memory_audit_rejection_check;
ALTER TABLE memory_audit ADD CONSTRAINT memory_audit_rejection_check
    CHECK (rejection IS NULL OR rejection IN ({_sql_list(REJECTION_REASONS)}));

CREATE TABLE IF NOT EXISTS session_turns (
    trace_id text PRIMARY KEY,
    session_id text NOT NULL,
    actor text NOT NULL,
    request text NOT NULL,
    response text,
    route text
        CONSTRAINT session_turns_route_check
        CHECK (route IS NULL OR route IN ({_sql_list(DECISION_ROUTES)})),
    failure text NOT NULL
        CONSTRAINT session_turns_failure_check
        CHECK (failure IN ({_sql_list(FAILURE_MODES)})),
    tool_name text,
    approval text NOT NULL
        CHECK (approval IN ({_sql_list(APPROVALS)})),
    started_at timestamptz NOT NULL,
    finished_at timestamptz NOT NULL,
    promoted_in_run text
);

CREATE INDEX IF NOT EXISTS session_turns_by_session
    ON session_turns (session_id, actor, started_at);

-- DecisionRoute and FailureMode are closed sets that have grown before
-- (FailureMode gained planner_loop, ADR 0019) and will again. The same
-- reasoning as trace_events_kind_check above applies here: a guarded table
-- creation never touches an existing constraint, so a database initialised
-- before a growth would silently keep rejecting the new member forever --
-- and this is the table RunOrchestrator writes to on every turn, logging
-- and eating the failure rather than surfacing it (never-fail, by design),
-- which is exactly what would let this kind of drift go unnoticed. Naming
-- both constraints and re-applying them here on every init run is what
-- makes the growth a schema update instead of a silent split.
ALTER TABLE session_turns DROP CONSTRAINT IF EXISTS session_turns_route_check;
ALTER TABLE session_turns ADD CONSTRAINT session_turns_route_check
    CHECK (route IS NULL OR route IN ({_sql_list(DECISION_ROUTES)}));
ALTER TABLE session_turns DROP CONSTRAINT IF EXISTS session_turns_failure_check;
ALTER TABLE session_turns ADD CONSTRAINT session_turns_failure_check
    CHECK (failure IN ({_sql_list(FAILURE_MODES)}));
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