"""The schema itself: what it declares, and what re-applying it survives.

The pure-Python tests need no database -- they check the statements
themselves. The one Postgres-marked test proves the migration hazard the
trace_events.kind constraint used to carry: naming it and re-applying it on
every init run is what makes a future EventKind growing safe on a database
that already exists, rather than a permanent split between databases
initialised before the growth and after it.
"""

from datetime import UTC, datetime

import pytest

from agentic_erp_assistant.persistence import (
    SCHEMA_STATEMENTS,
    apply_schema,
    tables_in,
)


def all_tables() -> list[str]:
    return [name for statement in SCHEMA_STATEMENTS for name in tables_in(statement)]


def test_session_turns_is_one_of_the_declared_tables() -> None:
    assert "session_turns" in all_tables()


def test_the_schema_declares_nine_tables() -> None:
    assert len(all_tables()) == 9


# --------------------------------------------------------------------------
# The constraint re-application survives a database that already exists
# --------------------------------------------------------------------------
# `database` (module-scoped, already applies the schema once) comes from
# tests/persistence/conftest.py.


@pytest.mark.postgres
def test_apply_schema_twice_still_accepts_every_declared_event_kind(database) -> None:
    """Simulates a database initialised before an EventKind grew: applying the
    schema a second time must not leave the old CHECK in place rejecting a
    kind the current build declares."""
    with database.cursor() as cursor:
        with database.transaction():
            apply_schema(cursor)

    when = datetime(2026, 9, 8, tzinfo=UTC)
    database.execute("TRUNCATE runs CASCADE")
    database.execute(
        "INSERT INTO runs (trace_id, actor, request, outcome, started_at, "
        "finished_at, state) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        ("run-schema-test", "bao", "hi", "terminal", when, when, "{}"),
    )
    database.execute(
        "INSERT INTO trace_events (trace_id, seq, node, kind, detail) "
        "VALUES (%s, 1, %s, %s, %s)",
        ("run-schema-test", "history", "history_recalled", "1 prior turn(s)"),
    )

    with database.cursor() as cursor:
        with database.transaction():
            apply_schema(cursor)

    # The constraint was just re-applied; a kind the build declares must
    # still be insertable, and the row inserted before re-applying it must
    # still be there -- re-applying is not the same as recreating.
    database.execute(
        "INSERT INTO trace_events (trace_id, seq, node, kind, detail) "
        "VALUES (%s, 2, %s, %s, %s)",
        ("run-schema-test", "history", "history_promoted", "1 turn(s) folded"),
    )
    rows = database.execute(
        "SELECT kind FROM trace_events WHERE trace_id = %s ORDER BY seq",
        ("run-schema-test",),
    ).fetchall()
    assert [row[0] for row in rows] == ["history_recalled", "history_promoted"]


@pytest.mark.postgres
def test_apply_schema_twice_still_accepts_every_declared_failure_mode(database) -> None:
    """The same migration hazard as the kind constraint above, for
    ``session_turns.failure`` -- named and re-applied for the same reason:
    FailureMode grew (``planner_loop``, ADR 0019) after this column shipped,
    and a database initialised before that growth must not be left with a
    CHECK that still rejects it."""
    with database.cursor() as cursor:
        with database.transaction():
            apply_schema(cursor)

    when = datetime(2026, 9, 12, tzinfo=UTC)
    database.execute("TRUNCATE session_turns CASCADE")
    database.execute(
        "INSERT INTO session_turns (trace_id, session_id, actor, request, "
        "response, route, failure, tool_name, approval, started_at, "
        "finished_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            "run-schema-test-failure",
            "session-schema-test",
            "bao",
            "hi",
            None,
            "think",
            "none",
            None,
            "not_required",
            when,
            when,
        ),
    )

    with database.cursor() as cursor:
        with database.transaction():
            apply_schema(cursor)

    # The constraint was just re-applied; a failure mode added since this
    # column shipped must still be insertable, and the row inserted before
    # re-applying it must still be there.
    database.execute(
        "INSERT INTO session_turns (trace_id, session_id, actor, request, "
        "response, route, failure, tool_name, approval, started_at, "
        "finished_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            "run-schema-test-planner-loop",
            "session-schema-test",
            "bao",
            "hi again",
            None,
            "fail",
            "planner_loop",
            "list_risks",
            "not_required",
            when,
            when,
        ),
    )
    rows = database.execute(
        "SELECT failure FROM session_turns WHERE session_id = %s ORDER BY trace_id",
        ("session-schema-test",),
    ).fetchall()
    assert {row[0] for row in rows} == {"none", "planner_loop"}


@pytest.mark.postgres
def test_apply_schema_twice_still_accepts_every_declared_rejection(database) -> None:
    """The same migration hazard as the two constraints above, for
    ``memory_audit.rejection``: it grew (``not_established``, the memory
    refactor) on a database that already existed, and a database initialised
    before that growth must not be left with a CHECK that still rejects it --
    consolidation logs an audit-insert failure rather than raising, which is
    exactly what would let the refusal go unnoticed."""
    with database.cursor() as cursor:
        with database.transaction():
            apply_schema(cursor)

    when = datetime(2026, 9, 14, tzinfo=UTC)
    database.execute("TRUNCATE memory_audit CASCADE")
    database.execute(
        "INSERT INTO memory_audit (occurred_at, trace_id, session_id, "
        "project_code, actor, memory_id, kind, decision, rejection, reason, "
        "statement_summary) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            when, "run-schema-test-rejection", "session-schema-test", "atlas",
            "bao", "mem-schema-test", "fact", "reject", "not_established",
            "the turn established nothing", "absence claim",
        ),
    )

    with database.cursor() as cursor:
        with database.transaction():
            apply_schema(cursor)

    # The constraint was just re-applied; a rejection added since this column
    # shipped must still be insertable, and the row inserted before
    # re-applying it must still be there.
    database.execute(
        "INSERT INTO memory_audit (occurred_at, trace_id, session_id, "
        "project_code, actor, memory_id, kind, decision, rejection, reason, "
        "statement_summary) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            when, "run-schema-test-rejection", "session-schema-test", "atlas",
            "bao", "mem-schema-test-2", "fact", "reject", "not_established",
            "the turn established nothing again", "another absence claim",
        ),
    )
    rows = database.execute(
        "SELECT memory_id FROM memory_audit WHERE session_id = %s "
        "ORDER BY memory_id",
        ("session-schema-test",),
    ).fetchall()
    assert [row[0] for row in rows] == ["mem-schema-test", "mem-schema-test-2"]
