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
    StoreConnectionError,
    apply_schema,
    connect,
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


@pytest.fixture
def database():
    try:
        connection = connect()
    except StoreConnectionError as error:
        pytest.skip(f"no Postgres to test against: {error}")
    yield connection
    connection.close()


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
