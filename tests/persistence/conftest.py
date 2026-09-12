"""The one ``database`` fixture every module in this package shares.

Five modules used to open their own connection, each with ``connect()`` and
no URL -- which means ``POSTGRES_URL``, or the compose default, which is the
*same database a running server writes to*. Running ``uv run pytest -m
postgres`` with the dev container up therefore truncated the evidence store
(ADR 0018). This fixture is the fix: it resolves ``POSTGRES_TEST_URL``
instead, refuses to connect anywhere whose database is not named ``_test``
(:func:`~agentic_erp_assistant.persistence.assert_test_database`), and only
then opens the connection every test in this package uses.
"""

import pytest

from agentic_erp_assistant.persistence import (
    StoreConnectionError,
    apply_schema,
    assert_test_database,
    connect,
    test_url_from_environment,
)


@pytest.fixture(scope="module")
def database():
    """One connection per module, against the test database only.

    ``assert_test_database`` is not caught: a URL that fails it is a
    configuration mistake (an unset ``POSTGRES_TEST_URL``, a copy-pasted
    ``.env`` line), not a missing container, and it must fail the run rather
    than silently skip. A connection failure, by contrast, means "the test
    database has not been created yet" and is exactly what
    ``scripts/init_postgres.py --test`` is for -- so that one skips, with the
    command named in the message.
    """
    url = test_url_from_environment()
    assert_test_database(url)
    try:
        connection = connect(url)
    except StoreConnectionError as error:
        pytest.skip(
            f"no Postgres test database to test against: {error}. Create it "
            f"with `uv run python scripts/init_postgres.py --test`."
        )
    with connection.cursor() as cursor:
        with connection.transaction():
            apply_schema(cursor)
    yield connection
    connection.close()
