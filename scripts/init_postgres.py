"""Create the evidence schema in Postgres.

The deliberate act that makes the evidence store usable, run once after the
container is up and re-runnable forever after:

    docker compose up -d postgres
    uv run python scripts/init_postgres.py

The schema is ``CREATE TABLE IF NOT EXISTS`` statements applied in one
transaction, so an init run either creates everything or nothing, and a
second run against an initialized database is a no-op that says so. Nothing
migrates on connect: an init run is something an operator sees, and when a
future schema change is incompatible with stored data, that is a decision to
record in an ADR and apply deliberately -- not something a library guesses at
on first request.

``--test`` targets the *other* database (ADR 0018): the one
``tests/persistence/`` is allowed to ``TRUNCATE``, never the one a running
server writes to.

    uv run python scripts/init_postgres.py --test
    uv run pytest -m postgres
"""

import argparse
import logging
import re
import sys

import psycopg

from agentic_erp_assistant.persistence import (
    SCHEMA_STATEMENTS,
    StoreConfigurationError,
    StoreConnectionError,
    apply_schema,
    assert_test_database,
    connect,
    database_name_in,
    tables_in,
    test_url_from_environment,
    url_from_environment,
)

logger = logging.getLogger("init_postgres")

_VALID_DATABASE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=None,
        help="the Postgres conninfo to connect to (default: POSTGRES_URL, "
        "or POSTGRES_TEST_URL with --test; the docker-compose default of "
        "either when unset)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="target the _test database (ADR 0018) instead of the dev one, "
        "creating it first if it does not exist yet",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="log the SQL as it is applied"
    )
    return parser.parse_args(argv)


def _ensure_test_database_exists(url: str) -> None:
    """``CREATE DATABASE`` for ``url``'s database, unless it already exists.

    Connects to the *dev* database first -- the same compose user owns both,
    and Postgres has no "create if not exists" for a database, only a check
    against ``pg_database`` followed by an unconditional ``CREATE DATABASE``,
    which cannot run inside a transaction block (autocommit, per
    :func:`~agentic_erp_assistant.persistence.connect`, is what allows it to
    run at all here).
    """
    name = database_name_in(url)
    if not _VALID_DATABASE_NAME.match(name):
        raise StoreConfigurationError(
            f"{name!r} is not a valid Postgres database name -- refusing to "
            f"issue CREATE DATABASE for it"
        )
    admin_connection = connect(url_from_environment())
    try:
        with admin_connection:
            exists = admin_connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
            ).fetchone()
            if exists:
                logger.info("database %s already exists", name)
                return
            admin_connection.execute(f'CREATE DATABASE "{name}"')
            logger.info("created database %s", name)
    except psycopg.Error as error:
        raise StoreConnectionError(
            f"could not create test database {name!r} via "
            f"{url_from_environment()!r}: {error}"
        ) from error


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if arguments.test:
        url = arguments.url or test_url_from_environment()
        try:
            assert_test_database(url)
            _ensure_test_database_exists(url)
        except (StoreConfigurationError, StoreConnectionError) as error:
            logger.error("%s", error)
            return 2
    else:
        url = arguments.url

    try:
        connection = connect(url)
    except StoreConnectionError as error:
        # Already phrased for an operator by the connection module, so it is
        # reported as it stands rather than wrapped in a traceback.
        logger.error("%s", error)
        return 2

    with connection, connection.cursor() as cursor:
        with connection.transaction():
            tables = apply_schema(cursor)

    for statement in SCHEMA_STATEMENTS:
        created = tables_in(statement)
        if created:
            logger.info("table %s present", ", ".join(created))

    logger.info(
        "evidence store ready: %d table(s) -- %s",
        len(tables),
        ", ".join(tables),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())