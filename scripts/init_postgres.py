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
"""

import argparse
import logging
import sys

from agentic_erp_assistant.persistence import (
    SCHEMA_STATEMENTS,
    StoreConnectionError,
    apply_schema,
    connect,
    tables_in,
)

logger = logging.getLogger("init_postgres")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=None,
        help="the Postgres conninfo to connect to (default: POSTGRES_URL or "
        "the docker-compose default)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="log the SQL as it is applied"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        connection = connect(arguments.url)
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