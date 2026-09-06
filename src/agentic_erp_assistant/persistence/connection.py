"""Opening the connection to the evidence store.

This module is one of the two places in the package that may name psycopg at
all -- the adapters take a connection from here and never open one, the same
split :class:`~agentic_erp_assistant.rag.vector_index.QdrantVectorIndex` makes
with its client. It keeps the rule enforceable: a test that greps for
``psycopg.connect`` finds it in exactly one file.

The configuration pattern matches ``QDRANT_URL``: the URL comes from
``POSTGRES_URL``, read lazily at construction with ``load_dotenv`` so a test
process can set the variable after import, and the constructor default is the
compose file's throwaway credential -- a fresh checkout with no ``.env`` works
against ``docker compose up -d postgres`` with nothing configured.

Connection problems are one typed error, raised at connect. A database that
is unreachable is a deployment fact the caller cannot route around, and the
worst version of it is one that surfaces as a failed ``save_run`` halfway
through a turn: connect once, at composition, where the failure means
"start the container" -- not mid-request, where it means an approved write
that never happened.
"""

import logging
import os

import psycopg
from dotenv import load_dotenv

__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_POSTGRES_URL",
    "StoreConnectionError",
    "connect",
    "url_from_environment",
]

logger = logging.getLogger(__name__)

DEFAULT_POSTGRES_URL = "postgresql://agentic_erp:agentic_erp@localhost:5432/agentic_erp"
"""The docker-compose default. Deliberately not secret: the compose file uses
the same throwaway credential, so the two agree by construction and a fresh
checkout needs no ``.env``. A real deployment replaces the whole URL through
``POSTGRES_URL`` and these values never leave the compose file."""

CONNECT_TIMEOUT_SECONDS = 3
"""How long one connect attempt may hang before the answer is "no database".

libpq's own default is long, and the price is paid in the wrong place: the
integration tests ask this question once per test to decide whether to skip,
so with the container down a default-timeout suite spends minutes waiting for
a refusal that takes three seconds to be sure of."""


class StoreConnectionError(RuntimeError):
    """The evidence store cannot be reached or would not accept the URL.

    Raised at connect, never mid-request: by the time a store method runs, the
    caller has already proven the database is there, so a failure inside a
    save is a bug and a failure here is "is the container up?" -- and only one
    of those belongs in an operator's hands.
    """


def url_from_environment() -> str:
    """``POSTGRES_URL``, or the compose default when blank or unset.

    Reads ``.env`` lazily rather than at import, so the value observed is the
    one the running process -- a test, a script, a server -- has by the time
    it connects, not the one that happened to be on disk at import.
    """
    load_dotenv(override=False)
    url = (os.environ.get("POSTGRES_URL") or "").strip()
    return url or DEFAULT_POSTGRES_URL


def connect(url: str | None = None) -> psycopg.Connection:
    """Open one connection, or raise one typed error.

    Args:
        url: A libpq conninfo string. ``None`` reads ``POSTGRES_URL``, falling
            back to the compose default -- so ``connect()`` with no arguments is
            the normal case, and an explicit URL is what a test or an operator
            passes to point somewhere else.

    Returns:
        An open psycopg connection, autocommit **on**: every statement stands
        on its own unless a method wraps several in an explicit
        ``with connection.transaction()`` block -- which is what makes the
        batch save of a run atomic and the per-row inserts of an audit log
        immediate, without every adapter remembering to commit.

    Raises:
        StoreConnectionError: The URL is malformed, the server is unreachable,
            or it refused the credentials.
    """
    resolved = url if url is not None else url_from_environment()
    try:
        connection = psycopg.connect(
            resolved,
            autocommit=True,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
        )
    except psycopg.Error as error:
        raise StoreConnectionError(
            f"cannot connect to the evidence store at {resolved!r}: {error}. "
            f"Is Postgres running? Start it with `docker compose up -d "
            f"postgres`, or set POSTGRES_URL to point at the one you mean."
        ) from error
    logger.info("connected to the evidence store at %s", resolved)
    return connection