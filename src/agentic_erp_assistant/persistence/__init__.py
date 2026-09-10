"""Postgres, and only Postgres: the durable ends of the evidence ports.

This package is to :mod:`agentic_erp_assistant.trace` and
:mod:`agentic_erp_assistant.tools.audit` what
:mod:`agentic_erp_assistant.llm.adapters` is to the provider ports: the one
directory that may name the concrete backend, so every other module keeps
talking to Protocols. ``psycopg`` is imported here and nowhere else -- the
same rule, enforced the same way, as ``httpx`` never being imported outside
the OpenAI adapter.

What lives here and why nothing else does:

* :mod:`~agentic_erp_assistant.persistence.connection` opens connections and
  turns every way opening one can fail into one typed error.
* :mod:`~agentic_erp_assistant.persistence.schema` is the nine tables as
  hand-written DDL, applied by an explicit script -- nothing migrates on
  connect.
* The adapters satisfy the ports structurally, with no import of the
  ports they satisfy -- the same move the in-memory fakes make, so a fake
  and an adapter can never disagree about whose contract they implement.

The adapters take an open connection and never open or close one, so the
composition point -- a script, the future web layer -- decides where the
database is, and a test can point one at the compose container or skip
itself when no container is up.
"""

from agentic_erp_assistant.persistence.connection import (
    DEFAULT_POSTGRES_URL,
    StoreConnectionError,
    connect,
    url_from_environment,
)
from agentic_erp_assistant.persistence.postgres_audit import PostgresAuditLog
from agentic_erp_assistant.persistence.postgres_conversation import (
    PostgresConversationStore,
)
from agentic_erp_assistant.persistence.postgres_memory import (
    PostgresMemoryAudit,
    PostgresMemoryStore,
)
from agentic_erp_assistant.persistence.postgres_pause import PostgresPauseStore
from agentic_erp_assistant.persistence.postgres_trace import PostgresTraceStore
from agentic_erp_assistant.persistence.schema import (
    SCHEMA_STATEMENTS,
    apply_schema,
    tables_in,
)

__all__ = [
    "DEFAULT_POSTGRES_URL",
    "PostgresAuditLog",
    "PostgresConversationStore",
    "PostgresMemoryAudit",
    "PostgresMemoryStore",
    "PostgresPauseStore",
    "PostgresTraceStore",
    "SCHEMA_STATEMENTS",
    "StoreConnectionError",
    "apply_schema",
    "connect",
    "tables_in",
    "url_from_environment",
]