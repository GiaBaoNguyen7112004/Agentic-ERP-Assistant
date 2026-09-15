"""The process-wide resources: expensive to build, safe to share across turns.

Everything here is either a connection (the OpenAI HTTP client, the Qdrant
clients) or something built once by reading one back (the lexical index, the
ERP file, the tool registry's handler closures). None of it is safe to rebuild
per request -- the lexical index alone means re-reading every embedded chunk
back out of Qdrant -- and none of it holds per-turn state, which is what makes
sharing it across concurrent requests fine. :mod:`agentic_erp_assistant.
composition.turn` is where a request's own state (a connection, a trace id, an
actor) gets layered over this.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import psycopg

from agentic_erp_assistant.composition.settings import Settings
from agentic_erp_assistant.composition.users import UserDirectory
from agentic_erp_assistant.erp.mock import MockErp
from agentic_erp_assistant.llm.adapters.openai_chat import OpenAIChatClient
from agentic_erp_assistant.memory.vector_store import MemoryVectorStorePort
from agentic_erp_assistant.persistence.connection import connect as connect_postgres
from agentic_erp_assistant.rag.manifest import ManifestEntry, load_documents
from agentic_erp_assistant.rag.ports import EmbeddingsPort
from agentic_erp_assistant.rag.retriever import RetrievalService
from agentic_erp_assistant.tools.limits import InMemoryRateLimiter, RateLimiter
from agentic_erp_assistant.tools.registry import ToolRegistry, build_default_registry

__all__ = ["AppResources"]

logger = logging.getLogger(__name__)


class ResourcesError(RuntimeError):
    """A process-wide resource could not be built.

    Raised at startup, never mid-request: by the time a turn runs, every
    resource in :class:`AppResources` has already proven it exists, so a
    failure here means "start the container" or "run the ingest script" --
    an operator's problem, not a turn's.
    """


@dataclass
class AppResources:
    """Everything one process needs, built once and handed to every turn.

    Not frozen: :meth:`close` is the one place anything here changes, and it
    runs exactly once, at shutdown.
    """

    settings: Settings
    users: UserDirectory
    chat_client: OpenAIChatClient
    """Satisfies both :class:`~agentic_erp_assistant.llm.ports.
    LargeLanguageModelClient` and :class:`~agentic_erp_assistant.llm.ports.
    ToolCallingClient` -- one HTTP client for the answering and the routing
    call alike, since both are the same provider connection."""

    embeddings: EmbeddingsPort
    retrieval: RetrievalService
    """Holds the vector store, the embeddings client, and the lexical index
    built once by reading the vector store back (see
    :meth:`~agentic_erp_assistant.rag.retriever.RetrievalService.load`)."""

    memory_index: MemoryVectorStorePort
    erp: MockErp
    registry: ToolRegistry
    """Every tool, bound to :attr:`erp` and to the read budget
    :attr:`~agentic_erp_assistant.composition.settings.Settings.
    dev_tool_rate_limit` may have tightened."""

    limiter: RateLimiter
    """Shared across every turn's :class:`~agentic_erp_assistant.tools.
    gateway.ToolGateway` -- the budget is per actor, per tool, per process,
    not per request."""

    manifest: Mapping[str, ManifestEntry]
    """Every document's manifest row, by ``document_id`` -- what
    ``/api/documents/{id}`` checks a request against before serving a file."""

    connect: Callable[[], psycopg.Connection]
    """Opens one Postgres connection. Called once per turn -- see
    :mod:`agentic_erp_assistant.composition.turn` -- never shared across
    requests: a connection is not safe for concurrent use, and one per
    request is the documented rule (e2e-code-plan.md fact 12)."""

    @classmethod
    def from_env(cls) -> "AppResources":
        """Build every process-wide resource from the environment.

        Raises:
            SettingsError: Required configuration (``OPENAI_MODEL``,
                ``OPENAI_CONTEXT_WINDOW``, a malformed ``DEV_*`` toggle) is
                missing.
            ResourcesError: Qdrant has no chunks indexed, or the users file
                or ERP fixture cannot be read.
            ProviderAuthError: No ``OPENAI_API_KEY``.
        """
        settings = Settings.from_env()
        settings.log_dev_toggles()

        users = UserDirectory.load(settings.users_path)

        chat_client = OpenAIChatClient(model=settings.model)

        from agentic_erp_assistant.memory.qdrant_index import QdrantMemoryIndex
        from agentic_erp_assistant.rag.embeddings import OpenAIEmbeddingsClient
        from agentic_erp_assistant.rag.vector_index import QdrantVectorIndex

        embeddings = OpenAIEmbeddingsClient()
        retrieval = RetrievalService.load(
            vector_index=QdrantVectorIndex.from_env(), embeddings=embeddings
        )
        if len(retrieval.lexical_index) == 0:
            raise ResourcesError(
                "the Qdrant collection has no chunks indexed; run "
                "`uv run python scripts/ingest_documents.py` and try again"
            )

        memory_index = QdrantMemoryIndex.from_env()

        erp = MockErp.load()
        limiter = InMemoryRateLimiter()
        # build_default_registry's own default (DEFAULT_RATE_LIMIT) is exactly
        # what "no override" should mean, so the dev toggle is passed only
        # when it is actually set.
        registry = (
            build_default_registry(erp, read_rate_limit=settings.dev_tool_rate_limit)
            if settings.dev_tool_rate_limit is not None
            else build_default_registry(erp)
        )

        manifest = {doc.document_id: doc.entry for doc in load_documents()}

        return cls(
            settings=settings,
            users=users,
            chat_client=chat_client,
            embeddings=embeddings,
            retrieval=retrieval,
            memory_index=memory_index,
            erp=erp,
            registry=registry,
            limiter=limiter,
            manifest=manifest,
            connect=connect_postgres,
        )

    def close(self) -> None:
        """Release every connection this process opened.

        Never raises: shutdown is not the moment to discover a client's
        ``close`` is unreliable, and a leaked connection at process exit
        costs nothing an OS does not already reclaim.
        """
        for closer in (self.chat_client.close, self.retrieval.close):
            try:
                closer()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.warning("a resource failed to close cleanly", exc_info=True)
        close_index = getattr(self.memory_index, "close", None)
        if callable(close_index):
            try:
                close_index()
            except Exception:  # noqa: BLE001 - see above
                logger.warning(
                    "the memory index failed to close cleanly", exc_info=True
                )
