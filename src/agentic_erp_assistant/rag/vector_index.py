"""The dense half of search, and the store of record for chunks.

Qdrant holds the vectors, and it also holds the chunks themselves as point
payloads. That second job is a decision, not a convenience: the lexical index is
built by reading this store back rather than by a second pass over the corpus,
so there is one write path and a chunk cannot exist in one index and not the
other. A passage that a keyword question can find and a semantic one cannot --
with nothing in any trace to say why -- is exactly the bug that arrangement makes
impossible.

Access control is pushed into the query, not applied to the answer
------------------------------------------------------------------

:func:`~agentic_erp_assistant.rag.access.qdrant_filter` goes into the request, so
a restricted chunk never occupies a slot in the top ``k``. Filtering afterwards
would silently shrink results for the users with the fewest permissions, which is
the opposite of what a limit is supposed to mean. The returned points are then
re-checked with :func:`~agentic_erp_assistant.rag.access.is_authorized` -- not
because the filter is doubted, but because the day someone edits one of the two
is the day the second check earns its keep, and it costs a comparison per hit.

The vector width comes from the data, never from a constant
-----------------------------------------------------------

:meth:`QdrantVectorIndex.ensure_ready` is called with a width measured from a
real embedding batch. A collection that already exists at a different width is a
hard failure, because that is what "you changed OPENAI_EMBEDDING_MODEL and not
the collection" looks like from in here -- and the alternative to failing is a
collection holding two vector widths, whose similarity scores mean nothing.

Ids are derived from the chunk id
---------------------------------

Qdrant wants a uuid or an integer, and a chunk id is neither, so the point id is
a UUIDv5 of the chunk id. Deterministic on purpose: re-ingesting a document
replaces its points rather than accumulating a second copy of every passage
beside the first.
"""

import logging
import os
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import wraps
from typing import Any, Self

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from agentic_erp_assistant.rag.access import (
    PROJECT_FIELD,
    SCOPE_FIELD,
    RetrievalContext,
    is_authorized,
    qdrant_filter,
)
from agentic_erp_assistant.rag.chunking import Chunk
from agentic_erp_assistant.rag.ports import ScoredChunk

__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_QDRANT_URL",
    "DOCUMENT_FIELD",
    "HASH_FIELD",
    "QdrantVectorIndex",
    "translates_transport",
    "VectorIndexError",
    "VectorStoreUnavailable",
    "VectorWidthMismatch",
]

logger = logging.getLogger(__name__)


DEFAULT_QDRANT_URL = "http://localhost:6333"
"""Where ``docker compose up -d qdrant`` puts it.

A constructor default rather than a required variable, like
:data:`~agentic_erp_assistant.llm.adapters.openai_chat.DEFAULT_BASE_URL`: it
costs nothing, reveals nothing, and pointing somewhere else is a deliberate act
at a place a reader can see.
"""

DEFAULT_COLLECTION = "project_documents"
"""The collection name when nothing says otherwise.

Worth overriding per embedding model in a shared deployment -- the vectors in a
collection are only comparable to vectors from the model that made them -- which
is why :meth:`QdrantVectorIndex.ensure_ready` refuses a width change rather than
trusting the name to have been changed.
"""

DOCUMENT_FIELD = "document_id"
HASH_FIELD = "content_hash"
"""Payload keys the incremental re-ingest reads. Constants for the reason
``access.PROJECT_FIELD`` is one: a literal repeated across a writer, a filter and
a reader is a rename away from silently matching nothing."""

_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")
"""A fixed namespace, so a chunk id maps to the same point id on every run and
in every process. Any constant uuid does; this one is arbitrary and pinned."""

_SCROLL_PAGE = 256
"""How many points one scroll page carries. Large enough that this corpus is one
or two round trips, small enough that a bigger one does not build a huge list in
a single response."""


class VectorIndexError(RuntimeError):
    """The vector store cannot serve the request as configured."""


class VectorStoreUnavailable(VectorIndexError):
    """The store could not be reached, or refused the request outright.

    The vendor exception underneath says "connection refused", which is true and
    unhelpful at three in the morning. This one says what to do about it, and it
    exists here rather than in the script for the reason
    ``llm/adapters/openai_chat.py`` translates httpx errors: the adapter is where
    a vendor vocabulary stops, so no caller above it has to import a Qdrant
    exception to know that the database is down.
    """


class VectorWidthMismatch(VectorIndexError):
    """The collection holds vectors of a different width than the ones offered.

    Its own type because the fix is specific and a caller may want to say so:
    either the embedding model changed and the collection must be rebuilt, or the
    collection name is being shared by two models that should each have their
    own.
    """


def translates_transport[**P, R](
    method: Callable[P, R],
) -> Callable[P, R]:
    """Turn a Qdrant transport failure into :class:`VectorStoreUnavailable`.

    Public, and used by :mod:`agentic_erp_assistant.memory.qdrant_index` as
    well as here. One translator rather than two: the message it produces --
    naming the collection and telling the reader to start the container -- is
    the thing an operator sees at three in the morning, and two copies of it
    is one that stops matching the other.
    """

    @wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return method(*args, **kwargs)
        except (ResponseHandlingException, UnexpectedResponse) as error:
            index = args[0]
            raise VectorStoreUnavailable(
                f"cannot reach the vector store for collection "
                f"{getattr(index, 'collection', '?')!r}: {error}. Is Qdrant "
                f"running? Start it with `docker compose up -d qdrant`, or set "
                f"QDRANT_URL to point at the one you mean."
            ) from error

    return wrapper


def point_id(chunk_id: str) -> str:
    """The deterministic point id for a chunk id."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class QdrantVectorIndex:
    """A :class:`~agentic_erp_assistant.rag.ports.VectorIndexPort` over Qdrant.

    Takes a client rather than building one, so the same class runs against a
    server in a deployment and against ``QdrantClient(":memory:")`` in a test --
    the real adapter, the real filter evaluator, no mock and no container. See
    :meth:`from_env` for the construction a script uses.
    """

    def __init__(self, client: QdrantClient, *, collection: str = DEFAULT_COLLECTION) -> None:
        self._client = client
        self.collection = collection

    @classmethod
    def from_env(cls, *, collection: str | None = None) -> Self:
        """Build a client from ``QDRANT_URL``, ``QDRANT_API_KEY`` and
        ``QDRANT_COLLECTION``.

        The api key is optional and normally absent: a local Qdrant started by
        the compose file has no authentication, and requiring a value would mean
        inventing one.
        """
        load_dotenv(override=False)
        url = (os.environ.get("QDRANT_URL") or "").strip() or DEFAULT_QDRANT_URL
        api_key = (os.environ.get("QDRANT_API_KEY") or "").strip() or None
        name = (
            collection
            or (os.environ.get("QDRANT_COLLECTION") or "").strip()
            or DEFAULT_COLLECTION
        )
        logger.info("connecting to qdrant at %s, collection %s", url, name)
        return cls(QdrantClient(url=url, api_key=api_key), collection=name)

    def close(self) -> None:
        """Release the connection. Safe to call more than once."""
        self._client.close()

    # -- writing -----------------------------------------------------------

    @translates_transport
    def ensure_ready(self, dimensions: int) -> None:
        """Create the collection at this width, or check the existing one.

        Raises:
            ValueError: ``dimensions`` is not positive, which means it was
                derived from an empty batch and nothing was measured.
            VectorWidthMismatch: The collection exists at a different width.
        """
        if dimensions <= 0:
            raise ValueError(
                f"dimensions must be positive, got {dimensions}; a width of zero "
                f"means it was taken from an empty batch rather than measured"
            )

        if not self._client.collection_exists(self.collection):
            logger.info(
                "creating collection %s at %d dimensions", self.collection, dimensions
            )
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dimensions, distance=models.Distance.COSINE
                ),
            )
            self._index_payload_fields()
            return

        existing = self._existing_width()
        if existing != dimensions:
            raise VectorWidthMismatch(
                f"collection {self.collection!r} holds {existing}-dimension "
                f"vectors and was offered {dimensions}. An embedding model has "
                f"changed: rebuild the collection, or point QDRANT_COLLECTION at "
                f"one of its own. Mixing widths would make every similarity "
                f"score in it meaningless."
            )

    def _existing_width(self) -> int:
        """The width of the collection as it stands."""
        params = self._client.get_collection(self.collection).config.params.vectors
        if isinstance(params, models.VectorParams):
            return params.size
        raise VectorIndexError(
            f"collection {self.collection!r} uses named vectors, which this "
            f"index does not write; it expects a single unnamed vector per point"
        )

    def _index_payload_fields(self) -> None:
        """Index the fields the access filter matches on.

        Correctness does not depend on this -- an unindexed filter still filters
        -- but every query this index ever serves carries both access
        conditions, so they are worth indexing at creation rather than
        discovering later under load.

        Local mode has no payload indexes and warns about it. That is a true
        statement about a backend used for tests, not a problem to fail on, so
        the warning is recorded in the log rather than raised at a developer who
        cannot act on it.
        """
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            for field in (PROJECT_FIELD, SCOPE_FIELD, DOCUMENT_FIELD):
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

        for warning in raised:
            logger.debug("payload index on %s: %s", self.collection, warning.message)

    @translates_transport
    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Store chunks with their vectors, replacing any that share an id.

        Raises:
            ValueError: The two sequences are different lengths. Pairing is
                positional, so a mismatch means somebody would be stored under
                another passage's vector -- undetectable afterwards, because a
                mis-paired vector is still a valid vector.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"{len(chunks)} chunk(s) and {len(vectors)} vector(s): pairing is "
                f"positional, so these have to match exactly"
            )
        if not chunks:
            return

        self._client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector=list(vector),
                    payload=chunk.model_dump(mode="json"),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        logger.info("upserted %d chunk(s) into %s", len(chunks), self.collection)

    @translates_transport
    def delete_documents(self, document_ids: Iterable[str]) -> None:
        """Remove every stored chunk belonging to these documents.

        By filter rather than by id, because the ids of a document's *old*
        chunks are not knowable from the new ones: re-chunking a document that
        gained a section renames the parts after it, and deleting only the ids
        about to be written would leave the renamed remnants behind as
        unreachable duplicates.
        """
        ids = [document_id for document_id in document_ids]
        if not ids or not self._client.collection_exists(self.collection):
            return

        self._client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=DOCUMENT_FIELD, match=models.MatchAny(any=sorted(ids))
                        )
                    ]
                )
            ),
        )
        logger.info("deleted chunks for %d document(s)", len(ids))

    # -- reading -----------------------------------------------------------

    @translates_transport
    def _scroll(self, fields: list[str] | bool) -> list[dict[str, Any]]:
        """Page through every payload in the collection."""
        if not self._client.collection_exists(self.collection):
            return []

        payloads: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=fields,
                with_vectors=False,
            )
            payloads.extend(point.payload or {} for point in points)
            if offset is None:
                return payloads

    def document_hashes(self) -> Mapping[str, str]:
        """``{document_id: content_hash}`` for everything stored.

        Reads only the two fields it needs. A document whose stored chunks
        disagree about the hash is reported under the first one seen and will
        compare unequal to the corpus, so it gets re-ingested -- which is the
        right outcome for an index left half-written by an interrupted run.
        """
        hashes: dict[str, str] = {}
        for payload in self._scroll([DOCUMENT_FIELD, HASH_FIELD]):
            document_id = payload.get(DOCUMENT_FIELD)
            content_hash = payload.get(HASH_FIELD)
            if isinstance(document_id, str) and isinstance(content_hash, str):
                hashes.setdefault(document_id, content_hash)
        return hashes

    def iter_chunks(self) -> tuple[Chunk, ...]:
        """Every stored chunk, in a stable order.

        Ordered by document and position rather than by whatever order the store
        pages them out in, so the lexical index built from this is identical
        between runs -- and so a BM25 score, which depends on corpus statistics
        and not on order, is at least reproducible in its tie-breaking.
        """
        chunks = [Chunk.model_validate(payload) for payload in self._scroll(True)]
        chunks.sort(key=lambda chunk: (chunk.document_id, chunk.position))
        return tuple(chunks)

    @translates_transport
    def search(
        self,
        query_vector: Sequence[float],
        *,
        context: RetrievalContext,
        limit: int,
    ) -> tuple[ScoredChunk, ...]:
        """The nearest chunks this context may read, best first.

        Args:
            query_vector: The embedded query.
            context: Applied as a payload filter, so restricted chunks are never
                ranked at all.
            limit: How many to return. Must be positive.

        Returns:
            Up to ``limit`` scored chunks, descending. Empty is a normal answer.

        Raises:
            ValueError: ``limit`` is not positive.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        if not any(query_vector):
            # Cosine distance is undefined for a zero vector. Qdrant would
            # either refuse it or return nonsense; either way the honest answer
            # to "what is nearest to nothing" is nothing.
            logger.warning("refusing a zero query vector; returning no hits")
            return ()

        if not self._client.collection_exists(self.collection):
            logger.warning("collection %s does not exist yet", self.collection)
            return ()

        response = self._client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            query_filter=qdrant_filter(context),
            limit=limit,
            with_payload=True,
        )

        hits: list[ScoredChunk] = []
        for point in response.points:
            chunk = Chunk.model_validate(point.payload or {})
            if not is_authorized(chunk, context):
                # Unreachable while the filter and the rule agree, and cheap
                # enough to keep for the day one of them is edited alone.
                logger.error(
                    "vector search returned %s, which %s may not read; the "
                    "payload filter and is_authorized have diverged",
                    chunk.chunk_id,
                    context.actor,
                )
                continue
            hits.append(ScoredChunk(chunk=chunk, score=float(point.score)))
        return tuple(hits)
