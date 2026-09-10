"""Semantic recall over memory, in a Qdrant collection of its own.

The sibling of :mod:`agentic_erp_assistant.rag.vector_index`, written the same
way -- a client passed in rather than built, a transport translator, deterministic
point ids, a width that comes from the data -- and different in the two places
where memory is not a document.

**It stores identifiers, not records.** The payload carries only what the filter
matches on plus the memory id; the statement itself is never written here. So
there is exactly one copy of a memory's text, in Postgres, and this index cannot
serve a stale version of it. That is the reverse of ADR 0010, where the vector
store *is* the chunk store, and the reason is that a chunk has no lifecycle
while a memory is retired the moment something better replaces it. Two copies of
a mutable record drift, and the copy that drifts is the one being read.

**Freshness is part of the filter.** A retired memory would otherwise occupy a
slot in the top ``k`` and displace a live one -- the same argument
:func:`~agentic_erp_assistant.rag.access.qdrant_filter` makes for access, applied
to time. It is an optimization and never the guarantee: the caller hydrates
through
:meth:`~agentic_erp_assistant.memory.store.MemoryStorePort.by_id`, which
re-checks liveness and scope, so an id this index was slow to retire is dropped
one layer up rather than reaching a prompt.

The access half is not re-expressed here
----------------------------------------

The project and scope conditions come from
:func:`~agentic_erp_assistant.rag.access.qdrant_filter` unchanged, with the
memory-specific clauses appended. ADR 0008 says one access rule for every search
path, and this is the third path: a memory is at least as sensitive as the
document that taught it, and a second filter expression would be one that
diverges from the first the day somebody edits only one of them.
"""

import logging
import os
import uuid
import warnings
from collections.abc import Sequence
from typing import Any, Self

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from agentic_erp_assistant.memory.models import (
    ACTOR_BOUNDED_KINDS,
    SESSION_BOUNDED_KINDS,
    MemoryScope,
)
from agentic_erp_assistant.memory.vector_store import ScoredMemory
from agentic_erp_assistant.rag.access import (
    PROJECT_FIELD,
    SCOPE_FIELD,
    qdrant_filter,
)
from agentic_erp_assistant.rag.vector_index import (
    DEFAULT_QDRANT_URL,
    VectorIndexError,
    VectorWidthMismatch,
    translates_transport,
)
from agentic_erp_assistant.state.memory import MemoryRecord

__all__ = [
    "ACTOR_FIELD",
    "DEFAULT_MEMORY_COLLECTION",
    "KIND_FIELD",
    "MEMORY_ID_FIELD",
    "QdrantMemoryIndex",
    "RETIRED_FIELD",
    "SESSION_FIELD",
]

logger = logging.getLogger(__name__)


DEFAULT_MEMORY_COLLECTION = "project_memories"
"""Where memories are indexed when nothing says otherwise.

A second collection rather than a shared one with a type flag. Documents and
memories are ingested on different schedules, retired by different rules and
rebuilt from different sources -- a corpus re-ingest that dropped the collection
would take every remembered preference with it, which is a failure mode nobody
would predict from the command they typed.
"""

MEMORY_ID_FIELD = "memory_id"
KIND_FIELD = "kind"
SESSION_FIELD = "session_id"
ACTOR_FIELD = "actor"
RETIRED_FIELD = "retired"
"""The payload keys, as constants for the reason
:data:`~agentic_erp_assistant.rag.access.PROJECT_FIELD` is: the writer, the
filter and the reader have to agree, and a literal repeated in three places is a
rename away from a filter that silently matches nothing -- which fails by
returning no results, so it looks like a recall problem rather than a bug.

``retired`` is a boolean rather than a nullable timestamp. Qdrant can match a
boolean with an ordinary keyword condition; expressing "superseded_at is null"
would need a null-check clause, and the index does not need to know *when* a
memory was retired -- only that it was.
"""

_POINT_NAMESPACE = uuid.UUID("9c3f2d16-4a8e-5b70-9f21-6d4c8ab35e79")
"""A fixed namespace, so a memory id maps to the same point id in every process.
Any constant uuid does; this one is arbitrary and pinned. Distinct from the
chunk namespace so the two collections' ids never coincide, which costs nothing
and removes a whole class of confusing coincidence."""


def point_id(memory_id: str) -> str:
    """The deterministic point id for a memory id.

    Deterministic so re-indexing a memory replaces its point rather than
    accumulating a second copy beside the first -- the same reason
    :func:`~agentic_erp_assistant.rag.vector_index.point_id` is.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, memory_id))


def memory_filter(scope: MemoryScope) -> models.Filter:
    """The access rule, the bounds and freshness, as one server-side filter.

    Built by taking :func:`~agentic_erp_assistant.rag.access.qdrant_filter` and
    appending, never by restating its two conditions. The project and scope
    clauses are one policy with one expression, and this function's whole job is
    the three clauses that are specific to memory:

    * ``retired = false``, so a superseded memory never occupies a slot;
    * the session clause, so one conversation's intent and summary stay inside
      it;
    * the actor clause, so one person's preferences stay theirs.

    The last two are expressed as "either this kind is unbounded, or the value
    matches", which is :func:`~agentic_erp_assistant.memory.models.bounds` in
    Qdrant's vocabulary: a project decision has no session or actor to match, and
    a filter demanding one would hide every decision from every session but the
    one that made it.
    """
    access = qdrant_filter(scope.access)
    conditions: list[Any] = list(access.must or [])
    conditions.append(
        models.FieldCondition(
            key=RETIRED_FIELD, match=models.MatchValue(value=False)
        )
    )
    conditions.append(
        models.Filter(
            should=[
                models.FieldCondition(
                    key=KIND_FIELD,
                    match=models.MatchExcept(**{"except": sorted(SESSION_BOUNDED_KINDS)}),
                ),
                models.FieldCondition(
                    key=SESSION_FIELD,
                    match=models.MatchValue(value=scope.session_id),
                ),
            ]
        )
    )
    conditions.append(
        models.Filter(
            should=[
                models.FieldCondition(
                    key=KIND_FIELD,
                    match=models.MatchExcept(**{"except": sorted(ACTOR_BOUNDED_KINDS)}),
                ),
                models.FieldCondition(
                    key=ACTOR_FIELD, match=models.MatchValue(value=scope.actor)
                ),
            ]
        )
    )
    return models.Filter(must=conditions)


class QdrantMemoryIndex:
    """A :class:`~agentic_erp_assistant.memory.vector_store.MemoryVectorStorePort`
    over Qdrant.

    Takes a client rather than building one, so the same class runs against a
    server in a deployment and against ``QdrantClient(":memory:")`` in a test --
    the real adapter and the real filter evaluator, no mock and no container. The
    arrangement :class:`~agentic_erp_assistant.rag.vector_index.QdrantVectorIndex`
    already uses, and the reason its access tests are worth reading.
    """

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection: str = DEFAULT_MEMORY_COLLECTION,
    ) -> None:
        self._client = client
        self.collection = collection

    @classmethod
    def from_env(cls, *, collection: str | None = None) -> Self:
        """Build a client from ``QDRANT_URL``, ``QDRANT_API_KEY`` and
        ``QDRANT_MEMORY_COLLECTION``.

        A separate variable from ``QDRANT_COLLECTION`` on purpose: pointing the
        corpus somewhere else and pointing memory somewhere else are different
        decisions, and one variable serving both would make the second happen by
        accident whenever somebody made the first.
        """
        load_dotenv(override=False)
        url = (os.environ.get("QDRANT_URL") or "").strip() or DEFAULT_QDRANT_URL
        api_key = (os.environ.get("QDRANT_API_KEY") or "").strip() or None
        name = (
            collection
            or (os.environ.get("QDRANT_MEMORY_COLLECTION") or "").strip()
            or DEFAULT_MEMORY_COLLECTION
        )
        logger.info("connecting to qdrant at %s, memory collection %s", url, name)
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
                derived from an empty batch rather than measured.
            VectorWidthMismatch: The collection exists at a different width --
                an embedding model changed, and mixing widths would make every
                similarity score in it meaningless.
        """
        if dimensions <= 0:
            raise ValueError(
                f"dimensions must be positive, got {dimensions}; a width of zero "
                f"means it was taken from an empty batch rather than measured"
            )

        if not self._client.collection_exists(self.collection):
            logger.info(
                "creating memory collection %s at %d dimensions",
                self.collection,
                dimensions,
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
                f"memory collection {self.collection!r} holds {existing}-dimension "
                f"vectors and was offered {dimensions}. An embedding model has "
                f"changed: rebuild the collection from the memories table, which "
                f"is the record and loses nothing."
            )

    def _existing_width(self) -> int:
        params = self._client.get_collection(self.collection).config.params.vectors
        if isinstance(params, models.VectorParams):
            return params.size
        raise VectorIndexError(
            f"collection {self.collection!r} uses named vectors, which this "
            f"index does not write; it expects a single unnamed vector per point"
        )

    def _index_payload_fields(self) -> None:
        """Index the fields every query filters on.

        Correctness does not depend on it -- an unindexed filter still filters --
        but every query carries all five conditions, so they are worth indexing
        at creation rather than discovering under load. Local mode has no payload
        indexes and says so; that is a true statement about a test backend, not a
        problem to fail on, so it is logged.
        """
        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            for field in (
                PROJECT_FIELD,
                SCOPE_FIELD,
                KIND_FIELD,
                SESSION_FIELD,
                ACTOR_FIELD,
            ):
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name=RETIRED_FIELD,
                field_schema=models.PayloadSchemaType.BOOL,
            )

        for warning in raised:
            logger.debug("payload index on %s: %s", self.collection, warning.message)

    @translates_transport
    def upsert(
        self, records: Sequence[MemoryRecord], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Index these records under these vectors.

        The payload is the filter's inputs and the memory id, and nothing else --
        no statement, no timestamps. See the module docstring: one copy of the
        text, in the store that owns it.

        Raises:
            ValueError: The two sequences are different lengths. Pairing is
                positional, so a mismatch would file a memory under another
                memory's vector -- undetectable afterwards, because a mis-paired
                vector is still a valid vector.
        """
        if len(records) != len(vectors):
            raise ValueError(
                f"{len(records)} record(s) and {len(vectors)} vector(s): pairing "
                f"is positional, so these have to match exactly"
            )
        if not records:
            return

        self._client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=point_id(record.memory_id),
                    vector=list(vector),
                    payload={
                        MEMORY_ID_FIELD: record.memory_id,
                        KIND_FIELD: record.kind,
                        PROJECT_FIELD: record.project_code,
                        SCOPE_FIELD: record.required_scope,
                        SESSION_FIELD: record.session_id,
                        ACTOR_FIELD: record.actor,
                        RETIRED_FIELD: not record.live,
                    },
                )
                for record, vector in zip(records, vectors, strict=True)
            ],
        )
        logger.info("indexed %d memor(y|ies) in %s", len(records), self.collection)

    def retire(self, memory_ids: Sequence[str]) -> None:
        """Flag these as no longer current, so they stop being ranked.

        Not a delete: the index mirrors the store's supersession, so an index
        rebuilt from the ``memories`` table lands in the same state.

        Never raises, per the port. This runs after the store has already retired
        the record, so an exception here would report a failure for work that was
        done -- and the consequence of it failing is bounded, because hydration
        drops a retired id anyway. The log line is what says recall is running
        degraded.
        """
        if not memory_ids:
            return
        try:
            if not self._client.collection_exists(self.collection):
                return
            self._client.set_payload(
                collection_name=self.collection,
                payload={RETIRED_FIELD: True},
                points=[point_id(memory_id) for memory_id in memory_ids],
            )
        except Exception:  # noqa: BLE001 - the port forbids raising; see above
            logger.error(
                "could not retire %d memor(y|ies) in %s; they are already "
                "superseded in the store and hydration will drop them, so "
                "recall is degraded rather than wrong",
                len(memory_ids),
                self.collection,
            )

    # -- reading -----------------------------------------------------------

    @translates_transport
    def search(
        self,
        query_vector: Sequence[float],
        *,
        scope: MemoryScope,
        limit: int,
    ) -> tuple[ScoredMemory, ...]:
        """The nearest live memories this scope may read, best first.

        Args:
            query_vector: The embedded request.
            scope: Applied as a payload filter, so a memory this actor may not
                read is never ranked at all.
            limit: How many to return. Must be positive.

        Returns:
            Up to ``limit`` scored ids, descending. Empty is a normal answer and
            the common one -- most turns have no memory worth recalling.

        Raises:
            ValueError: ``limit`` is not positive.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        if not any(query_vector):
            # Cosine is undefined for a zero vector, and the honest answer to
            # "what is nearest to nothing" is nothing.
            logger.warning("refusing a zero query vector; returning no memories")
            return ()

        if not self._client.collection_exists(self.collection):
            logger.info("memory collection %s does not exist yet", self.collection)
            return ()

        response = self._client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            query_filter=memory_filter(scope),
            limit=limit,
            with_payload=True,
        )

        hits: list[ScoredMemory] = []
        for point in response.points:
            payload = point.payload or {}
            memory_id = payload.get(MEMORY_ID_FIELD)
            if not isinstance(memory_id, str) or not memory_id:
                logger.error(
                    "point %s in %s carries no memory id; it cannot be hydrated "
                    "and is being dropped",
                    point.id,
                    self.collection,
                )
                continue
            hits.append(ScoredMemory(memory_id=memory_id, score=float(point.score)))
        return tuple(hits)
