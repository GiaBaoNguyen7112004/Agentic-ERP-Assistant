"""Semantic recall over memory, as a shape -- and an exact one for tests.

The port is provider-neutral for the reason
:class:`~agentic_erp_assistant.rag.ports.VectorIndexPort` is: what stores vectors
and searches them is somebody else's software, so the rest of the package depends
on a shape and the Qdrant code sits behind it. What is *not* shared with the
document index is the contract, and the difference is the design.

The index returns identifiers, never records
--------------------------------------------

:meth:`MemoryVectorStorePort.search` gives back ``(memory_id, score)`` pairs, and
the caller hydrates them through
:meth:`~agentic_erp_assistant.memory.store.MemoryStorePort.by_id`. That is the
opposite of ADR 0010, where the vector store *is* the chunk store, and the
opposition is deliberate: a chunk has no lifecycle, so a copy of it in Qdrant
cannot be wrong, while a memory is retired the moment a better one replaces it.
Two copies of a mutable record drift, and the copy that drifts is always the one
being read.

So the relational store is the record and this is an index over it. The
consequence a reviewer should hold this to: a memory the index was slow to
retire is dropped at hydration, because the hydration step re-checks liveness and
scope. The freshness filter here is an optimization -- it keeps a retired memory
from occupying a slot in the top ``k`` -- and never the thing that makes a stale
recall impossible.

Both filters are the access rule, unchanged
-------------------------------------------

Tenant and scope filtering is
:func:`~agentic_erp_assistant.rag.access.is_authorized`, applied to a
:class:`~agentic_erp_assistant.state.memory.MemoryRecord` exactly as it is
applied to a chunk, plus the session and actor bounds
:func:`~agentic_erp_assistant.memory.models.in_bounds` sets. Neither is
re-expressed here. A second access rule for memory is a rule that diverges from
the document one the day somebody edits only one of them, and the divergence is
invisible until a preference is recalled for the wrong person.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentic_erp_assistant.memory.models import MemoryScope, in_bounds
from agentic_erp_assistant.rag.access import is_authorized
from agentic_erp_assistant.state.memory import MemoryRecord

__all__ = [
    "InMemoryMemoryVectorStore",
    "MemoryVectorStorePort",
    "ScoredMemory",
]


@dataclass(frozen=True)
class ScoredMemory:
    """One search hit: which memory, and how close it was.

    An id rather than a record, for the reason the module docstring gives. The
    score is here and deliberately not on
    :class:`~agentic_erp_assistant.state.memory.MemoryRecord` -- the same
    argument :class:`~agentic_erp_assistant.rag.ports.ScoredChunk` makes: a
    number nothing branches on, riding inside the object that goes into a
    prompt, is telemetry pretending to be knowledge.
    """

    memory_id: str
    score: float


@runtime_checkable
class MemoryVectorStorePort(Protocol):
    """What recall assumes about a semantic index over memory."""

    def ensure_ready(self, dimensions: int) -> None:
        """Make the index able to accept vectors of this width.

        Called with a width measured from a real embedding batch, never a
        constant -- and an index already holding a different width must refuse
        rather than accept a mixture, for the reason
        :meth:`~agentic_erp_assistant.rag.ports.VectorIndexPort.ensure_ready`
        gives: similarity across two widths means nothing.
        """
        ...

    def upsert(
        self, records: Sequence[MemoryRecord], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Index these records under these vectors, replacing any sharing an id.

        Pairing is positional, so the two sequences must be the same length. A
        mismatch would file a memory under another memory's vector, which
        nothing downstream can detect -- a mis-paired vector is still a valid
        vector, and the symptom is recall that is quietly, unexplainably wrong.
        """
        ...

    def retire(self, memory_ids: Sequence[str]) -> None:
        """Mark these as no longer current, so they stop being ranked.

        Not a delete. The index mirrors the store's supersession rather than
        forgetting rows, so an index rebuilt from the store lands in the same
        state -- and so a failure here degrades recall instead of destroying it.

        Must not raise on an id it does not hold: this runs after the store has
        already retired the record, and an exception at this point would report
        a failure for work that was actually done.
        """
        ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        scope: MemoryScope,
        limit: int,
    ) -> tuple[ScoredMemory, ...]:
        """The nearest live memories this scope may read, best first.

        The scope is applied *before* ranking, never after. Filtering afterwards
        would silently shrink the result for exactly the actors with the fewest
        entitlements -- see
        :func:`~agentic_erp_assistant.rag.access.qdrant_filter`, whose reasoning
        applies here unchanged.

        Returning fewer than ``limit``, including none, is a normal answer: a
        session with no memory worth recalling is the common case, not an error.
        """
        ...


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, or ``0.0`` when either side has no direction.

    Zero rather than an exception for a zero vector: "what is nearest to
    nothing" has an honest answer, and it is nothing -- the same call
    :meth:`~agentic_erp_assistant.rag.vector_index.QdrantVectorIndex.search`
    makes for the same reason.
    """
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    scale = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / scale if scale else 0.0


class InMemoryMemoryVectorStore:
    """An exact index over a dict, and the default when no Qdrant is configured.

    Exact rather than approximate, which is the one way it differs from the real
    adapter and the one way that is safe to differ: with a few dozen memories a
    brute-force cosine *is* the correct answer, so a test asserting on ranking is
    asserting on the ranking a real index approximates.

    What it does not fake is the filtering. Access and bounds go through
    :func:`~agentic_erp_assistant.rag.access.is_authorized` and
    :func:`~agentic_erp_assistant.memory.models.in_bounds` -- the same functions
    the Qdrant filter is built from -- so a test proving that a retired memory or
    another project's memory never comes back is proving it about the rule, not
    about this class.
    """

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._retired: set[str] = set()
        self.dimensions: int | None = None

    def ensure_ready(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError(
                f"dimensions must be positive, got {dimensions}; a width of zero "
                f"means it was taken from an empty batch rather than measured"
            )
        if self.dimensions is not None and self.dimensions != dimensions:
            raise ValueError(
                f"index holds {self.dimensions}-dimension vectors and was offered "
                f"{dimensions}; mixing widths would make every score meaningless"
            )
        self.dimensions = dimensions

    def upsert(
        self, records: Sequence[MemoryRecord], vectors: Sequence[Sequence[float]]
    ) -> None:
        if len(records) != len(vectors):
            raise ValueError(
                f"{len(records)} record(s) and {len(vectors)} vector(s): pairing "
                f"is positional, so these have to match exactly"
            )
        for record, vector in zip(records, vectors, strict=True):
            self._records[record.memory_id] = record
            self._vectors[record.memory_id] = tuple(float(value) for value in vector)
            self._retired.discard(record.memory_id)
            if record.superseded_at is not None:
                self._retired.add(record.memory_id)

    def retire(self, memory_ids: Sequence[str]) -> None:
        self._retired.update(memory_ids)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        scope: MemoryScope,
        limit: int,
    ) -> tuple[ScoredMemory, ...]:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if not any(query_vector):
            return ()

        scored = [
            ScoredMemory(
                memory_id=memory_id,
                score=_cosine(query_vector, self._vectors[memory_id]),
            )
            for memory_id, record in self._records.items()
            if memory_id not in self._retired
            and is_authorized(record, scope.access)
            and in_bounds(record, scope)
        ]
        scored.sort(key=lambda hit: (-hit.score, hit.memory_id))
        return tuple(scored[:limit])
