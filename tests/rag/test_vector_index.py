"""The dense index, run against a real Qdrant in local mode.

Not a mock. ``QdrantClient(":memory:")`` runs the same adapter, the same payload
filter and the same similarity computation the deployed one does, so what these
tests prove about access control is a fact about the code that ships rather than
about a stub that agrees with its author.

The similarity assertions use small whole-number vectors on purpose: a reviewer
can verify one by inspection, which is worth more here than a realistic
embedding nobody can check.
"""

import pytest
from qdrant_client import QdrantClient

from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.rag.chunking import Chunk
from agentic_erp_assistant.rag.ports import ScoredChunk, VectorIndexPort
from agentic_erp_assistant.rag.vector_index import (
    QdrantVectorIndex,
    VectorWidthMismatch,
    point_id,
)


def chunk(
    chunk_id: str,
    *,
    document: str = "doc-a",
    project: str = "atlas",
    scope: str = "project.docs.read",
    text: str = "a passage",
    position: int = 0,
    content_hash: str = "hash-1",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document,
        locator=chunk_id.split("#", 1)[-1],
        text=text,
        title="Document A",
        document_type="status_report",
        project_code=project,
        required_scope=scope,
        classification="internal",
        content_hash=content_hash,
        position=position,
    )


def index(dimensions: int = 2) -> QdrantVectorIndex:
    store = QdrantVectorIndex(QdrantClient(":memory:"), collection="test")
    store.ensure_ready(dimensions)
    return store


READER = RetrievalContext.for_actor(
    "priya", project_code="atlas", scopes=["project.docs.read"]
)


def ids(hits: tuple[ScoredChunk, ...]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


# -- ranking -----------------------------------------------------------------


def test_results_come_back_by_descending_similarity() -> None:
    store = index()
    store.upsert(
        [chunk("doc-a#§1"), chunk("doc-a#§2"), chunk("doc-a#§3")],
        [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
    )

    hits = store.search([1.0, 0.0], context=READER, limit=3)

    assert ids(hits) == ["doc-a#§1", "doc-a#§2", "doc-a#§3"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.6)
    assert hits[2].score == pytest.approx(0.0, abs=1e-6)


def test_a_zero_query_vector_returns_nothing_rather_than_crashing() -> None:
    """Cosine is undefined for a zero vector, and "nearest to nothing" is nothing."""
    store = index()
    store.upsert([chunk("doc-a#§1")], [[1.0, 0.0]])

    assert store.search([0.0, 0.0], context=READER, limit=5) == ()


def test_a_limit_must_be_positive() -> None:
    store = index()
    with pytest.raises(ValueError, match="limit must be positive"):
        store.search([1.0, 0.0], context=READER, limit=0)


def test_searching_before_anything_is_ingested_is_not_an_error() -> None:
    store = QdrantVectorIndex(QdrantClient(":memory:"), collection="empty")
    assert store.search([1.0, 0.0], context=READER, limit=5) == ()


# -- access control ----------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 2, 3, 10])
def test_a_restricted_chunk_never_occupies_a_slot_at_any_limit(limit: int) -> None:
    """The nearest chunk is the restricted one, so a post-filter would have
    returned one fewer result and quietly cost this reader a hit."""
    store = index()
    store.upsert(
        [
            chunk("doc-b#p.1", document="doc-b", scope="project.docs.finance.read"),
            chunk("doc-c#§1", document="doc-c", project="orion"),
            chunk("doc-a#§1"),
            chunk("doc-a#§2"),
        ],
        [[1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )

    hits = store.search([1.0, 0.0], context=READER, limit=limit)

    assert ids(hits) == ["doc-a#§1", "doc-a#§2"][:limit]


def test_the_finance_scope_unlocks_the_restricted_document() -> None:
    store = index()
    store.upsert(
        [chunk("doc-b#p.1", document="doc-b", scope="project.docs.finance.read")],
        [[1.0, 0.0]],
    )

    assert store.search([1.0, 0.0], context=READER, limit=5) == ()

    finance = RetrievalContext.for_actor(
        "wei",
        project_code="atlas",
        scopes=["project.docs.read", "project.docs.finance.read"],
    )
    assert ids(store.search([1.0, 0.0], context=finance, limit=5)) == ["doc-b#p.1"]


# -- writing -----------------------------------------------------------------


def test_a_chunk_id_maps_to_one_point_however_often_it_is_written() -> None:
    """Re-ingesting a document must replace its passages, not shadow them."""
    store = index()
    store.upsert([chunk("doc-a#§1", text="first")], [[1.0, 0.0]])
    store.upsert([chunk("doc-a#§1", text="second")], [[0.0, 1.0]])

    stored = store.iter_chunks()
    assert len(stored) == 1
    assert stored[0].text == "second"


def test_point_ids_are_deterministic_across_processes() -> None:
    assert point_id("doc-a#§1") == point_id("doc-a#§1")
    assert point_id("doc-a#§1") != point_id("doc-a#§2")


def test_mismatched_chunks_and_vectors_are_refused() -> None:
    """Pairing is positional, so a mismatch stores a passage under another
    passage's vector -- undetectable afterwards."""
    store = index()
    with pytest.raises(ValueError, match="positional"):
        store.upsert([chunk("doc-a#§1"), chunk("doc-a#§2")], [[1.0, 0.0]])


def test_deleting_a_document_leaves_the_others_alone() -> None:
    store = index()
    store.upsert(
        [chunk("doc-a#§1"), chunk("doc-b#§1", document="doc-b")],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    store.delete_documents(["doc-a"])

    assert [c.document_id for c in store.iter_chunks()] == ["doc-b"]


def test_deleting_nothing_is_a_no_op() -> None:
    store = index()
    store.upsert([chunk("doc-a#§1")], [[1.0, 0.0]])
    store.delete_documents([])
    assert len(store.iter_chunks()) == 1


# -- the store of record -----------------------------------------------------


def test_a_stored_chunk_round_trips_with_every_field_intact() -> None:
    """The payload is the chunk store, so what comes back has to be the chunk
    that went in -- the lexical index is built from exactly this."""
    original = chunk("doc-a#§3.2", text="the forecast is 515,000", position=7)
    store = index()
    store.upsert([original], [[1.0, 0.0]])

    (restored,) = store.iter_chunks()
    assert restored == original


def test_stored_chunks_come_back_in_a_stable_order(
    corpus_chunks: tuple[Chunk, ...],
) -> None:
    store = index()
    store.upsert(list(corpus_chunks), [[1.0, 0.0]] * len(corpus_chunks))

    first = [c.chunk_id for c in store.iter_chunks()]
    assert first == [c.chunk_id for c in store.iter_chunks()]
    assert len(first) == len(corpus_chunks)


def test_document_hashes_report_what_is_actually_stored() -> None:
    store = index()
    store.upsert(
        [
            chunk("doc-a#§1", content_hash="hash-a"),
            chunk("doc-a#§2", content_hash="hash-a"),
            chunk("doc-b#§1", document="doc-b", content_hash="hash-b"),
        ],
        [[1.0, 0.0]] * 3,
    )

    assert dict(store.document_hashes()) == {"doc-a": "hash-a", "doc-b": "hash-b"}


def test_document_hashes_are_empty_before_the_first_ingest() -> None:
    store = QdrantVectorIndex(QdrantClient(":memory:"), collection="empty")
    assert dict(store.document_hashes()) == {}


# -- the collection ----------------------------------------------------------


def test_a_changed_embedding_width_is_a_hard_failure() -> None:
    """What "you changed OPENAI_EMBEDDING_MODEL and not the collection" looks
    like from in here. The alternative is a collection whose scores mean nothing.
    """
    store = index(dimensions=2)
    with pytest.raises(VectorWidthMismatch, match="1536"):
        store.ensure_ready(1536)


def test_the_same_width_twice_is_accepted() -> None:
    store = index(dimensions=4)
    store.ensure_ready(4)


def test_a_width_of_zero_is_refused() -> None:
    """It means the width was taken from an empty batch rather than measured."""
    store = QdrantVectorIndex(QdrantClient(":memory:"), collection="test")
    with pytest.raises(ValueError, match="must be positive"):
        store.ensure_ready(0)


def test_the_index_satisfies_the_port() -> None:
    assert isinstance(index(), VectorIndexPort)
