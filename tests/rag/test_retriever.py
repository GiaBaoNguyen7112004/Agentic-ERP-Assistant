"""The retriever the graph holds: one embedding call, one bound context.

Everything here runs against a real in-memory Qdrant and a real embeddings
client behind ``httpx.MockTransport``. The transport is what makes the
cost assertion meaningful: "exactly one embeddings request per search" is
counted, not assumed.
"""

import httpx
import pytest

from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.rag.chunking import Chunk
from agentic_erp_assistant.rag.embeddings import OpenAIEmbeddingsClient
from agentic_erp_assistant.rag.lexical import BM25Index
from agentic_erp_assistant.rag.retriever import (
    LEXICAL,
    VECTOR,
    HybridRetriever,
    RetrievalService,
)
from agentic_erp_assistant.rag.vector_index import QdrantVectorIndex
from agentic_erp_assistant.runtime.ports import DocumentRetrieverPort
from qdrant_client import QdrantClient

READER = RetrievalContext.for_actor(
    "priya", project_code="atlas", scopes=["project.docs.read"]
)
FINANCE = RetrievalContext.for_actor(
    "wei",
    project_code="atlas",
    scopes=["project.docs.read", "project.docs.finance.read"],
)


def chunk(
    chunk_id: str,
    text: str,
    *,
    project: str = "atlas",
    scope: str = "project.docs.read",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("#", 1)[0],
        locator=chunk_id.split("#", 1)[-1],
        text=text,
        title="Document",
        document_type="status_report",
        project_code=project,
        required_scope=scope,
        classification="internal",
        content_hash="hash-1",
        position=0,
    )


class QueryEmbedder:
    """A transport that answers every embeddings request with one vector."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": self.vector}
                ],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )


def build(
    chunks: list[Chunk],
    vectors: list[list[float]],
    *,
    query_vector: list[float],
    minimum_similarity: float = 0.30,
) -> tuple[RetrievalService, QueryEmbedder]:
    store = QdrantVectorIndex(QdrantClient(":memory:"), collection="test")
    store.ensure_ready(len(vectors[0]))
    store.upsert(chunks, vectors)

    handler = QueryEmbedder(query_vector)
    embeddings = OpenAIEmbeddingsClient(
        api_key="sk-test",
        model="text-embedding-3-small",
        transport=httpx.MockTransport(handler),
    )
    service = RetrievalService.load(
        vector_index=store,
        embeddings=embeddings,
        minimum_similarity=minimum_similarity,
    )
    return service, handler


def ids(hits) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


# -- cost --------------------------------------------------------------------


def test_exactly_one_embeddings_request_per_search() -> None:
    """The candidates were embedded at ingestion. A query costs one call."""
    service, handler = build(
        [chunk("doc-a#§1", "finance cutover"), chunk("doc-a#§2", "warehouse plan")],
        [[1.0, 0.0], [0.0, 1.0]],
        query_vector=[1.0, 0.0],
    )
    retriever = service.for_context(READER)

    retriever.search("finance cutover", limit=2)

    assert len(handler.requests) == 1

    retriever.search("warehouse plan", limit=2)
    assert len(handler.requests) == 2


def test_the_query_is_what_gets_embedded() -> None:
    import json

    service, handler = build(
        [chunk("doc-a#§1", "finance cutover")],
        [[1.0, 0.0]],
        query_vector=[1.0, 0.0],
    )

    service.for_context(READER).search("why is M2 late", limit=1)

    assert json.loads(handler.requests[0].content)["input"] == ["why is M2 late"]


# -- sufficiency -------------------------------------------------------------


def test_nothing_near_enough_returns_no_evidence_at_all() -> None:
    """A dense index always returns its nearest neighbours; "it returned
    something" is not evidence. Refusing here costs nothing."""
    service, _ = build(
        [chunk("doc-a#§1", "finance cutover")],
        [[1.0, 0.0]],
        query_vector=[0.0, 1.0],
    )

    assert service.for_context(READER).search("unrelated question", limit=4) == ()


def test_the_lexical_half_is_not_consulted_once_the_gate_has_failed() -> None:
    """Refusing before spending means not spending anything, including the walk
    over the lexical index."""

    class Exploding(BM25Index):
        def search(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("the lexical index was consulted after the gate failed")

    service, _ = build(
        [chunk("doc-a#§1", "finance cutover")],
        [[1.0, 0.0]],
        query_vector=[0.0, 1.0],
    )
    service.lexical_index = Exploding(service.lexical_index.chunks)

    assert service.for_context(READER).search("unrelated", limit=4) == ()


def test_a_lexical_only_chunk_is_lifted_once_the_gate_passes() -> None:
    """Dense decides whether; both decide what. The chunk carrying the exact
    identifier is far from the query vector and still reaches the answer."""
    service, _ = build(
        [
            chunk("doc-a#§1", "the finance cutover window"),
            chunk("doc-b#row R-2", "risk r-2 vendor test window owner priya"),
        ],
        [[1.0, 0.0], [0.2, 0.98]],
        query_vector=[1.0, 0.0],
    )

    hits = service.for_context(READER).search_ranked("r-2 owner", limit=4)

    assert "doc-b#row R-2" in ids(hits)
    by_id = {hit.chunk.chunk_id: hit for hit in hits}
    assert by_id["doc-b#row R-2"].ranks.keys() == {LEXICAL}
    assert VECTOR in by_id["doc-a#§1"].ranks


# -- the port ----------------------------------------------------------------


def test_the_retriever_satisfies_the_graph_port() -> None:
    service, _ = build(
        [chunk("doc-a#§1", "finance cutover")],
        [[1.0, 0.0]],
        query_vector=[1.0, 0.0],
    )
    assert isinstance(service.for_context(READER), DocumentRetrieverPort)


def test_search_returns_snippets_that_carry_their_citation() -> None:
    service, _ = build(
        [chunk("doc-a#§3.2", "the forecast is 515,000")],
        [[1.0, 0.0]],
        query_vector=[1.0, 0.0],
    )

    (snippet,) = service.for_context(READER).search("forecast", limit=4)

    assert snippet.source_id == "doc-a"
    assert snippet.locator == "§3.2"
    assert snippet.tag == "[doc-a#§3.2]"
    assert snippet.text == "the forecast is 515,000"


def test_the_limit_is_honoured() -> None:
    service, _ = build(
        [chunk(f"doc-a#§{n}", "finance cutover window") for n in range(1, 6)],
        [[1.0, 0.0]] * 5,
        query_vector=[1.0, 0.0],
    )

    assert len(service.for_context(READER).search("cutover", limit=2)) == 2


def test_a_blank_query_or_a_zero_limit_is_a_caller_bug() -> None:
    service, _ = build(
        [chunk("doc-a#§1", "finance")], [[1.0, 0.0]], query_vector=[1.0, 0.0]
    )
    retriever = service.for_context(READER)

    with pytest.raises(ValueError, match="limit must be positive"):
        retriever.search("finance", limit=0)
    with pytest.raises(ValueError, match="must not be blank"):
        retriever.search("   ", limit=4)


def test_the_same_query_twice_gives_the_same_order() -> None:
    service, _ = build(
        [chunk(f"doc-a#§{n}", "finance cutover window") for n in range(1, 6)],
        [[1.0, 0.0]] * 5,
        query_vector=[1.0, 0.0],
    )
    retriever = service.for_context(READER)

    assert [s.tag for s in retriever.search("cutover", limit=5)] == [
        s.tag for s in retriever.search("cutover", limit=5)
    ]


# -- the bound context -------------------------------------------------------


def test_one_service_serves_two_actors_who_see_different_corpora() -> None:
    service, _ = build(
        [
            chunk("doc-a#§1", "the approved budget is 480,000"),
            chunk(
                "doc-b#p.1",
                "the approved budget is 480,000",
                scope="project.docs.finance.read",
            ),
            chunk("doc-c#§1", "the approved budget is 480,000", project="orion"),
        ],
        [[1.0, 0.0]] * 3,
        query_vector=[1.0, 0.0],
    )

    reader = [s.source_id for s in service.for_context(READER).search("budget", limit=5)]
    finance = [
        s.source_id for s in service.for_context(FINANCE).search("budget", limit=5)
    ]

    assert reader == ["doc-a"]
    assert sorted(finance) == ["doc-a", "doc-b"]


def test_a_retriever_cannot_be_asked_to_search_outside_its_context() -> None:
    """There is no method that takes a context, so there is no call site that
    can forget to pass one."""
    assert not hasattr(HybridRetriever, "search_as")
    assert "context" in HybridRetriever.__dataclass_fields__


# -- construction ------------------------------------------------------------


def test_the_lexical_index_is_built_from_what_is_actually_stored(
    corpus_chunks: tuple[Chunk, ...],
) -> None:
    """Not from a second pass over the corpus: one write path means a chunk
    cannot be in one index and not the other."""
    store = QdrantVectorIndex(QdrantClient(":memory:"), collection="test")
    store.ensure_ready(2)
    store.upsert(list(corpus_chunks), [[1.0, 0.0]] * len(corpus_chunks))

    service = RetrievalService.load(
        vector_index=store,
        embeddings=OpenAIEmbeddingsClient(
            api_key="sk-test",
            model="text-embedding-3-small",
            transport=httpx.MockTransport(QueryEmbedder([1.0, 0.0])),
        ),
    )

    assert {c.chunk_id for c in service.lexical_index.chunks} == {
        c.chunk_id for c in store.iter_chunks()
    }
    assert len(service.lexical_index) == len(corpus_chunks)
