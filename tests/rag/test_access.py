"""The whole access-control policy, and the proof that it is only one policy.

The last test is the one that matters most. The rule is written twice -- once as
a comparison the lexical path runs, once as a payload filter the vector index
runs -- and the failure mode this module exists to prevent is those two drifting
apart. So it puts the real corpus into a real (in-memory) Qdrant collection and
asserts the filter accepts exactly the chunks the comparison accepts.
"""

from dataclasses import dataclass

import pytest
from qdrant_client import QdrantClient, models

from agentic_erp_assistant.rag.access import (
    PROJECT_FIELD,
    SCOPE_FIELD,
    RetrievalContext,
    authorized,
    is_authorized,
    qdrant_filter,
)
from agentic_erp_assistant.rag.chunking import Chunk


@dataclass(frozen=True)
class Restricted:
    project_code: str
    required_scope: str


ATLAS_DOC = Restricted("atlas", "project.docs.read")
ATLAS_FINANCE = Restricted("atlas", "project.docs.finance.read")
ORION_DOC = Restricted("orion", "project.docs.read")


def context(project: str = "atlas", *scopes: str) -> RetrievalContext:
    return RetrievalContext.for_actor(
        "priya", project_code=project, scopes=scopes
    )


# -- the rule ----------------------------------------------------------------


def test_a_matching_project_and_scope_is_allowed() -> None:
    assert is_authorized(ATLAS_DOC, context("atlas", "project.docs.read"))


def test_another_project_is_denied_even_with_a_valid_scope() -> None:
    assert not is_authorized(ORION_DOC, context("atlas", "project.docs.read"))


def test_a_narrower_scope_is_denied_inside_the_right_project() -> None:
    assert not is_authorized(ATLAS_FINANCE, context("atlas", "project.docs.read"))
    assert is_authorized(
        ATLAS_FINANCE,
        context("atlas", "project.docs.read", "project.docs.finance.read"),
    )


def test_holding_no_scopes_is_denied_rather_than_unrestricted() -> None:
    """An empty scope set is far more often a bug than a real entitlement."""
    assert not is_authorized(ATLAS_DOC, context("atlas"))


def test_an_unknown_scope_grants_nothing() -> None:
    assert not is_authorized(ATLAS_DOC, context("atlas", "project.docs.admin"))


def test_scopes_are_compared_exactly() -> None:
    assert not is_authorized(ATLAS_DOC, context("atlas", "PROJECT.DOCS.READ"))
    assert not is_authorized(ATLAS_DOC, context("Atlas", "project.docs.read"))


# -- the context -------------------------------------------------------------


def test_a_context_must_name_an_actor_and_a_project() -> None:
    with pytest.raises(ValueError, match="actor"):
        RetrievalContext(actor="  ", project_code="atlas")
    with pytest.raises(ValueError, match="project_code"):
        RetrievalContext(actor="priya", project_code="")


def test_bulk_filtering_keeps_order_and_drops_the_rest() -> None:
    kept = authorized(
        [ATLAS_DOC, ORION_DOC, ATLAS_FINANCE, ATLAS_DOC],
        context("atlas", "project.docs.read"),
    )
    assert kept == [ATLAS_DOC, ATLAS_DOC]


# -- the drift test ----------------------------------------------------------


def corpus_collection(
    client: QdrantClient, name: str, chunks: tuple[Chunk, ...]
) -> tuple[Chunk, ...]:
    """Put the real corpus into a real in-memory collection."""
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=index,
                vector=[1.0, 0.0],
                payload={
                    "chunk_id": chunk.chunk_id,
                    PROJECT_FIELD: chunk.project_code,
                    SCOPE_FIELD: chunk.required_scope,
                },
            )
            for index, chunk in enumerate(chunks)
        ],
    )
    return chunks


@pytest.mark.parametrize(
    "project,scopes",
    [
        ("atlas", ("project.docs.read",)),
        ("atlas", ("project.docs.read", "project.docs.finance.read")),
        ("atlas", ("project.docs.finance.read",)),
        ("atlas", ()),
        ("orion", ("project.docs.read",)),
        ("orion", ("project.docs.finance.read",)),
        ("nonexistent", ("project.docs.read",)),
    ],
)
def test_the_payload_filter_accepts_exactly_what_the_rule_accepts(
    project: str, scopes: tuple[str, ...], corpus_chunks: tuple[Chunk, ...]
) -> None:
    """One policy, written twice, proven to be one policy.

    Run against a real Qdrant filter evaluator in local mode -- not against a
    hand-rolled reimplementation of what the filter is assumed to do, which
    would only ever prove that the assumption agrees with itself.
    """
    client = QdrantClient(":memory:")
    chunks = corpus_collection(client, "drift", corpus_chunks)
    ctx = RetrievalContext.for_actor("priya", project_code=project, scopes=scopes)

    points, _ = client.scroll(
        collection_name="drift",
        scroll_filter=qdrant_filter(ctx),
        limit=len(chunks) + 1,
        with_payload=True,
    )

    by_filter = {point.payload["chunk_id"] for point in points}
    by_rule = {chunk.chunk_id for chunk in chunks if is_authorized(chunk, ctx)}
    assert by_filter == by_rule


def test_the_filter_refuses_everything_for_an_actor_with_no_scopes(
    corpus_chunks: tuple[Chunk, ...],
) -> None:
    """A filter that dropped the clause would turn "no entitlements" into "no
    restriction", which is the one direction this must never fail in."""
    client = QdrantClient(":memory:")
    corpus_collection(client, "empty-scopes", corpus_chunks)

    points, _ = client.scroll(
        collection_name="empty-scopes",
        scroll_filter=qdrant_filter(context("atlas")),
        limit=100,
    )
    assert points == []
