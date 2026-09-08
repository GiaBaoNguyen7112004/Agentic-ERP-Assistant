"""The real adapter against a real Qdrant, in local mode.

``QdrantClient(":memory:")`` runs Qdrant's own filter evaluator in-process, so
these are not mock assertions about a filter object -- they are assertions about
which points the database returns for a query. That is what makes the access
tests worth reading: a filter that looks right and matches the wrong points fails
here, and would fail nowhere else.

No container, no network, no marker. The distinction ``tests/persistence`` draws
does not apply: Postgres has no in-process mode, and Qdrant does.
"""

import pytest
from qdrant_client import QdrantClient

from tests.memory.builders import LATER, make_record, make_scope

from agentic_erp_assistant.memory.qdrant_index import (
    DEFAULT_MEMORY_COLLECTION,
    RETIRED_FIELD,
    QdrantMemoryIndex,
    point_id,
)
from agentic_erp_assistant.memory.vector_store import MemoryVectorStorePort
from agentic_erp_assistant.rag.vector_index import (
    VectorWidthMismatch,
    point_id as chunk_point_id,
)

NORTH = [1.0, 0.0]
EAST = [0.0, 1.0]
NORTH_BY_EAST = [0.9, 0.3]


@pytest.fixture
def index() -> QdrantMemoryIndex:
    return QdrantMemoryIndex(QdrantClient(":memory:"))


def stocked(index: QdrantMemoryIndex, *pairs) -> QdrantMemoryIndex:
    index.ensure_ready(2)
    index.upsert([record for record, _ in pairs], [vector for _, vector in pairs])
    return index


def ids(hits) -> list[str]:
    return [hit.memory_id for hit in hits]


def test_the_adapter_satisfies_the_port_structurally(index) -> None:
    assert isinstance(index, MemoryVectorStorePort)


def test_the_collection_is_its_own_and_not_the_corpus_one(index) -> None:
    """A corpus re-ingest that dropped the collection would otherwise take
    every remembered preference with it."""
    assert index.collection == DEFAULT_MEMORY_COLLECTION
    assert index.collection != "project_documents"


def test_a_memory_and_a_chunk_with_the_same_id_get_different_points() -> None:
    """Distinct namespaces, so the two collections' ids never coincide."""
    assert point_id("shared-id") != chunk_point_id("shared-id")


# --------------------------------------------------------------------------
# What is stored, and what deliberately is not
# --------------------------------------------------------------------------


def test_the_payload_carries_the_id_and_the_filter_inputs_only(index) -> None:
    """One copy of a memory's text, in the store that owns it -- so this index
    cannot serve a stale version of it."""
    stocked(index, (make_record(), NORTH))

    points, _ = index._client.scroll(
        collection_name=index.collection, limit=10, with_payload=True
    )

    payload = points[0].payload
    assert payload["memory_id"] == "mem-1"
    assert "statement" not in payload
    assert "Vietnamese" not in str(payload)


def test_a_search_returns_identifiers_and_scores(index) -> None:
    stocked(index, (make_record(), NORTH))

    hits = index.search(NORTH, scope=make_scope(), limit=4)

    assert ids(hits) == ["mem-1"]
    assert hits[0].score == pytest.approx(1.0)


def test_the_nearest_memory_ranks_first(index) -> None:
    stocked(
        index,
        (make_record(memory_id="mem-near", key="a"), NORTH),
        (make_record(memory_id="mem-far", key="b"), EAST),
    )

    assert ids(index.search(NORTH_BY_EAST, scope=make_scope(), limit=4)) == [
        "mem-near",
        "mem-far",
    ]


def test_re_indexing_replaces_a_point_rather_than_adding_one(index) -> None:
    stocked(index, (make_record(), NORTH))

    index.upsert([make_record()], [EAST])

    assert len(index.search(EAST, scope=make_scope(), limit=10)) == 1


# --------------------------------------------------------------------------
# The access rule, evaluated by the database rather than asserted about
# --------------------------------------------------------------------------


def test_another_projects_memory_never_occupies_a_slot(index) -> None:
    stocked(index, (make_record(memory_id="mem-b", project_code="borealis"), NORTH))

    assert index.search(NORTH, scope=make_scope(), limit=4) == ()


def test_a_memory_whose_scope_the_actor_lacks_is_never_ranked(index) -> None:
    stocked(
        index,
        (
            make_record(
                memory_id="mem-f", required_scope="project.docs.finance.read"
            ),
            NORTH,
        ),
    )

    assert index.search(NORTH, scope=make_scope(), limit=4) == ()


def test_another_actors_preference_is_never_ranked(index) -> None:
    stocked(index, (make_record(memory_id="mem-m", actor="marco"), NORTH))

    assert index.search(NORTH, scope=make_scope(), limit=4) == ()


def test_another_sessions_intent_is_never_ranked(index) -> None:
    stocked(
        index,
        (
            make_record(
                memory_id="mem-o", kind="intent", key="intent:x", session_id="sess-9"
            ),
            NORTH,
        ),
    )

    assert index.search(NORTH, scope=make_scope(), limit=4) == ()


def test_a_project_decision_is_ranked_across_sessions_and_actors(index) -> None:
    """The bounds clause is "either this kind is unbounded, or the value
    matches" -- a filter demanding a session would hide every decision from
    every session but the one that made it."""
    stocked(
        index,
        (
            make_record(
                memory_id="mem-d",
                kind="decision",
                key="deployment_window",
                actor="marco",
                session_id="sess-other",
            ),
            NORTH,
        ),
    )

    assert ids(index.search(NORTH, scope=make_scope(), limit=4)) == ["mem-d"]


def test_this_sessions_summary_is_ranked(index) -> None:
    stocked(
        index,
        (
            make_record(memory_id="mem-s", kind="session_summary", key="session"),
            NORTH,
        ),
    )

    assert ids(index.search(NORTH, scope=make_scope(), limit=4)) == ["mem-s"]


# --------------------------------------------------------------------------
# Freshness, which is a filter clause and not the guarantee
# --------------------------------------------------------------------------


def test_a_retired_memory_stops_occupying_a_slot(index) -> None:
    stocked(index, (make_record(), NORTH))

    index.retire(["mem-1"])

    assert index.search(NORTH, scope=make_scope(), limit=4) == ()


def test_retiring_flags_the_point_rather_than_deleting_it(index) -> None:
    """The index mirrors the store's supersession, so a rebuild from the
    memories table lands in the same state."""
    stocked(index, (make_record(), NORTH))

    index.retire(["mem-1"])

    points, _ = index._client.scroll(
        collection_name=index.collection, limit=10, with_payload=True
    )
    assert len(points) == 1
    assert points[0].payload[RETIRED_FIELD] is True


def test_indexing_an_already_retired_record_does_not_resurrect_it(index) -> None:
    stocked(index, (make_record().retired(LATER), NORTH))

    assert index.search(NORTH, scope=make_scope(), limit=4) == ()


def test_retiring_an_unknown_id_does_not_raise(index) -> None:
    """This runs after the store has already retired the record."""
    stocked(index, (make_record(), NORTH))

    index.retire(["mem-nope"])


def test_retiring_before_the_collection_exists_does_not_raise(index) -> None:
    index.retire(["mem-1"])


def test_retiring_nothing_touches_the_database(index) -> None:
    index.retire([])


# --------------------------------------------------------------------------
# Widths, pairing and the empty cases
# --------------------------------------------------------------------------


def test_searching_before_anything_is_indexed_is_a_normal_answer(index) -> None:
    assert index.search(NORTH, scope=make_scope(), limit=4) == ()


def test_a_zero_query_vector_returns_nothing(index) -> None:
    stocked(index, (make_record(), NORTH))

    assert index.search([0.0, 0.0], scope=make_scope(), limit=4) == ()


def test_a_width_of_zero_means_nothing_was_measured(index) -> None:
    with pytest.raises(ValueError, match="positive"):
        index.ensure_ready(0)


def test_a_changed_embedding_model_is_a_loud_failure(index) -> None:
    """Mixing widths would make every similarity score in the collection
    meaningless."""
    index.ensure_ready(2)

    with pytest.raises(VectorWidthMismatch, match="rebuild the collection"):
        index.ensure_ready(3)


def test_mismatched_records_and_vectors_are_refused(index) -> None:
    index.ensure_ready(2)

    with pytest.raises(ValueError, match="positional"):
        index.upsert([make_record()], [NORTH, EAST])


def test_indexing_nothing_is_allowed(index) -> None:
    index.ensure_ready(2)

    index.upsert([], [])


def test_a_non_positive_limit_is_refused(index) -> None:
    with pytest.raises(ValueError, match="positive"):
        index.search(NORTH, scope=make_scope(), limit=0)
