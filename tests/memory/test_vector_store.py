"""The index ranks; it never decides who may read. Both filters are the access
rule, applied before ranking, and freshness is one of them."""

import pytest

from tests.memory.builders import LATER, make_record, make_scope

from agentic_erp_assistant.memory.vector_store import (
    InMemoryMemoryVectorStore,
    MemoryVectorStorePort,
    ScoredMemory,
)

NORTH = [1.0, 0.0]
EAST = [0.0, 1.0]
NORTH_BY_EAST = [0.9, 0.3]
"""Nearer NORTH than EAST, and deliberately not equidistant: a query exactly
between the two would be testing the tie-break rather than the ranking."""


def index(*pairs: tuple[object, list[float]]) -> InMemoryMemoryVectorStore:
    store = InMemoryMemoryVectorStore()
    store.ensure_ready(2)
    if pairs:
        store.upsert([record for record, _ in pairs], [vector for _, vector in pairs])  # type: ignore[arg-type]
    return store


def test_the_fake_satisfies_the_port_structurally() -> None:
    assert isinstance(InMemoryMemoryVectorStore(), MemoryVectorStorePort)


# --------------------------------------------------------------------------
# Ranking, and what comes back
# --------------------------------------------------------------------------


def test_search_returns_identifiers_and_scores_not_records() -> None:
    """The opposite of ADR 0010: a memory is retired the moment a better one
    replaces it, and two copies of a mutable record drift."""
    store = index((make_record(), NORTH))

    hits = store.search(NORTH, scope=make_scope(), limit=4)

    assert hits == (ScoredMemory(memory_id="mem-1", score=pytest.approx(1.0)),)


def test_the_nearest_memory_ranks_first() -> None:
    store = index(
        (make_record(memory_id="mem-near", key="a"), NORTH),
        (make_record(memory_id="mem-far", key="b"), EAST),
    )

    hits = store.search(NORTH_BY_EAST, scope=make_scope(), limit=4)

    assert [hit.memory_id for hit in hits] == ["mem-near", "mem-far"]


def test_the_limit_truncates_after_ranking() -> None:
    store = index(
        (make_record(memory_id="mem-near", key="a"), NORTH),
        (make_record(memory_id="mem-far", key="b"), EAST),
    )

    assert [h.memory_id for h in store.search(NORTH, scope=make_scope(), limit=1)] == [
        "mem-near"
    ]


def test_ties_break_on_identifier_so_a_ranking_is_reproducible() -> None:
    store = index(
        (make_record(memory_id="mem-b", key="b"), NORTH),
        (make_record(memory_id="mem-a", key="a"), NORTH),
    )

    hits = store.search(NORTH, scope=make_scope(), limit=4)

    assert [hit.memory_id for hit in hits] == ["mem-a", "mem-b"]


def test_an_empty_index_is_a_normal_answer() -> None:
    """A session with no memory worth recalling is the common case."""
    assert index().search(NORTH, scope=make_scope(), limit=4) == ()


def test_a_zero_query_vector_returns_nothing_rather_than_nonsense() -> None:
    """"What is nearest to nothing" has an honest answer."""
    store = index((make_record(), NORTH))

    assert store.search([0.0, 0.0], scope=make_scope(), limit=4) == ()


# --------------------------------------------------------------------------
# Tenant and scope: the document access rule, applied before ranking
# --------------------------------------------------------------------------


def test_another_projects_memory_never_occupies_a_slot() -> None:
    store = index((make_record(memory_id="mem-b", project_code="borealis"), NORTH))

    assert store.search(NORTH, scope=make_scope(), limit=4) == ()


def test_a_memory_whose_scope_the_actor_lacks_is_never_ranked() -> None:
    """Filtering after ranking would silently shrink the result for exactly the
    actors with the fewest entitlements."""
    store = index(
        (make_record(memory_id="mem-f", required_scope="project.docs.finance.read"), NORTH)
    )

    assert store.search(NORTH, scope=make_scope(), limit=4) == ()


def test_another_actors_preference_is_never_ranked() -> None:
    store = index((make_record(memory_id="mem-m", actor="marco"), NORTH))

    assert store.search(NORTH, scope=make_scope(), limit=4) == ()


def test_another_sessions_intent_is_never_ranked() -> None:
    store = index(
        (
            make_record(
                memory_id="mem-o", kind="intent", key="intent:x", session_id="sess-9"
            ),
            NORTH,
        )
    )

    assert store.search(NORTH, scope=make_scope(), limit=4) == ()


def test_a_project_decision_is_ranked_for_any_session_on_the_project() -> None:
    store = index(
        (
            make_record(
                memory_id="mem-d",
                kind="decision",
                key="deployment_window",
                actor="marco",
                session_id="sess-other",
            ),
            NORTH,
        )
    )

    assert [h.memory_id for h in store.search(NORTH, scope=make_scope(), limit=4)] == [
        "mem-d"
    ]


# --------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------


def test_a_retired_memory_stops_being_ranked() -> None:
    store = index((make_record(), NORTH))

    store.retire(["mem-1"])

    assert store.search(NORTH, scope=make_scope(), limit=4) == ()


def test_indexing_an_already_retired_record_does_not_resurrect_it() -> None:
    """A rebuild from the store has to land in the same state the store is in."""
    store = index((make_record().retired(LATER), NORTH))

    assert store.search(NORTH, scope=make_scope(), limit=4) == ()


def test_retiring_an_unknown_id_is_not_an_error() -> None:
    """This runs after the store has already retired the record; raising here
    would report a failure for work that was done."""
    index().retire(["mem-nope"])


def test_re_indexing_a_live_record_makes_it_rankable_again() -> None:
    store = index((make_record(), NORTH))
    store.retire(["mem-1"])

    store.upsert([make_record()], [NORTH])

    assert len(store.search(NORTH, scope=make_scope(), limit=4)) == 1


# --------------------------------------------------------------------------
# Widths and pairing
# --------------------------------------------------------------------------


def test_a_width_of_zero_means_nothing_was_measured() -> None:
    with pytest.raises(ValueError, match="positive"):
        InMemoryMemoryVectorStore().ensure_ready(0)


def test_an_index_refuses_a_second_vector_width() -> None:
    """Mixing widths would make every score in it meaningless."""
    store = index()

    with pytest.raises(ValueError, match="width"):
        store.ensure_ready(3)


def test_mismatched_records_and_vectors_are_refused() -> None:
    """A mis-paired vector is still a valid vector, so nothing downstream could
    detect it -- the symptom would be recall that is quietly wrong."""
    store = index()

    with pytest.raises(ValueError, match="positional"):
        store.upsert([make_record()], [NORTH, EAST])


def test_a_non_positive_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        index().search(NORTH, scope=make_scope(), limit=0)
