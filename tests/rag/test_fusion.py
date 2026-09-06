"""Reciprocal Rank Fusion: agreement wins, and one list is never enough to lose.

The properties asserted here are the reasons RRF was chosen over a weighted sum
of raw scores. They are all statements about *order*, because order is the only
thing the two lists genuinely share.
"""

import pytest

from agentic_erp_assistant.rag.chunking import Chunk
from agentic_erp_assistant.rag.fusion import RRF_K, reciprocal_rank_fusion
from agentic_erp_assistant.rag.ports import ScoredChunk

VECTOR, LEXICAL = "vector", "lexical"


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("#", 1)[0],
        locator=chunk_id.split("#", 1)[-1],
        text="a passage",
        title="Document",
        document_type="status_report",
        project_code="atlas",
        required_scope="project.docs.read",
        classification="internal",
        content_hash="hash-1",
        position=0,
    )


def ranked(*pairs: tuple[str, float]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=chunk(chunk_id), score=score) for chunk_id, score in pairs]


def ids(hits) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


def test_first_in_both_lists_outranks_first_in_only_one() -> None:
    hits = reciprocal_rank_fusion(
        {
            VECTOR: ranked(("a", 0.9), ("b", 0.8)),
            LEXICAL: ranked(("b", 12.0), ("c", 11.0)),
        }
    )

    assert ids(hits)[0] == "b"


def test_a_chunk_found_by_only_one_half_still_appears() -> None:
    """The entire point of running two halves."""
    hits = reciprocal_rank_fusion(
        {VECTOR: ranked(("a", 0.9)), LEXICAL: ranked(("z", 30.0))}
    )

    assert set(ids(hits)) == {"a", "z"}


def test_incomparable_scales_do_not_decide_the_order() -> None:
    """A weighted sum would have let the BM25 number, which is unbounded, drown
    the cosine one. Only rank position is used, so it cannot."""
    hits = reciprocal_rank_fusion(
        {
            VECTOR: ranked(("a", 0.99), ("b", 0.98)),
            LEXICAL: ranked(("b", 400.0), ("a", 399.0)),
        }
    )

    # a is 1st and 2nd, b is 2nd and 1st: the same total, broken by chunk id.
    assert ids(hits) == ["a", "b"]
    assert hits[0].score == pytest.approx(hits[1].score)


def test_each_hit_records_where_every_list_put_it() -> None:
    """"It came back because both halves ranked it second" is an explanation;
    a lone fused float is not."""
    hits = reciprocal_rank_fusion(
        {
            VECTOR: ranked(("a", 0.9), ("b", 0.5)),
            LEXICAL: ranked(("b", 12.0)),
        }
    )
    by_id = {hit.chunk.chunk_id: hit for hit in hits}

    assert by_id["a"].ranks == {VECTOR: 1}
    assert by_id["a"].scores == {VECTOR: 0.9}
    assert by_id["a"].found_by == (VECTOR,)
    assert by_id["b"].ranks == {VECTOR: 2, LEXICAL: 1}
    assert by_id["b"].found_by == (LEXICAL, VECTOR)


def test_the_score_is_the_sum_of_reciprocal_ranks() -> None:
    (hit,) = reciprocal_rank_fusion({VECTOR: ranked(("a", 0.9))})
    assert hit.score == pytest.approx(1.0 / (RRF_K + 1))

    (both,) = reciprocal_rank_fusion(
        {VECTOR: ranked(("a", 0.9)), LEXICAL: ranked(("a", 3.0))}
    )
    assert both.score == pytest.approx(2.0 / (RRF_K + 1))


def test_the_same_inputs_always_produce_the_same_order() -> None:
    """Retrieval is a computation, not a source of nondeterminism."""
    rankings = {
        VECTOR: ranked(("c", 0.5), ("a", 0.5), ("b", 0.5)),
        LEXICAL: ranked(("b", 1.0), ("c", 1.0), ("a", 1.0)),
    }
    assert ids(reciprocal_rank_fusion(rankings)) == ids(
        reciprocal_rank_fusion(rankings)
    )


def test_ties_break_on_chunk_id() -> None:
    hits = reciprocal_rank_fusion({VECTOR: ranked(("b", 1.0)), LEXICAL: ranked(("a", 1.0))})
    assert ids(hits) == ["a", "b"]


def test_the_limit_truncates_after_fusion_not_before() -> None:
    hits = reciprocal_rank_fusion(
        {
            VECTOR: ranked(("a", 0.9), ("b", 0.8), ("c", 0.7)),
            LEXICAL: ranked(("c", 9.0)),
        },
        limit=2,
    )
    assert ids(hits) == ["c", "a"]


def test_a_smaller_k_sharpens_the_preference_for_rank_one() -> None:
    rankings = {
        VECTOR: ranked(("a", 0.9), ("b", 0.8)),
        LEXICAL: ranked(("b", 9.0), ("a", 8.0)),
    }
    sharp = reciprocal_rank_fusion(rankings, k=0)
    assert sharp[0].score == pytest.approx(1.0 + 0.5)


def test_empty_input_fuses_to_nothing() -> None:
    assert reciprocal_rank_fusion({}) == ()
    assert reciprocal_rank_fusion({VECTOR: [], LEXICAL: []}) == ()


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="k must not be negative"):
        reciprocal_rank_fusion({}, k=-1)
    with pytest.raises(ValueError, match="limit must be positive"):
        reciprocal_rank_fusion({}, limit=0)
