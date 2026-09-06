"""BM25: that it is really BM25, and that it cannot see past access control.

The scoring tests are written so the expected order follows from the formula
rather than from a recorded number -- a rare term beats a common one, a short
chunk beats a long one with the same match -- because a test that pins a float
proves only that the code still does whatever it did.
"""

import pytest

from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.rag.chunking import Chunk
from agentic_erp_assistant.rag.lexical import BM25Index, tokenize
from agentic_erp_assistant.rag.ports import ScoredChunk

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
    document: str | None = None,
    project: str = "atlas",
    scope: str = "project.docs.read",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document or chunk_id.split("#", 1)[0],
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


def ids(hits: tuple[ScoredChunk, ...]) -> list[str]:
    return [hit.chunk.chunk_id for hit in hits]


# -- the tokenizer -----------------------------------------------------------


def test_identifiers_survive_tokenization_whole() -> None:
    """Splitting R-1 into "r" and "1" turns the most specific term in a question
    into two of the least specific."""
    assert tokenize("Risk R-1 slips to 2026-09-11, costing 515,000 USD in 3.2") == [
        "risk",
        "r-1",
        "slips",
        "to",
        "2026-09-11",
        "costing",
        "515,000",
        "usd",
        "in",
        "3.2",
    ]


# -- the formula -------------------------------------------------------------


def test_a_rare_term_outranks_a_common_one() -> None:
    """The whole reason this is BM25 and not a word-overlap ratio."""
    common = [chunk(f"doc-a#§{n}", "the migration is the migration") for n in range(9)]
    index = BM25Index(
        [*common, chunk("doc-b#§1", "the migration exceeded its cutover window")]
    )

    hits = index.search("migration cutover", context=READER, limit=3)

    assert ids(hits)[0] == "doc-b#§1"


def test_more_matching_rare_terms_beat_fewer() -> None:
    index = BM25Index(
        [
            chunk("doc-a#§1", "reconciliation exceptions blocked the cutover"),
            chunk("doc-b#§1", "reconciliation was discussed"),
            chunk("doc-c#§1", "the committee met"),
        ]
    )

    hits = index.search(
        "reconciliation exceptions cutover", context=READER, limit=3
    )

    assert ids(hits) == ["doc-a#§1", "doc-b#§1"]


def test_a_shorter_chunk_wins_on_an_equal_match() -> None:
    """Length normalization, which is what ``b`` is for."""
    index = BM25Index(
        [
            chunk("doc-a#§1", "hypercare staffing"),
            chunk("doc-b#§1", "hypercare staffing " + "filler words here " * 20),
        ]
    )

    assert ids(index.search("hypercare staffing", context=READER, limit=2))[0] == (
        "doc-a#§1"
    )


def test_a_term_in_every_chunk_never_subtracts_from_a_score() -> None:
    """The classic Robertson idf goes negative above 50% document frequency,
    which would let a chunk rank higher for *not* containing a query term."""
    index = BM25Index(
        [
            chunk("doc-a#§1", "the the the budget"),
            chunk("doc-b#§1", "the the the the the"),
        ]
    )

    hits = index.search("the", context=READER, limit=2)

    assert hits
    assert all(hit.score > 0 for hit in hits)


def test_a_query_matching_nothing_returns_nothing() -> None:
    """"No lexical evidence" is a real answer; dressing it up as a weak hit is
    how a refusal turns into a guess."""
    index = BM25Index([chunk("doc-a#§1", "the finance cutover")])

    assert index.search("quantum entanglement", context=READER, limit=5) == ()
    assert index.search("   ", context=READER, limit=5) == ()


def test_the_same_query_twice_gives_the_same_order() -> None:
    index = BM25Index(
        [chunk(f"doc-{letter}#§1", "identical text here") for letter in "abcdef"]
    )

    first = ids(index.search("identical text", context=READER, limit=6))
    assert first == ids(index.search("identical text", context=READER, limit=6))
    assert first == sorted(first), "ties break on chunk id, so the order is total"


def test_a_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        BM25Index([]).search("anything", context=READER, limit=0)


def test_an_empty_index_returns_nothing() -> None:
    assert BM25Index([]).search("anything", context=READER, limit=5) == ()


# -- access control ----------------------------------------------------------


def test_the_lexical_path_enforces_access_the_same_way_the_vector_path_does() -> None:
    index = BM25Index(
        [
            chunk("doc-a#§1", "the approved budget is 480,000"),
            chunk(
                "doc-b#p.1",
                "the approved budget is 480,000",
                scope="project.docs.finance.read",
            ),
            chunk("doc-c#§1", "the approved budget is 480,000", project="orion"),
        ]
    )

    assert ids(index.search("approved budget", context=READER, limit=10)) == [
        "doc-a#§1"
    ]
    assert ids(index.search("approved budget", context=FINANCE, limit=10)) == [
        "doc-a#§1",
        "doc-b#p.1",
    ]


def test_a_restricted_chunk_does_not_influence_ranking_through_statistics() -> None:
    """The subtle half. If document frequency were computed over the whole
    corpus, a term common in restricted documents would be scored as ordinary
    for a reader who cannot see them, and the restricted document would have
    shaped a ranking it never appeared in.
    """
    visible = [
        chunk("doc-a#§1", "contingency drawdown"),
        chunk("doc-b#§1", "contingency was discussed"),
    ]
    restricted = [
        chunk(f"doc-x#p.{n}", "contingency contingency contingency",
              scope="project.docs.finance.read")
        for n in range(1, 12)
    ]

    without = BM25Index(visible).search("contingency", context=READER, limit=5)
    with_hidden = BM25Index([*visible, *restricted]).search(
        "contingency", context=READER, limit=5
    )

    assert [hit.chunk.chunk_id for hit in without] == [
        hit.chunk.chunk_id for hit in with_hidden
    ]
    assert [hit.score for hit in without] == pytest.approx(
        [hit.score for hit in with_hidden]
    )


def test_statistics_are_computed_per_context_not_once() -> None:
    """Two readers of the same index see different corpora, so they must not
    share a cached view of it."""
    index = BM25Index(
        [
            chunk("doc-a#§1", "forecast overrun"),
            chunk("doc-b#p.1", "forecast overrun", scope="project.docs.finance.read"),
        ]
    )

    assert len(index.search("forecast", context=READER, limit=5)) == 1
    assert len(index.search("forecast", context=FINANCE, limit=5)) == 2
    assert len(index.search("forecast", context=READER, limit=5)) == 1


# -- against the real corpus -------------------------------------------------


def test_an_identifier_question_finds_its_row(
    corpus_chunks: tuple[Chunk, ...],
) -> None:
    """What the lexical half exists for: an exact string a dense index blurs."""
    index = BM25Index(corpus_chunks)

    hits = index.search("risk R-2 owner", context=READER, limit=3)

    assert ids(hits)[0] == "risk-register#row R-2"


def test_the_restricted_budget_pdf_is_absent_for_an_ordinary_reader(
    corpus_chunks: tuple[Chunk, ...],
) -> None:
    index = BM25Index(corpus_chunks)

    hits = index.search(
        "forecast at completion contingency drawdown", context=READER, limit=10
    )

    assert hits
    assert all(hit.chunk.document_id != "budget-summary-q3" for hit in hits)
    assert any(
        hit.chunk.document_id == "budget-summary-q3"
        for hit in index.search(
            "forecast at completion contingency drawdown",
            context=FINANCE,
            limit=10,
        )
    )
