"""The harness, scored against a retriever whose answers are known in advance.

The point of these tests is the scoring rules, not retrieval quality: an
"absent" case has to score a hit when the document is correctly missing, an
"empty" case has to score a hit when nothing came back, and neither of those is
what a naive hit-rate implementation does.
"""

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from agentic_erp_assistant.eval.golden_cases import (
    DEFAULT_CASES,
    FINANCE_SCOPES,
    READER_SCOPES,
    GoldenCase,
)
from agentic_erp_assistant.eval.retrieval import evaluate
from agentic_erp_assistant.rag.chunking import Chunk
from agentic_erp_assistant.rag.ports import EmbeddingBatch
from agentic_erp_assistant.rag.retriever import RetrievalService
from agentic_erp_assistant.rag.vector_index import QdrantVectorIndex

MODEL = "text-embedding-3-small"


class ScriptedEmbedder:
    """Returns a chosen vector per query, so ranking is decided by the test."""

    model_name = MODEL

    def __init__(self, vectors: dict[str, list[float]], default: list[float]) -> None:
        self.vectors = vectors
        self.default = default
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.calls += 1
        return EmbeddingBatch(
            vectors=tuple(
                tuple(self.vectors.get(text, self.default)) for text in texts
            ),
            model=MODEL,
            prompt_tokens=len(texts),
        )


class BrokenEmbedder:
    model_name = MODEL

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        raise RuntimeError("the provider is down")


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


CHUNKS = [
    chunk("status#§1", "milestone M2 is two days late"),
    chunk("budget#p.2", "the forecast at completion is 515,000", scope="project.docs.finance.read"),
    chunk("orion#§1", "the Orion CRM migration is green", project="orion"),
]
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
# Three dimensions rather than two, so a query can be genuinely far from
# everything: with two orthogonal chunk vectors, every unit query is within 0.71
# of one of them and the floor could never be exercised.
FAR = [0.1, 0.1, 0.99]
NOWHERE = [0.0, 0.0, 0.0]


def service(embedder: object) -> RetrievalService:
    store = QdrantVectorIndex(QdrantClient(":memory:"), collection="eval")
    store.ensure_ready(3)
    store.upsert(CHUNKS, VECTORS)
    return RetrievalService.load(vector_index=store, embeddings=embedder)


def case(
    case_id: str,
    query: str,
    expectation: str,
    documents: Sequence[str] = (),
    *,
    project: str = "atlas",
    scopes: frozenset[str] = READER_SCOPES,
) -> GoldenCase:
    return GoldenCase(
        case_id=case_id,
        query=query,
        actor="priya",
        project_code=project,
        scopes=scopes,
        expectation=expectation,  # type: ignore[arg-type]
        documents=frozenset(documents),
    )


# -- scoring -----------------------------------------------------------------


def test_an_authorized_query_scores_a_hit_when_its_document_is_found() -> None:
    embedder = ScriptedEmbedder({"m2": [1.0, 0.0, 0.0]}, default=NOWHERE)
    report = evaluate(
        [case("m2", "m2", "present", ["status"])], service=service(embedder)
    )

    assert report.hit_rate == 1.0
    assert report.cases[0].retrieved_documents == ("status",)
    assert report.cases[0].best_similarity == pytest.approx(1.0)


def test_a_refusal_case_scores_a_hit_on_correct_absence() -> None:
    """The document is unreachable for this actor, which is the pass condition
    -- not the failure a naive hit-rate would record."""
    embedder = ScriptedEmbedder({"forecast": [0.0, 1.0, 0.0]}, default=NOWHERE)
    report = evaluate(
        [case("denied", "forecast", "absent", ["budget"])], service=service(embedder)
    )

    assert report.hit_rate == 1.0
    assert "budget" not in report.cases[0].retrieved_documents


def test_a_refusal_case_fails_when_the_document_does_come_back() -> None:
    embedder = ScriptedEmbedder({"forecast": [0.0, 1.0, 0.0]}, default=NOWHERE)
    report = evaluate(
        [
            case(
                "leaked",
                "forecast",
                "absent",
                ["budget"],
                scopes=FINANCE_SCOPES,
            )
        ],
        service=service(embedder),
    )

    assert report.hit_rate == 0.0
    assert report.failures[0].case_id == "leaked"


def test_a_cross_project_case_scores_a_hit_when_the_other_project_stays_out() -> None:
    embedder = ScriptedEmbedder({"orion": [1.0, 0.0, 0.0]}, default=NOWHERE)
    report = evaluate(
        [case("cross", "orion", "absent", ["orion"])], service=service(embedder)
    )

    assert report.hit_rate == 1.0


def test_an_unanswerable_case_scores_a_hit_only_on_an_empty_result() -> None:
    far = ScriptedEmbedder({"weather": FAR}, default=NOWHERE)
    report = evaluate([case("none", "weather", "empty")], service=service(far))

    assert report.hit_rate == 1.0
    assert report.cases[0].retrieved_chunks == ()
    assert report.cases[0].gated is True
    assert report.cases[0].best_similarity == pytest.approx(0.1005, abs=1e-3)

    near = ScriptedEmbedder({"weather": [1.0, 0.0, 0.0]}, default=NOWHERE)
    assert evaluate([case("none", "weather", "empty")], service=service(near)).hit_rate == 0.0


# -- the report --------------------------------------------------------------


def test_an_empty_case_list_gives_a_zero_report_not_a_crash() -> None:
    report = evaluate([], service=service(ScriptedEmbedder({}, default=[1.0, 0.0, 0.0])))

    assert report.total == 0
    assert report.hit_rate == 0.0
    assert report.average_latency_ms == 0.0
    assert report.to_dict()["cases"] == []


def test_every_aggregate_is_computed_from_the_run() -> None:
    embedder = ScriptedEmbedder({"m2": [1.0, 0.0, 0.0], "x": [0.0, 1.0, 0.0]}, default=NOWHERE)
    report = evaluate(
        [
            case("hit", "m2", "present", ["status"]),
            case("miss", "x", "present", ["status"]),
        ],
        service=service(embedder),
    )

    assert report.total == 2 and report.hits == 1
    assert report.hit_rate == 0.5
    assert report.average_latency_ms > 0.0
    assert [c.case_id for c in report.failures] == ["miss"]


def test_the_report_names_the_gap_the_floor_should_sit_in() -> None:
    embedder = ScriptedEmbedder(
        {"m2": [1.0, 0.0, 0.0], "weather": FAR}, default=NOWHERE
    )
    report = evaluate(
        [
            case("answerable", "m2", "present", ["status"]),
            case("unanswerable", "weather", "empty"),
        ],
        service=service(embedder),
    )

    gap = report.similarity_gap
    assert gap["answerable_min"] > gap["unanswerable_max"], (
        "a floor between these two numbers is a measured floor"
    )


def test_the_report_carries_the_chunk_ids_an_answer_would_cite() -> None:
    embedder = ScriptedEmbedder({"m2": [1.0, 0.0, 0.0]}, default=NOWHERE)
    report = evaluate([case("m2", "m2", "present", ["status"])], service=service(embedder))

    assert report.cases[0].retrieved_chunks == ("status#§1",)
    assert report.to_dict()["cases"][0]["retrieved_chunks"] == ["status#§1"]


def test_a_case_that_raises_is_recorded_and_the_run_continues() -> None:
    """A provider outage halfway through should produce a report, not a
    traceback and nothing."""
    report = evaluate(
        [case("a", "m2", "present", ["status"]), case("b", "x", "empty")],
        service=service(BrokenEmbedder()),
    )

    assert report.total == 2 and report.hits == 0
    assert all("RuntimeError" in (c.error or "") for c in report.cases)
    assert report.average_latency_ms == 0.0, "a case that never ran is not timed"


def test_a_non_positive_k_is_refused() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        evaluate([], service=service(ScriptedEmbedder({}, default=[1.0, 0.0, 0.0])), k=0)


# -- the golden set ----------------------------------------------------------


def test_the_golden_set_is_what_it_claims_to_be() -> None:
    assert len(DEFAULT_CASES) == 9
    assert len({case.case_id for case in DEFAULT_CASES}) == 9
    assert all(case.why.strip() for case in DEFAULT_CASES)

    by_expectation = {expectation: 0 for expectation in ("present", "absent", "empty")}
    for golden in DEFAULT_CASES:
        by_expectation[golden.expectation] += 1
    assert by_expectation == {"present": 6, "absent": 2, "empty": 1}


def test_the_golden_set_covers_every_format_in_the_corpus() -> None:
    named = {document for case in DEFAULT_CASES for document in case.documents}
    assert {
        "status-report-2026-09",  # markdown
        "architecture-notes",  # html
        "risk-register",  # csv
        "budget-summary-q3",  # pdf
    } <= named


def test_the_two_budget_cases_ask_the_identical_question() -> None:
    """Only the entitlement differs, so a pass on the refusal cannot be luck."""
    cases = {case.case_id: case for case in DEFAULT_CASES}
    allowed = cases["forecast-overrun-driver"]
    denied = cases["forecast-overrun-denied"]

    assert allowed.query == denied.query
    assert allowed.scopes != denied.scopes
    assert allowed.documents == denied.documents


def test_a_case_must_describe_the_assertion_it_makes() -> None:
    with pytest.raises(ValueError, match="naming documents"):
        case("bad", "q", "empty", ["something"])
    with pytest.raises(ValueError, match="has to name the documents"):
        case("bad", "q", "present")
