"""Running the golden cases and counting what happened.

Every number this produces is computed from a run. There is no place in this
module where a hit rate or a latency can be written down by hand, which is the
only property that makes the report evidence rather than a claim.

What it measures, and what it deliberately does not
---------------------------------------------------

Retrieval quality only: did the right document come back in the top ``k``, did
the wrong one stay out, and how long did it take. It does not judge an answer.
Answer quality needs a rubric, adversarial cases and trace replay, and building
those on top of an unmeasured retriever would be measuring the wrong layer
first -- a fluent, well-cited answer drawn from the wrong passages is wrong in a
way no answer-level score detects.

A failed case is recorded, not raised
-------------------------------------

A provider outage halfway through a nine-case run should produce a report
saying which case failed and why, not a traceback and no report. The evidence of
a bad run is still evidence.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentic_erp_assistant.eval.golden_cases import Expectation, GoldenCase
from agentic_erp_assistant.rag.retriever import RetrievalService

__all__ = ["CaseResult", "EvaluationReport", "evaluate"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseResult:
    """What one case did."""

    case_id: str
    query: str
    actor: str
    expectation: Expectation
    expected_documents: tuple[str, ...]

    hit: bool
    """Whether the case passed. ``False`` for a case that raised."""

    retrieved_chunks: tuple[str, ...]
    """The chunk ids in the top ``k``, in order -- the same identifiers an
    answer would cite, so a reader of this report can check a citation against
    it directly."""

    retrieved_documents: tuple[str, ...]
    """Those chunks' documents, deduplicated, in first-seen order."""

    latency_ms: float
    """Real wall-clock time for the search, embedding call included."""

    best_similarity: float | None
    """The closest dense candidate, before the floor was applied.

    The number that calibrates
    :data:`~agentic_erp_assistant.rag.retriever.MIN_COSINE_SIMILARITY`: the
    floor belongs between what the answerable cases score and what the
    unanswerable one scores, and this column is how that gap is read off a real
    run rather than guessed.
    """

    gated: bool
    """Whether the floor is what made this case return nothing."""

    dense_candidates: int
    lexical_candidates: int

    error: str | None = None
    """Why the case could not be run, when it could not be."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "actor": self.actor,
            "expectation": self.expectation,
            "expected_documents": list(self.expected_documents),
            "hit": self.hit,
            "retrieved_chunks": list(self.retrieved_chunks),
            "retrieved_documents": list(self.retrieved_documents),
            "latency_ms": round(self.latency_ms, 2),
            "best_similarity": (
                round(self.best_similarity, 4)
                if self.best_similarity is not None
                else None
            ),
            "gated": self.gated,
            "dense_candidates": self.dense_candidates,
            "lexical_candidates": self.lexical_candidates,
            "error": self.error,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """The run, and the numbers derived from it.

    Every aggregate is a property computed from :attr:`cases`. None of them can
    be set, which is the whole point: a report cannot disagree with the run it
    came from.
    """

    cases: tuple[CaseResult, ...]
    k: int
    embedding_model: str
    minimum_similarity: float

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def hits(self) -> int:
        return sum(1 for case in self.cases if case.hit)

    @property
    def hit_rate(self) -> float:
        """Fraction of cases that passed. ``0.0`` for an empty set, not a crash."""
        return self.hits / self.total if self.cases else 0.0

    @property
    def average_latency_ms(self) -> float:
        """Mean wall-clock latency across the cases that ran."""
        timed = [case.latency_ms for case in self.cases if case.error is None]
        return sum(timed) / len(timed) if timed else 0.0

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(case for case in self.cases if not case.hit)

    @property
    def similarity_gap(self) -> dict[str, float | None]:
        """The two numbers the floor sits between.

        ``answerable_min`` is the worst dense score among cases that expect a
        document to be found; ``unanswerable_max`` is the best among cases that
        expect nothing. A floor set between them is a measured floor. When they
        cross, no single threshold separates the two groups and that is worth
        knowing explicitly rather than discovering as a flaky case.
        """
        answerable = [
            case.best_similarity
            for case in self.cases
            if case.expectation == "present" and case.best_similarity is not None
        ]
        unanswerable = [
            case.best_similarity
            for case in self.cases
            if case.expectation == "empty" and case.best_similarity is not None
        ]
        return {
            "answerable_min": min(answerable) if answerable else None,
            "unanswerable_max": max(unanswerable) if unanswerable else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """The evidence artifact, as it is written to disk."""
        return {
            "k": self.k,
            "embedding_model": self.embedding_model,
            "minimum_similarity": self.minimum_similarity,
            "case_count": self.total,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 4),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "similarity_gap": self.similarity_gap,
            "failed_case_ids": [case.case_id for case in self.failures],
            "cases": [case.to_dict() for case in self.cases],
        }


def _run_case(case: GoldenCase, service: RetrievalService, k: int) -> CaseResult:
    retriever = service.for_context(case.context)

    started = time.perf_counter()
    try:
        outcome = retriever.search_detailed(case.query, limit=k)
    except Exception as error:  # noqa: BLE001 - a failed case, not a failed run
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.warning("case %s failed: %s", case.case_id, error)
        return CaseResult(
            case_id=case.case_id,
            query=case.query,
            actor=case.actor,
            expectation=case.expectation,
            expected_documents=tuple(sorted(case.documents)),
            hit=False,
            retrieved_chunks=(),
            retrieved_documents=(),
            latency_ms=elapsed,
            best_similarity=None,
            gated=False,
            dense_candidates=0,
            lexical_candidates=0,
            error=f"{type(error).__name__}: {error}",
        )
    elapsed = (time.perf_counter() - started) * 1000.0

    chunks = tuple(hit.chunk.chunk_id for hit in outcome.hits)
    documents: list[str] = []
    for hit in outcome.hits:
        if hit.chunk.document_id not in documents:
            documents.append(hit.chunk.document_id)

    return CaseResult(
        case_id=case.case_id,
        query=case.query,
        actor=case.actor,
        expectation=case.expectation,
        expected_documents=tuple(sorted(case.documents)),
        hit=case.scores(documents),
        retrieved_chunks=chunks,
        retrieved_documents=tuple(documents),
        latency_ms=elapsed,
        best_similarity=outcome.best_similarity,
        gated=outcome.gated,
        dense_candidates=outcome.dense_candidates,
        lexical_candidates=outcome.lexical_candidates,
    )


def evaluate(
    cases: Sequence[GoldenCase],
    *,
    service: RetrievalService,
    k: int = 5,
) -> EvaluationReport:
    """Run every case through the real retriever and score it.

    Each case gets a retriever bound to its own context, which is what lets two
    cases ask the identical question under different entitlements and be scored
    as the different cases they are.

    Args:
        cases: The golden set. An empty sequence produces a well-formed zero
            report rather than a division by zero.
        service: The loaded retrieval service. Real: the point of this harness
            is that the numbers describe what the deployed pipeline does.
        k: How many results a case sees. The same ``k`` for every case, because
            a hit rate over mixed depths is not a hit rate.

    Returns:
        An :class:`EvaluationReport`. Every aggregate on it is computed.

    Raises:
        ValueError: ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    results = tuple(_run_case(case, service, k) for case in cases)
    report = EvaluationReport(
        cases=results,
        k=k,
        embedding_model=service.embeddings.model_name,
        minimum_similarity=service.minimum_similarity,
    )
    logger.info(
        "evaluated %d case(s) at k=%d: hit rate %.2f, mean latency %.0f ms",
        report.total,
        k,
        report.hit_rate,
        report.average_latency_ms,
    )
    for failure in report.failures:
        logger.warning(
            "case %s missed: expected %s %s, got %s",
            failure.case_id,
            failure.expectation,
            sorted(failure.expected_documents) or "no results",
            list(failure.retrieved_documents) or "nothing",
        )
    return report
