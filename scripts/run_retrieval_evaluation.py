"""Run the golden cases against the real pipeline and write the evidence.

    docker compose up -d qdrant
    uv run python scripts/ingest_documents.py
    uv run python scripts/run_retrieval_evaluation.py

Everything this writes is measured. There is no argument that lets a number be
supplied, and the report carries the commit it was produced at, so a reviewer
can re-run it and compare rather than take it on trust.

It costs one embeddings call per case -- nine short queries -- and nothing else.
No chat model is involved: this measures retrieval, and an answer-level harness
built on an unmeasured retriever would be measuring the wrong layer first.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from agentic_erp_assistant.eval.golden_cases import DEFAULT_CASES
from agentic_erp_assistant.eval.retrieval import EvaluationReport, evaluate
from agentic_erp_assistant.llm.ports import LLMClientError
from agentic_erp_assistant.rag.retriever import RetrievalService
from agentic_erp_assistant.rag.vector_index import VectorIndexError

logger = logging.getLogger("run_retrieval_evaluation")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = REPO_ROOT / "evidence" / "rag" / "retrieval-report.json"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5, help="results per case")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_REPORT_PATH, help="where to write"
    )
    parser.add_argument(
        "--collection", default=None, help="override QDRANT_COLLECTION"
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help=(
            "exit non-zero below this hit rate. Off by default: a run records "
            "what happened, and turning it into a gate is a separate decision"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def commit() -> str | None:
    """The commit the report was produced at, when there is one.

    Written into the report because a hit rate without the code that produced it
    is a number nobody can reproduce or argue with.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def summarize(report: EvaluationReport) -> None:
    """Log the per-case table a reader wants before opening the JSON."""
    logger.info("")
    logger.info(
        "%-28s %-10s %-5s %-9s %s",
        "case",
        "expects",
        "hit",
        "best cos",
        "documents",
    )
    for case in report.cases:
        logger.info(
            "%-28s %-10s %-5s %-9s %s",
            case.case_id,
            case.expectation,
            "yes" if case.hit else "NO",
            f"{case.best_similarity:.3f}" if case.best_similarity is not None else "-",
            ", ".join(case.retrieved_documents) or "(nothing)",
        )

    gap = report.similarity_gap
    logger.info("")
    logger.info(
        "hit rate %.2f (%d of %d), mean latency %.0f ms, k=%d, model %s",
        report.hit_rate,
        report.hits,
        report.total,
        report.average_latency_ms,
        report.k,
        report.embedding_model,
    )
    logger.info(
        "similarity floor is %.2f; answerable cases bottom out at %s and the "
        "unanswerable case tops out at %s",
        report.minimum_similarity,
        f"{gap['answerable_min']:.3f}" if gap["answerable_min"] is not None else "-",
        f"{gap['unanswerable_max']:.3f}"
        if gap["unanswerable_max"] is not None
        else "-",
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(message)s",
    )

    try:
        service = RetrievalService.from_env()
    except (LLMClientError, VectorIndexError) as error:
        logger.error("%s: %s", type(error).__name__, error)
        return 3

    try:
        indexed = len(service.lexical_index)
        if not indexed:
            logger.error(
                "the collection is empty; run scripts/ingest_documents.py first"
            )
            return 4

        report = evaluate(list(DEFAULT_CASES), service=service, k=arguments.k)
    finally:
        service.close()

    summarize(report)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit(),
        "indexed_chunks": indexed,
        **report.to_dict(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    logger.info("")
    logger.info("wrote %s", arguments.output)

    if arguments.fail_under is not None and report.hit_rate < arguments.fail_under:
        logger.error(
            "hit rate %.2f is below the required %.2f",
            report.hit_rate,
            arguments.fail_under,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
