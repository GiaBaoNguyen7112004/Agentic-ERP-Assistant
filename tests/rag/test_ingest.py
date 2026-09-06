"""Ingestion, and specifically what it refuses to pay for twice.

The reference this project follows says ingestion is network glue with no test
of its own. It is testable, and the property most worth testing is the one that
costs money if it breaks: re-running after a one-word edit must embed one
document, not the corpus. A fake embeddings client that counts calls proves that
offline, for nothing.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from agentic_erp_assistant.rag.ingest import IngestReport, ingest
from agentic_erp_assistant.rag.manifest import load_documents
from agentic_erp_assistant.rag.ports import EmbeddingBatch
from agentic_erp_assistant.rag.retriever import RetrievalService
from agentic_erp_assistant.rag.vector_index import QdrantVectorIndex

MODEL = "text-embedding-3-small"


class CountingEmbedder:
    """A fake embeddings client that records every request it is given."""

    model_name = MODEL

    def __init__(self, dimensions: int = 4) -> None:
        self.dimensions = dimensions
        self.batches: list[list[str]] = []

    @property
    def requests(self) -> int:
        return len(self.batches)

    @property
    def texts(self) -> list[str]:
        return [text for batch in self.batches for text in batch]

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.batches.append(list(texts))
        return EmbeddingBatch(
            vectors=tuple(
                tuple(
                    float(len(text) % 7 + index + axis)
                    for axis in range(self.dimensions)
                )
                for index, text in enumerate(texts)
            ),
            model=MODEL,
            prompt_tokens=sum(len(text) for text in texts),
        )


def row(document_id: str, path: str) -> dict:
    return {
        "document_id": document_id,
        "path": path,
        "title": f"Document {document_id}",
        "document_type": "status_report",
        "project_code": "atlas",
        "required_scope": "project.docs.read",
        "classification": "internal",
        "effective_date": "2026-09-01",
    }


def corpus(tmp_path: Path, files: dict[str, str], rows: list[dict] | None = None) -> Path:
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": rows
                if rows is not None
                else [row(name.split(".")[0], name) for name in files]
            }
        ),
        encoding="utf-8",
    )
    return manifest


def index() -> QdrantVectorIndex:
    return QdrantVectorIndex(QdrantClient(":memory:"), collection="test")


def run(manifest: Path, store: QdrantVectorIndex, embedder: CountingEmbedder, **kw) -> IngestReport:
    return ingest(
        load_documents(manifest), vector_index=store, embeddings=embedder, **kw
    )


DOC_A = "# A\n\n## 1. One\n\nthe finance cutover window\n"
DOC_B = "# B\n\n## 1. One\n\nthe warehouse rollout plan\n"


# -- the cold path -----------------------------------------------------------


def test_a_cold_index_embeds_everything_in_one_request(tmp_path: Path) -> None:
    """One request for N chunks is one round trip and one rate-limit slot."""
    manifest = corpus(tmp_path, {"a.md": DOC_A, "b.md": DOC_B})
    store, embedder = index(), CountingEmbedder()

    report = run(manifest, store, embedder)

    assert embedder.requests == 1
    assert report.documents_embedded == 2
    assert report.chunks_embedded == len(embedder.texts)
    assert report.dimensions == 4
    assert report.prompt_tokens > 0
    assert report.model == MODEL


def test_every_chunk_produced_is_a_vector_stored(tmp_path: Path) -> None:
    manifest = corpus(tmp_path, {"a.md": DOC_A, "b.md": DOC_B})
    store, embedder = index(), CountingEmbedder()

    report = run(manifest, store, embedder)

    assert len(store.iter_chunks()) == report.chunks_embedded


def test_the_collection_is_created_at_the_measured_width(tmp_path: Path) -> None:
    """Measured from a real reply, never from a constant."""
    manifest = corpus(tmp_path, {"a.md": DOC_A})
    store = index()

    run(manifest, store, CountingEmbedder(dimensions=4))

    store.ensure_ready(4)  # would raise VectorWidthMismatch at any other width


def test_both_indexes_are_populated_from_the_identical_chunk_list(
    tmp_path: Path,
) -> None:
    """A chunk can never exist in one index and not the other, because the
    lexical one is read back out of the vector one."""
    manifest = corpus(tmp_path, {"a.md": DOC_A, "b.md": DOC_B})
    store, embedder = index(), CountingEmbedder()
    run(manifest, store, embedder)

    service = RetrievalService.load(vector_index=store, embeddings=embedder)

    assert {c.chunk_id for c in service.lexical_index.chunks} == {
        c.chunk_id for c in store.iter_chunks()
    }


# -- what a re-run costs -----------------------------------------------------


def test_re_running_an_unchanged_corpus_embeds_nothing(tmp_path: Path) -> None:
    manifest = corpus(tmp_path, {"a.md": DOC_A, "b.md": DOC_B})
    store, embedder = index(), CountingEmbedder()
    run(manifest, store, embedder)

    report = run(manifest, store, embedder)

    assert embedder.requests == 1, "the second run made no request at all"
    assert report.documents_embedded == 0
    assert report.chunks_embedded == 0
    assert report.skipped == 2


def test_editing_one_document_bills_for_one_document(tmp_path: Path) -> None:
    manifest = corpus(tmp_path, {"a.md": DOC_A, "b.md": DOC_B})
    store, embedder = index(), CountingEmbedder()
    run(manifest, store, embedder)
    first_run_texts = len(embedder.texts)

    (tmp_path / "b.md").write_text(
        "# B\n\n## 1. One\n\nthe warehouse rollout plan, revised\n", encoding="utf-8"
    )
    report = run(manifest, store, embedder)

    assert report.documents_embedded == 1
    assert embedder.batches[-1] == [
        text for text in embedder.batches[-1] if "warehouse" in text
    ]
    assert len(embedder.texts) < first_run_texts * 2


def test_an_edit_leaves_the_other_document_in_the_index(tmp_path: Path) -> None:
    manifest = corpus(tmp_path, {"a.md": DOC_A, "b.md": DOC_B})
    store, embedder = index(), CountingEmbedder()
    run(manifest, store, embedder)

    (tmp_path / "b.md").write_text("# B\n\n## 1. One\n\nrevised\n", encoding="utf-8")
    run(manifest, store, embedder)

    assert {c.document_id for c in store.iter_chunks()} == {"a", "b"}


def test_a_re_chunk_leaves_no_renamed_remnants(tmp_path: Path) -> None:
    """Overwriting by id would strand the old part names as unreachable
    duplicates quoting text no longer in the document."""
    long_section = "# B\n\n## 1. One\n\n" + "\n\n".join(
        f"paragraph {n} about the warehouse rollout plan" for n in range(30)
    )
    manifest = corpus(tmp_path, {"b.md": long_section})
    store, embedder = index(), CountingEmbedder()
    run(manifest, store, embedder, chunk_tokens=60, overlap_tokens=10)
    before = {c.chunk_id for c in store.iter_chunks()}
    assert len(before) > 1, "the fixture has to split into several parts"

    (tmp_path / "b.md").write_text("# B\n\n## 1. One\n\nshort now\n", encoding="utf-8")
    run(manifest, store, embedder, chunk_tokens=60, overlap_tokens=10)

    after = {c.chunk_id for c in store.iter_chunks()}
    assert after == {"b#§1"}
    assert not after & (before - after)


def test_a_document_dropped_from_the_manifest_leaves_the_index(
    tmp_path: Path,
) -> None:
    """A corpus that shrank must stop answering from what was withdrawn."""
    manifest = corpus(tmp_path, {"a.md": DOC_A, "b.md": DOC_B})
    store, embedder = index(), CountingEmbedder()
    run(manifest, store, embedder)

    manifest.write_text(json.dumps({"documents": [row("a", "a.md")]}), encoding="utf-8")
    report = run(manifest, store, embedder)

    assert report.documents_removed == 1
    assert {c.document_id for c in store.iter_chunks()} == {"a"}


# -- batching ----------------------------------------------------------------


def test_a_large_corpus_is_split_into_whole_batches(tmp_path: Path) -> None:
    files = {
        f"d{n}.md": f"# D{n}\n\n## 1. One\n\nsection body {n}\n" for n in range(10)
    }
    manifest = corpus(tmp_path, files)
    store, embedder = index(), CountingEmbedder()

    report = run(manifest, store, embedder, batch_size=3)

    assert embedder.requests == report.requests > 1
    assert sum(len(batch) for batch in embedder.batches) == report.chunks_embedded
    assert len(store.iter_chunks()) == report.chunks_embedded


def test_an_impossible_batch_size_is_refused(tmp_path: Path) -> None:
    manifest = corpus(tmp_path, {"a.md": DOC_A})
    store, embedder = index(), CountingEmbedder()

    with pytest.raises(ValueError, match="must be positive"):
        run(manifest, store, embedder, batch_size=0)
    with pytest.raises(ValueError, match="single embeddings request"):
        run(manifest, store, embedder, batch_size=10_000)
    assert embedder.requests == 0


# -- the real corpus ---------------------------------------------------------


def test_the_shipped_corpus_ingests_end_to_end() -> None:
    store, embedder = index(), CountingEmbedder()

    report = ingest(
        load_documents(), vector_index=store, embeddings=embedder
    )

    assert report.documents_seen == 8
    assert report.documents_embedded == 8
    assert report.requests == 1, "the whole corpus is one request"
    assert len(store.iter_chunks()) == report.chunks_embedded
    assert {c.document_id for c in store.iter_chunks()} == {
        "status-report-2026-09",
        "sprint-13-report",
        "steering-minutes-2026-08",
        "support-policy",
        "architecture-notes",
        "risk-register",
        "budget-summary-q3",
        "orion-status-report-2026-09",
    }
