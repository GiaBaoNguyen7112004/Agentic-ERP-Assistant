"""What a chunk is allowed to be: one address, one budget, no lost metadata.

The budget assertions re-measure with the real tokenizer rather than trusting a
number the chunker reports, because no such number is stored -- which is the
point. A chunk that quietly exceeded its budget would surface much later as a
provider rejecting an embeddings request, or as a prompt that no longer fits.
"""

from datetime import date
from pathlib import Path

import pytest

from agentic_erp_assistant.llm.tokenizer import count_tokens
from agentic_erp_assistant.rag.chunking import (
    CHUNK_TOKENS,
    chunk_document,
    chunk_documents,
)
from agentic_erp_assistant.rag.loaders import DocumentBlock
from agentic_erp_assistant.rag.manifest import (
    ManifestEntry,
    SourceDocument,
    load_documents,
)

MODEL = "text-embedding-3-small"


def document(*blocks: DocumentBlock, title: str = "Test Document") -> SourceDocument:
    entry = ManifestEntry(
        document_id="doc-a",
        path="a.md",
        title=title,
        document_type="status_report",
        project_code="atlas",
        required_scope="project.docs.read",
        classification="internal",
        effective_date=date(2026, 9, 1),
    )
    return SourceDocument(
        entry=entry, path=Path("a.md"), blocks=blocks, content_hash="hash-1"
    )


def block(locator: str, text: str, *path: str) -> DocumentBlock:
    return DocumentBlock(locator=locator, text=text, heading_path=path)


def tokens(text: str) -> int:
    return count_tokens(text, model=MODEL)


# -- the shipped corpus ------------------------------------------------------


def test_no_chunk_in_the_corpus_exceeds_its_budget() -> None:
    chunks = chunk_documents(load_documents(), model=MODEL)
    assert chunks
    assert all(tokens(chunk.text) <= CHUNK_TOKENS for chunk in chunks)


def test_every_chunk_in_the_corpus_has_a_unique_address() -> None:
    """Two chunks sharing a tag would make a citation resolve to whichever came
    first, and no reader could tell."""
    chunks = chunk_documents(load_documents(), model=MODEL)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_a_chunk_id_is_the_citation_tag() -> None:
    chunks = chunk_documents(load_documents(), model=MODEL)
    for chunk in chunks:
        assert chunk.chunk_id == f"{chunk.document_id}#{chunk.locator}"
        snippet = chunk.as_snippet()
        assert snippet.tag == chunk.tag == f"[{chunk.chunk_id}]"


def test_the_restricted_document_carries_its_scope_onto_every_chunk() -> None:
    chunks = chunk_documents(load_documents(), model=MODEL)
    budget = [c for c in chunks if c.document_id == "budget-summary-q3"]
    assert budget
    assert {c.required_scope for c in budget} == {"project.docs.finance.read"}
    assert {c.project_code for c in budget} == {"atlas"}


# -- addresses ---------------------------------------------------------------


def test_a_chunk_never_spans_two_addresses() -> None:
    doc = document(
        block("§1", "alpha " * 20, "1. One"),
        block("§2", "beta " * 20, "2. Two"),
    )
    chunks = chunk_document(doc, model=MODEL)
    assert [chunk.locator for chunk in chunks] == ["§1", "§2"]
    assert "beta" not in chunks[0].text


def test_a_section_that_fits_keeps_a_clean_address() -> None:
    doc = document(block("§3.2", "short body", "3. Risk", "3.2 Vendor"))
    (chunk,) = chunk_document(doc, model=MODEL)
    assert chunk.locator == "§3.2"
    assert "part" not in chunk.locator


def test_a_section_that_splits_is_numbered_by_part() -> None:
    doc = document(
        block("§1", "alpha " * 60, "1. One"),
        block("§1", "beta " * 60, "1. One"),
        block("§1", "gamma " * 60, "1. One"),
    )
    chunks = chunk_document(doc, model=MODEL, chunk_tokens=80, overlap_tokens=10)
    assert len(chunks) > 1
    assert [chunk.locator for chunk in chunks] == [
        f"§1 part {index}" for index in range(1, len(chunks) + 1)
    ]
    assert all(tokens(chunk.text) <= 80 for chunk in chunks)


# -- packing -----------------------------------------------------------------


def test_a_block_that_fits_is_never_split() -> None:
    """A table split across chunks arrives in a prompt without its header."""
    table = "\n".join(f"| row {index} | value {index} |" for index in range(8))
    doc = document(block("§1", table, "1. One"))
    (chunk,) = chunk_document(doc, model=MODEL)
    assert chunk.text.count("| row") == 8


def test_an_oversized_single_block_still_produces_bounded_chunks() -> None:
    """No chunk may exceed its budget however the source was written."""
    doc = document(block("§1", "sentence about migration exceptions. " * 200, "1. One"))
    chunks = chunk_document(doc, model=MODEL, chunk_tokens=120, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(tokens(chunk.text) <= 120 for chunk in chunks)


def test_a_boundary_block_is_carried_forward_as_overlap() -> None:
    """A fact stated across a paragraph break survives in at least one chunk."""
    doc = document(
        block("§1", "alpha " * 30, "1. One"),
        block("§1", "the boundary fact", "1. One"),
        block("§1", "gamma " * 30, "1. One"),
    )
    chunks = chunk_document(doc, model=MODEL, chunk_tokens=60, overlap_tokens=25)
    carrying = [chunk for chunk in chunks if "the boundary fact" in chunk.text]
    assert len(carrying) == 2


def test_packing_does_not_repeat_a_chunk_that_is_only_overlap() -> None:
    doc = document(
        block("§1", "alpha " * 40, "1. One"),
        block("§1", "tail", "1. One"),
    )
    chunks = chunk_document(doc, model=MODEL, chunk_tokens=60, overlap_tokens=25)
    assert len({chunk.text for chunk in chunks}) == len(chunks)


# -- the header --------------------------------------------------------------


def test_every_chunk_carries_its_document_title_and_section() -> None:
    """Two projects file the same report; the header is what separates them."""
    doc = document(
        block("§3.2", "the forecast is 515,000", "3. Budget", "3.2 Forecast"),
        title="Atlas ERP Rollout - Status Report",
    )
    (chunk,) = chunk_document(doc, model=MODEL)
    assert chunk.text.startswith("Atlas ERP Rollout - Status Report\n3. Budget > 3.2 Forecast")
    assert "the forecast is 515,000" in chunk.text


def test_the_header_is_counted_against_the_budget() -> None:
    long_title = "A very long document title repeated for the sake of counting " * 3
    doc = document(block("§1", "body " * 50, "1. One"), title=long_title)
    chunks = chunk_document(doc, model=MODEL, chunk_tokens=90, overlap_tokens=10)
    assert all(tokens(chunk.text) <= 90 for chunk in chunks)


# -- configuration mistakes fail loudly --------------------------------------


def test_an_overlap_that_fills_the_budget_is_refused() -> None:
    doc = document(block("§1", "body", "1. One"))
    with pytest.raises(ValueError, match="smaller than chunk_tokens"):
        chunk_document(doc, model=MODEL, chunk_tokens=50, overlap_tokens=50)


def test_a_header_that_leaves_no_room_is_refused() -> None:
    doc = document(block("§1", "body", "1. One"), title="title " * 40)
    with pytest.raises(ValueError, match="leaves no room"):
        chunk_document(doc, model=MODEL, chunk_tokens=20, overlap_tokens=5)
