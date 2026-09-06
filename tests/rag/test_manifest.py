"""The manifest is the statement of what is searchable, so it fails loudly.

Every test here is about a way the corpus could quietly become smaller or more
permissive than the manifest claims: a row that names a missing file, a row that
reaches outside the corpus, two rows claiming one identity, or a re-ingest that
decides nothing changed when something did.
"""

import json
from pathlib import Path

import pytest

from agentic_erp_assistant.rag.manifest import (
    DEFAULT_MANIFEST_PATH,
    ManifestEntryError,
    changed_documents,
    document_hashes,
    load_documents,
)

ROW = {
    "document_id": "doc-a",
    "path": "a.md",
    "title": "Document A",
    "document_type": "status_report",
    "project_code": "atlas",
    "required_scope": "project.docs.read",
    "classification": "internal",
    "effective_date": "2026-09-01",
}


def build(tmp_path: Path, rows: list[dict], files: dict[str, str]) -> Path:
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": rows}), encoding="utf-8")
    return manifest


# -- the shipped corpus ------------------------------------------------------


def test_every_shipped_row_resolves_to_a_readable_document() -> None:
    documents = load_documents(DEFAULT_MANIFEST_PATH)
    assert len(documents) == 8
    assert all(document.blocks for document in documents)
    assert all(document.text.strip() for document in documents)
    assert {document.entry.path.rsplit(".", 1)[-1] for document in documents} == {
        "md",
        "html",
        "csv",
        "pdf",
    }


def test_the_corpus_carries_both_denial_cases() -> None:
    """Policy is provable against real documents, not only against fixtures."""
    documents = {doc.document_id: doc for doc in load_documents()}
    assert documents["budget-summary-q3"].required_scope == "project.docs.finance.read"
    assert documents["orion-status-report-2026-09"].project_code == "orion"


def test_the_same_manifest_hashes_the_same_way_every_time() -> None:
    first = document_hashes(load_documents(DEFAULT_MANIFEST_PATH))
    second = document_hashes(load_documents(DEFAULT_MANIFEST_PATH))
    assert first == second
    assert len(set(first.values())) == len(first)


# -- rows that must not be tolerated -----------------------------------------


def test_a_missing_file_raises_rather_than_skipping_the_document(
    tmp_path: Path,
) -> None:
    manifest = build(tmp_path, [ROW], {})
    with pytest.raises(ManifestEntryError, match="no file at"):
        load_documents(manifest)


def test_a_duplicate_document_id_is_refused(tmp_path: Path) -> None:
    manifest = build(tmp_path, [ROW, dict(ROW)], {"a.md": "# T\n\n## 1. S\n\nbody\n"})
    with pytest.raises(ManifestEntryError, match="declared twice"):
        load_documents(manifest)


def test_a_path_escaping_the_corpus_is_refused(tmp_path: Path) -> None:
    """A manifest row is data. This one would have indexed a file from elsewhere."""
    (tmp_path.parent / "outside.md").write_text("secret", encoding="utf-8")
    manifest = build(tmp_path, [{**ROW, "path": "../outside.md"}], {})
    with pytest.raises(ManifestEntryError, match="outside the corpus"):
        load_documents(manifest)


def test_an_unsupported_format_is_refused(tmp_path: Path) -> None:
    manifest = build(tmp_path, [{**ROW, "path": "a.rtf"}], {"a.rtf": "body"})
    with pytest.raises(ManifestEntryError, match="no loader"):
        load_documents(manifest)


def test_a_document_with_no_readable_text_is_refused(tmp_path: Path) -> None:
    manifest = build(tmp_path, [ROW], {"a.md": "\n\n   \n"})
    with pytest.raises(ManifestEntryError, match="no readable text"):
        load_documents(manifest)


def test_a_row_missing_its_scope_is_refused(tmp_path: Path) -> None:
    row = {name: value for name, value in ROW.items() if name != "required_scope"}
    manifest = build(tmp_path, [row], {"a.md": "# T\n\n## 1. S\n\nbody\n"})
    with pytest.raises(ManifestEntryError, match="required_scope"):
        load_documents(manifest)


def test_an_unmodelled_field_is_refused(tmp_path: Path) -> None:
    """Retrieval policy that arrived in the corpus without passing review."""
    manifest = build(
        tmp_path, [{**ROW, "public": True}], {"a.md": "# T\n\n## 1. S\n\nbody\n"}
    )
    with pytest.raises(ManifestEntryError, match="public"):
        load_documents(manifest)


# -- what re-ingestion is allowed to skip ------------------------------------


def test_only_the_edited_document_is_reported_as_changed(tmp_path: Path) -> None:
    rows = [ROW, {**ROW, "document_id": "doc-b", "path": "b.md"}]
    files = {
        "a.md": "# A\n\n## 1. S\n\nalpha\n",
        "b.md": "# B\n\n## 1. S\n\nbeta\n",
    }
    manifest = build(tmp_path, rows, files)

    before = load_documents(manifest)
    assert changed_documents(before, {}) == before, "a cold index re-embeds everything"

    stored = document_hashes(before)
    assert changed_documents(before, stored) == ()

    (tmp_path / "b.md").write_text("# B\n\n## 1. S\n\nbeta and more\n", encoding="utf-8")
    after = load_documents(manifest)
    changed = changed_documents(after, stored)
    assert [document.document_id for document in changed] == ["doc-b"]


def test_renumbering_a_heading_counts_as_a_change(tmp_path: Path) -> None:
    """The words are identical and every citation into the document has moved."""
    manifest = build(tmp_path, [ROW], {"a.md": "# A\n\n## 3. S\n\nalpha\n"})
    stored = document_hashes(load_documents(manifest))

    (tmp_path / "a.md").write_text("# A\n\n## 4. S\n\nalpha\n", encoding="utf-8")
    assert changed_documents(load_documents(manifest), stored)
