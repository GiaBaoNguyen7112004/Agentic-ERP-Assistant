"""Which documents exist, who may read them, and whether one has changed.

The manifest is the control plane for retrieval. It is the only place that says
a file is part of the corpus, the only place that says what entitlement reading
it requires, and the only place a reviewer has to look to audit either. The
files themselves carry content and nothing else -- no classification banner that
code reads, no folder that means "restricted", no naming convention.

That split is the point. Policy expressed in file layout is policy that moves
when somebody tidies up, and policy repeated inside the documents is policy that
disagrees with itself the first time one copy is edited. Here, adding a document
is one file plus one row, and changing who may read it is a one-line diff.

Two types, because JSON only enters at one end
----------------------------------------------

:class:`ManifestEntry` is a validated model: it is built from untrusted JSON and
every field constraint on it is a constraint on the file. :class:`SourceDocument`
is a plain frozen dataclass built afterwards, holding the entry plus the blocks
that were actually read and the hash of what they contain. Nothing deserializes
a ``SourceDocument``, so nothing needs to validate one.

Why the hash is over the text and not the bytes
-----------------------------------------------

Hashing raw bytes is simpler and would be wrong here. Re-rendering the committed
PDF from an unchanged Markdown source produces different bytes -- a PDF carries a
creation timestamp -- and a byte hash would report a document nobody edited as
changed and re-embed it. Embeddings cost money per call, so the hash has to
track what a reader would call a change: the extracted text, and the locators it
sits under. A heading renumber counts as a change, because every citation into
that document just moved.
"""

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agentic_erp_assistant.rag.loaders import (
    SUPPORTED_SUFFIXES,
    DocumentBlock,
    UnsupportedFormat,
    load_blocks,
)

__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "ManifestEntry",
    "ManifestEntryError",
    "SourceDocument",
    "changed_documents",
    "document_hashes",
    "load_documents",
]

logger = logging.getLogger(__name__)


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "documents" / "manifest.json"
)
"""The repo corpus, so a caller with nothing to say about paths still gets one.

Mirrors :data:`~agentic_erp_assistant.erp.mock.DEFAULT_DATASET_PATH`, and for the
same reason: the fixture is the corpus this project is reviewed against, and a
script that had to spell out the path would be one more place to get it wrong.
"""


class ManifestEntryError(ValueError):
    """A manifest row does not describe a document that can be indexed.

    Raised, never logged and skipped. The manifest is the statement of what is
    searchable; a corpus that quietly indexes seven of its eight documents
    answers fewer questions than its author believes it does, and the failure
    shows up as a refusal nobody can explain.
    """


class ManifestEntry(BaseModel):
    """One row of the manifest, validated as it comes off disk.

    Frozen and ``extra="forbid"``: an unmodelled field here is retrieval policy
    that arrived in the corpus without passing through review, and a row that
    could be edited after validation is a row whose checks mean nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    """Stable identity, and the ``source_id`` of every citation into this file."""

    path: str = Field(min_length=1)
    """Where the file is, relative to the manifest."""

    title: str = Field(min_length=1)
    """Human-facing name, for reports and for the evidence report."""

    document_type: str = Field(min_length=1)
    """What class of document this is. Reported on, and available to a future
    index router; never consulted when deciding access."""

    project_code: str = Field(min_length=1)
    """Which project the content belongs to. Half of the authorization check."""

    required_scope: str = Field(min_length=1)
    """The one entitlement needed to read it.

    Exactly one, and not a list. A single scope is expressible as a vector-store
    filter and auditable as a single comparison; a set would need subset logic
    that the index cannot express and that would therefore end up enforced in
    two different ways on the two search paths.
    """

    classification: str = Field(min_length=1)
    """The sensitivity label a person sees. Recorded and displayed, never the
    thing that grants or denies access -- :attr:`required_scope` is."""

    effective_date: date
    """When this version of the document took effect."""


@dataclass(frozen=True)
class SourceDocument:
    """A manifest row, the file it resolved to, and what was read out of it."""

    entry: ManifestEntry
    """The row, exactly as the manifest declared it."""

    path: Path
    """The resolved file. Guaranteed to exist -- loading proved it."""

    blocks: tuple[DocumentBlock, ...]
    """What the loader read, in document order. Never empty."""

    content_hash: str
    """SHA-256 of the normalized text. See the module docstring for why not the
    bytes."""

    @property
    def document_id(self) -> str:
        return self.entry.document_id

    @property
    def project_code(self) -> str:
        return self.entry.project_code

    @property
    def required_scope(self) -> str:
        return self.entry.required_scope

    @property
    def text(self) -> str:
        """The whole document as one string, blocks in order.

        For hashing, for reporting, and for a reviewer who wants to see what was
        actually extracted from a PDF. Retrieval never uses it: chunking works
        from :attr:`blocks`, which still carry their addresses.
        """
        return "\n\n".join(block.text for block in self.blocks)


def _normalized(blocks: Sequence[DocumentBlock]) -> str:
    """The exact string the content hash is taken over.

    Locators are included deliberately. If a heading is renumbered, the text is
    unchanged but every citation into the document has moved, and re-embedding
    is the correct response.
    """
    return "\n".join(f"{block.locator}\n{block.text}" for block in blocks)


def _hash(blocks: Sequence[DocumentBlock]) -> str:
    return hashlib.sha256(_normalized(blocks).encode("utf-8")).hexdigest()


def load_documents(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[SourceDocument, ...]:
    """Read every manifest row and the file it points at.

    Args:
        manifest_path: The manifest. Paths inside it resolve relative to its own
            directory, so a corpus can be moved without editing every row.

    Returns:
        One :class:`SourceDocument` per row, in manifest order.

    Raises:
        ManifestEntryError: A row is invalid, names a duplicate id, points
            outside the corpus directory, names a missing file, names a format
            with no loader, or resolves to a document with no readable text.
        OSError: The manifest itself cannot be read.
        json.JSONDecodeError: The manifest is not JSON.
    """
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ManifestEntryError(
            f"{manifest_path}: expected a non-empty 'documents' array"
        )

    documents: list[SourceDocument] = []
    seen: set[str] = set()
    for position, row in enumerate(rows, start=1):
        entry = _entry(row, position, manifest_path)

        if entry.document_id in seen:
            raise ManifestEntryError(
                f"{manifest_path}: document_id {entry.document_id!r} is declared "
                f"twice; one of the two policies would silently win"
            )
        seen.add(entry.document_id)

        documents.append(_document(entry, root))

    logger.info("loaded %d document(s) from %s", len(documents), manifest_path)
    return tuple(documents)


def _entry(row: object, position: int, manifest_path: Path) -> ManifestEntry:
    """Validate one row, naming where it is when it fails."""
    if not isinstance(row, dict):
        raise ManifestEntryError(
            f"{manifest_path}: row {position} is {type(row).__name__}, expected an object"
        )
    try:
        return ManifestEntry.model_validate(row)
    except ValueError as error:
        raise ManifestEntryError(f"{manifest_path}: row {position}: {error}") from error


def _document(entry: ManifestEntry, root: Path) -> SourceDocument:
    """Resolve one entry to a file and read it, or say exactly what is wrong."""
    path = (root / entry.path).resolve()

    if not path.is_relative_to(root):
        # A row is data, and this one would have reached outside the corpus for
        # its content. Nothing legitimate needs that, and something that does is
        # asking to have a file from elsewhere on the machine indexed and quoted.
        raise ManifestEntryError(
            f"{entry.document_id}: path {entry.path!r} resolves outside the "
            f"corpus directory {root}"
        )

    if not path.is_file():
        raise ManifestEntryError(
            f"{entry.document_id}: no file at {path}. A manifest row is a "
            f"statement that this document is searchable; skipping it would "
            f"make the corpus quietly smaller than it claims to be."
        )

    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ManifestEntryError(
            f"{entry.document_id}: {path.name} has no loader; supported "
            f"formats are {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    try:
        blocks = load_blocks(path)
    except UnsupportedFormat as error:  # pragma: no cover - guarded just above
        raise ManifestEntryError(f"{entry.document_id}: {error}") from error

    if not blocks:
        raise ManifestEntryError(
            f"{entry.document_id}: {path.name} produced no readable text. An "
            f"empty document contributes nothing to retrieval and would be an "
            f"index entry that can never be cited."
        )

    return SourceDocument(
        entry=entry, path=path, blocks=blocks, content_hash=_hash(blocks)
    )


def document_hashes(documents: Iterable[SourceDocument]) -> dict[str, str]:
    """``{document_id: content_hash}`` for a loaded corpus."""
    return {document.document_id: document.content_hash for document in documents}


def changed_documents(
    documents: Sequence[SourceDocument],
    previous_hashes: Mapping[str, str],
) -> tuple[SourceDocument, ...]:
    """The documents worth re-chunking and re-embedding, in manifest order.

    This is a cost control, not an optimization. Embedding is billed per call
    against real money, and re-running ingestion after a one-word edit should
    pay for one document rather than for the corpus. A document is worth
    re-embedding when it is new to the index or when its hash has moved;
    everything else is already stored under a hash that still describes it.

    Deleting a document from the index is deliberately not this function's job.
    It answers "what has to be embedded", and an id present in
    ``previous_hashes`` but absent from ``documents`` needs a delete, not an
    embedding -- a different decision, made where the index is written.

    Args:
        documents: The corpus as it is now.
        previous_hashes: ``{document_id: content_hash}`` as last indexed. An
            empty mapping means a cold index, so everything is returned.

    Returns:
        The subset that has to be re-embedded.
    """
    changed = tuple(
        document
        for document in documents
        if previous_hashes.get(document.document_id) != document.content_hash
    )
    logger.info(
        "%d of %d document(s) changed since the last ingest",
        len(changed),
        len(documents),
    )
    return changed
