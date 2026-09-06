"""Corpus to index, in one pass, paying only for what changed.

The whole ingest-to-index path is one function so a reviewer can point at it
instead of tracing embedding calls scattered across the codebase. It is also the
only place in this package that spends money, which is why the two decisions it
makes are both about not spending it twice.

Batching is a cost decision, not an implementation detail
---------------------------------------------------------

One request carrying N chunks is one round trip and one rate-limit slot. N
requests carrying one chunk each is N of both, for identical output. The corpus
here is one request; the batch size exists so that a corpus ten times larger is
still a handful.

Re-ingestion embeds only what moved
-----------------------------------

Every stored chunk carries its document's content hash, so the index can be
asked what it already holds and the corpus compared against it. Editing one
sentence in one document and re-running should bill for one document. This is
the reason :mod:`agentic_erp_assistant.rag.manifest` hashes extracted text
rather than file bytes -- a hash that changed every time the PDF was re-rendered
would make this optimization silently do nothing.

Deleting before writing is deliberate
-------------------------------------

A changed document's old chunks are removed by document filter before the new
ones are written, rather than relying on the upsert to overwrite them.
Re-chunking a document that gained a section renames the parts after it, so
overwriting by id would leave the old names behind as unreachable duplicates --
passages that can still be retrieved and cited, quoting text that is no longer
in the document. The cost is a window where a document is absent from the index
if the process dies mid-ingest, and the fix for that is to run ingestion again,
which is cheap and idempotent.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_erp_assistant.rag.chunking import (
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    Chunk,
    chunk_documents,
)
from agentic_erp_assistant.rag.embeddings import MAX_INPUTS_PER_REQUEST
from agentic_erp_assistant.rag.manifest import SourceDocument, changed_documents
from agentic_erp_assistant.rag.ports import EmbeddingsPort, VectorIndexPort

__all__ = ["DEFAULT_BATCH_SIZE", "IngestReport", "ingest"]

logger = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 256
"""How many chunks go into one embeddings request.

Well under the provider's input ceiling, and chosen for the other limit instead:
a request also has a total-token budget, and 256 chunks at the chunk size this
project uses sits comfortably inside it. At the size of this corpus it means one
request, which is the point.
"""


@dataclass(frozen=True)
class IngestReport:
    """What one ingest actually did, in numbers a bill can be checked against."""

    documents_seen: int
    """How many the manifest listed."""

    documents_embedded: int
    """How many were new or had changed. The rest cost nothing."""

    documents_removed: int
    """How many were dropped from the index because the manifest no longer
    lists them."""

    chunks_embedded: int
    """How many chunks were sent to the provider."""

    requests: int
    """How many embeddings requests it took. One, unless the corpus is large."""

    prompt_tokens: int
    """What the provider says it billed, summed across the requests."""

    dimensions: int
    """The vector width, measured from the reply rather than assumed. ``0`` when
    nothing was embedded."""

    model: str
    """Which embedding model produced the vectors."""

    @property
    def skipped(self) -> int:
        """Documents that were already indexed under the same content."""
        return self.documents_seen - self.documents_embedded


def _batches(chunks: Sequence[Chunk], size: int) -> list[Sequence[Chunk]]:
    return [chunks[start : start + size] for start in range(0, len(chunks), size)]


def ingest(
    documents: Sequence[SourceDocument],
    *,
    vector_index: VectorIndexPort,
    embeddings: EmbeddingsPort,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestReport:
    """Chunk, embed and store everything that has changed.

    Args:
        documents: The corpus, as the manifest describes it now.
        vector_index: Where chunks and vectors go, and where the previously
            stored hashes come from.
        embeddings: What turns chunk text into vectors. Its ``model_name`` also
            sizes the chunks, because the token budget only means anything with
            respect to the model the text is being sent to.
        chunk_tokens: The chunk budget.
        overlap_tokens: How much a split section repeats.
        batch_size: Chunks per embeddings request. See
            :data:`DEFAULT_BATCH_SIZE`.

    Returns:
        An :class:`IngestReport` of what was done and what it cost.

    Raises:
        ValueError: ``batch_size`` is not positive or exceeds what one request
            may carry.
        ProviderAuthError: The embeddings provider refused definitively.
        TransientProviderError: An embeddings call failed. Nothing is retried
            here; ingestion is a script-level operation and re-running it is
            idempotent, which is a better answer than a retry loop inside the
            one function that spends money.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if batch_size > MAX_INPUTS_PER_REQUEST:
        raise ValueError(
            f"batch_size {batch_size} exceeds the {MAX_INPUTS_PER_REQUEST} a "
            f"single embeddings request may carry"
        )

    stored = dict(vector_index.document_hashes())
    current = {document.document_id for document in documents}

    # Documents the manifest no longer lists. Removed first, so a corpus that
    # shrank does not keep answering from a document nobody meant to publish.
    removed = sorted(set(stored) - current)
    if removed:
        logger.info("removing %d document(s) no longer in the manifest", len(removed))
        vector_index.delete_documents(removed)

    changed = changed_documents(documents, stored)
    if not changed:
        logger.info("nothing to embed: every document is already indexed")
        return IngestReport(
            documents_seen=len(documents),
            documents_embedded=0,
            documents_removed=len(removed),
            chunks_embedded=0,
            requests=0,
            prompt_tokens=0,
            dimensions=0,
            model=embeddings.model_name,
        )

    chunks = chunk_documents(
        changed,
        model=embeddings.model_name,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )

    vectors: list[tuple[float, ...]] = []
    prompt_tokens = 0
    dimensions = 0
    batches = _batches(chunks, batch_size)
    for number, batch in enumerate(batches, start=1):
        logger.info(
            "embedding batch %d of %d (%d chunk(s))", number, len(batches), len(batch)
        )
        result = embeddings.embed([chunk.text for chunk in batch])
        if len(result) != len(batch):
            raise ValueError(
                f"batch {number} returned {len(result)} vector(s) for "
                f"{len(batch)} chunk(s)"
            )
        if dimensions and result.dimensions != dimensions:
            raise ValueError(
                f"batch {number} returned {result.dimensions}-dimension vectors "
                f"after {dimensions}; one collection cannot hold both"
            )
        dimensions = result.dimensions
        vectors.extend(result.vectors)
        prompt_tokens += result.prompt_tokens

    # Only now, with a width measured from a real reply rather than assumed.
    vector_index.ensure_ready(dimensions)

    # Delete before writing; see the module docstring.
    vector_index.delete_documents(document.document_id for document in changed)
    vector_index.upsert(chunks, vectors)

    report = IngestReport(
        documents_seen=len(documents),
        documents_embedded=len(changed),
        documents_removed=len(removed),
        chunks_embedded=len(chunks),
        requests=len(batches),
        prompt_tokens=prompt_tokens,
        dimensions=dimensions,
        model=embeddings.model_name,
    )
    logger.info(
        "ingested %d of %d document(s): %d chunk(s) in %d request(s), "
        "%d prompt token(s) at %d dimensions",
        report.documents_embedded,
        report.documents_seen,
        report.chunks_embedded,
        report.requests,
        report.prompt_tokens,
        report.dimensions,
    )
    return report
