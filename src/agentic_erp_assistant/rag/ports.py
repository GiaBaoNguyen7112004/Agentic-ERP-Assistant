"""The replaceable pieces of retrieval, declared as shapes.

Two things in this pipeline are somebody else's software: whatever turns text
into vectors, and whatever stores those vectors and searches them. Both are
named here as Protocols so the rest of ``rag/`` depends on a shape, and the
concrete OpenAI and Qdrant code sits behind them -- the same arrangement
:mod:`agentic_erp_assistant.llm.ports` has with its vendor adapters, and for the
same reason: a test gets a ten-line fake instead of a network.

The failure types are reused, not redeclared
--------------------------------------------

An embeddings client is a provider client, so it raises the three errors
``llm.ports`` already declares. That is not laziness about naming: the retry
engine in :mod:`agentic_erp_assistant.llm.retry` retries exactly
:class:`TransientProviderError`, so an embeddings call reusing that type is
retryable by the existing engine with no new code and no second taxonomy for a
caller to learn. A parallel ``TransientEmbeddingsError`` would have meant either
a second retry path or a translation layer between two names for one idea.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentic_erp_assistant.llm.ports import (
    ClientConfigurationError,
    LLMClientError,
    ProviderAuthError,
    TransientProviderError,
)
from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.rag.chunking import Chunk

__all__ = [
    "ClientConfigurationError",
    "EmbeddingBatch",
    "EmbeddingsPort",
    "LLMClientError",
    "ProviderAuthError",
    "ScoredChunk",
    "TransientProviderError",
    "VectorIndexPort",
]


@dataclass(frozen=True)
class EmbeddingBatch:
    """The vectors for one embeddings request, and what it cost.

    ``dimensions`` is derived rather than stored, for the reason
    ``context/candidate.py`` refuses a stored token count: a number sitting
    beside the data it describes is a number that can disagree with it. Here the
    consequence would be specific and bad -- a collection created at the wrong
    width, accepting vectors it should have rejected.
    """

    vectors: tuple[tuple[float, ...], ...]
    """One vector per input text, in the order the inputs were given."""

    model: str
    """The model the provider says served the request."""

    prompt_tokens: int
    """What the provider says it billed. The invoice, not an estimate."""

    @property
    def dimensions(self) -> int:
        """The width of these vectors, or ``0`` for an empty batch."""
        return len(self.vectors[0]) if self.vectors else 0

    def __len__(self) -> int:
        return len(self.vectors)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk a search returned, with the score that put it there.

    The score exists here and deliberately not on
    :class:`~agentic_erp_assistant.state.evidence.EvidenceSnippet`, which
    documents why: a number nothing branches on, riding inside the object that
    goes into a prompt, is telemetry pretending to be evidence. This is the
    "ranked wrapper at the rag boundary" that docstring anticipates -- the
    fusion step needs ranks, and the evaluator needs scores.
    """

    chunk: Chunk
    score: float


@runtime_checkable
class EmbeddingsPort(Protocol):
    """What the pipeline assumes about anything that turns text into vectors."""

    model_name: str
    """The embedding model this client is bound to. Recorded in every report,
    because a vector is only comparable to another vector from the same model."""

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed every text in one request, preserving input order.

        Order is part of the contract, not a convenience. The caller pairs the
        returned vectors with the chunks it sent positionally, so a client that
        returned them in completion order would attach every passage to the
        wrong vector -- and nothing downstream could detect it, because a
        mismatched index is still a perfectly valid vector.

        Raises:
            ProviderAuthError: A definitive rejection.
            TransientProviderError: A failure a later attempt may survive.
            ValueError: The input is unusable -- a blank text, or more inputs
                than one request may carry.
        """
        ...


@runtime_checkable
class VectorIndexPort(Protocol):
    """What the pipeline assumes about a store of chunk vectors.

    Shaped after what this project needs rather than after what Qdrant offers,
    so a different backend is a constructor change. The two methods that look
    like implementation detail -- :meth:`document_hashes` and
    :meth:`iter_chunks` -- are here because both express decisions the pipeline
    makes rather than storage trivia: what may be skipped on a re-ingest, and
    where the lexical index gets its documents from.
    """

    def ensure_ready(self, dimensions: int) -> None:
        """Make the store ready to accept vectors of this width.

        Called with a width measured from a real embedding batch, never from a
        constant. A store already holding vectors of a different width must
        refuse rather than accept a mixture, because a collection with two
        vector widths in it is one whose similarity scores mean nothing.
        """
        ...

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Store chunks and their vectors, replacing any with the same ids."""
        ...

    def delete_documents(self, document_ids: Iterable[str]) -> None:
        """Remove every chunk belonging to these documents."""
        ...

    def document_hashes(self) -> Mapping[str, str]:
        """``{document_id: content_hash}`` for everything currently stored.

        What makes an incremental re-ingest possible: the corpus is compared
        against this, and only what moved is embedded again.
        """
        ...

    def iter_chunks(self) -> Sequence[Chunk]:
        """Every stored chunk.

        The lexical index is built from this rather than from a second pass over
        the corpus, so a chunk cannot exist in one index and not the other.
        """
        ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        context: RetrievalContext,
        limit: int,
    ) -> tuple[ScoredChunk, ...]:
        """The nearest chunks this context may read, best first.

        The context is not optional and is applied before ranking, never after.
        See :mod:`agentic_erp_assistant.rag.access`.
        """
        ...
