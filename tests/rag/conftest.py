"""Shared fixtures for the retrieval tests.

Loading and chunking the real corpus is the honest input for most of these
tests, and it costs a PDF parse and a few hundred tiktoken encodes. Done once
per session it is negligible; done per test it is the slowest thing in the
suite for no extra coverage, since nothing here mutates the chunks -- they are
frozen models.
"""

import pytest

from agentic_erp_assistant.rag.chunking import Chunk, chunk_documents
from agentic_erp_assistant.rag.manifest import SourceDocument, load_documents

EMBEDDING_MODEL = "text-embedding-3-small"
"""The model whose tokenizer sizes the chunks in these tests.

Named here rather than read from the environment: a test that changed shape
depending on a developer's .env would be a test nobody could reproduce.
"""


@pytest.fixture(scope="session")
def corpus() -> tuple[SourceDocument, ...]:
    """The eight committed documents, loaded through the real manifest."""
    return load_documents()


@pytest.fixture(scope="session")
def corpus_chunks(corpus: tuple[SourceDocument, ...]) -> tuple[Chunk, ...]:
    """The corpus as the indexes will see it."""
    return chunk_documents(corpus, model=EMBEDDING_MODEL)
