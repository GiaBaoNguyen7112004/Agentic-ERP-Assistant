"""Shared fixtures for the retrieval tests.

Loading and chunking the real corpus is the honest input for most of these
tests, and it costs a PDF parse and a few hundred tiktoken encodes. Done once
per session it is negligible; done per test it is the slowest thing in the
suite for no extra coverage, since nothing here mutates the chunks -- they are
frozen models.
"""

import pytest

from agentic_erp_assistant.rag import embeddings as embeddings_module
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


_OPENAI_VARIABLES = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real configuration out of these tests.

    The same fixture ``tests/llm/conftest.py`` has, for the same two hazards: a
    real key exported in the shell, and a real ``.env`` the client would load at
    construction. Either one turns the "configuration is missing" tests green
    for the wrong reason locally and red in CI. That the suite then runs with no
    key anywhere is the point -- nothing in it should be able to reach a
    provider even by accident.
    """
    monkeypatch.setattr(
        embeddings_module, "load_dotenv", lambda *args, **kwargs: False
    )
    for name in _OPENAI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
