"""AppResources: process-wide, built once. Tested with fakes -- from_env's
own real construction is proven live in docs/manual-test.md and by
scripts/run_turn.py, not here."""

import pytest

from agentic_erp_assistant.composition.resources import AppResources, ResourcesError
from agentic_erp_assistant.composition.settings import Settings
from agentic_erp_assistant.composition.users import User, UserDirectory


def a_settings() -> Settings:
    return Settings(model="gpt-4o", context_window=128_000)


def a_directory() -> UserDirectory:
    return UserDirectory(
        (
            User(
                actor="priya",
                display_name="Priya Raman",
                role="Delivery lead",
                project_code="atlas",
                scopes=frozenset({"project.docs.read"}),
            ),
        )
    )


class FailingCloser:
    def close(self) -> None:
        raise RuntimeError("connection already gone")


class RecordingCloser:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def a_resources(**overrides) -> AppResources:
    fields = {
        "settings": a_settings(),
        "users": a_directory(),
        "chat_client": RecordingCloser(),
        "embeddings": object(),
        "retrieval": RecordingCloser(),
        "memory_index": RecordingCloser(),
        "erp": object(),
        "registry": object(),
        "limiter": object(),
        "manifest": {},
        "connect": lambda: None,
    }
    fields.update(overrides)
    return AppResources(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# close(): shutdown must never raise
# --------------------------------------------------------------------------


def test_close_calls_every_closer() -> None:
    chat = RecordingCloser()
    retrieval = RecordingCloser()
    memory_index = RecordingCloser()
    resources = a_resources(
        chat_client=chat, retrieval=retrieval, memory_index=memory_index
    )

    resources.close()

    assert chat.closed
    assert retrieval.closed
    assert memory_index.closed


def test_close_never_raises_even_when_every_closer_fails() -> None:
    resources = a_resources(
        chat_client=FailingCloser(),
        retrieval=FailingCloser(),
        memory_index=FailingCloser(),
    )

    resources.close()  # must not raise


def test_close_tolerates_a_memory_index_with_no_close_method() -> None:
    resources = a_resources(memory_index=object())

    resources.close()  # must not raise, and must not call anything


# --------------------------------------------------------------------------
# from_env(): the loud failure on an empty index
# --------------------------------------------------------------------------


def test_from_env_refuses_an_empty_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "128000")

    class EmptyVectorIndex:
        @classmethod
        def from_env(cls, **kwargs):
            return cls()

        def iter_chunks(self):
            return ()

        def close(self) -> None:
            pass

    class FakeEmbeddings:
        def __init__(self, *a, **k) -> None:
            pass

    monkeypatch.setattr(
        "agentic_erp_assistant.rag.vector_index.QdrantVectorIndex", EmptyVectorIndex
    )
    monkeypatch.setattr(
        "agentic_erp_assistant.rag.embeddings.OpenAIEmbeddingsClient", FakeEmbeddings
    )
    monkeypatch.setattr(
        "agentic_erp_assistant.composition.users.UserDirectory.load",
        classmethod(lambda cls, path: a_directory()),
    )
    monkeypatch.setattr(
        "agentic_erp_assistant.llm.adapters.openai_chat.OpenAIChatClient.__init__",
        lambda self, **kwargs: setattr(self, "model_name", kwargs.get("model", "x")),
    )

    with pytest.raises(ResourcesError, match="ingest_documents"):
        AppResources.from_env()
