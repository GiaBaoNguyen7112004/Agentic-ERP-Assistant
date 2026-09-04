"""The provider port is a contract, so its guarantees are tested, not assumed."""

import ast
from pathlib import Path

import pytest

from agentic_erp_assistant.llm.ports import (
    ClientConfigurationError,
    ClientConfigurationError as _ConfigAlias,
    CompletionResponse,
    LargeLanguageModelClient,
    LLMClientError,
    Message,
    ProviderAuthError,
    TransientProviderError,
)


class FakeClient:
    """A stand-in provider: no SDK, no network, nothing to configure."""

    model_name = "fake-model-1"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Message, ...], float]] = []

    def complete(
        self,
        messages,
        *,
        temperature: float,
    ) -> CompletionResponse:
        self.calls.append((tuple(messages), temperature))
        return {
            "text": "ok",
            "model": self.model_name,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }


def test_a_plain_class_satisfies_the_port_without_inheriting_from_it() -> None:
    client = FakeClient()
    assert isinstance(client, LargeLanguageModelClient)


def test_a_class_missing_complete_does_not_satisfy_the_port() -> None:
    class NotAClient:
        model_name = "nope"

    assert not isinstance(NotAClient(), LargeLanguageModelClient)


def test_complete_returns_the_normalized_response_shape() -> None:
    client: LargeLanguageModelClient = FakeClient()
    messages: list[Message] = [{"role": "user", "content": "hi"}]

    response = client.complete(messages, temperature=0.0)

    assert response["text"] == "ok"
    assert response["model"] == "fake-model-1"
    assert response["usage"]["input_tokens"] == 3


def test_temperature_is_keyword_only() -> None:
    client = FakeClient()
    with pytest.raises(TypeError):
        client.complete([{"role": "user", "content": "hi"}], 0.5)  # type: ignore[misc]


@pytest.mark.parametrize(
    "error",
    [TransientProviderError, ProviderAuthError, ClientConfigurationError],
)
def test_every_failure_is_catchable_as_one_base(error: type[LLMClientError]) -> None:
    with pytest.raises(LLMClientError):
        raise error("boom")


@pytest.mark.parametrize(
    ("raised", "other"),
    [
        (ProviderAuthError, TransientProviderError),
        (ClientConfigurationError, TransientProviderError),
        (TransientProviderError, ProviderAuthError),
        (TransientProviderError, ClientConfigurationError),
        (ProviderAuthError, ClientConfigurationError),
        (ClientConfigurationError, ProviderAuthError),
    ],
)
def test_the_three_failures_never_catch_each_other(
    raised: type[LLMClientError],
    other: type[LLMClientError],
) -> None:
    """The retry engine catches TransientProviderError; it must catch only that."""
    with pytest.raises(raised):
        try:
            raise raised("boom")
        except other:  # pragma: no cover - the point is that this never runs
            pytest.fail(f"{raised.__name__} was caught as {other.__name__}")


def test_a_missing_key_is_configuration_not_auth() -> None:
    """Nothing was sent, so it is a local bug -- not the provider refusing us."""
    assert not issubclass(_ConfigAlias, ProviderAuthError)


def test_ports_imports_nothing_outside_the_standard_typing_surface() -> None:
    """The acceptance criterion: no provider-specific type reaches this module."""
    source = Path("src/agentic_erp_assistant/llm/ports.py").read_text(encoding="utf-8")
    modules = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert modules <= {"typing", "collections.abc"}
