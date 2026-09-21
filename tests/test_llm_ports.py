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
    ToolCallingClient,
    TransientProviderError,
)
from agentic_erp_assistant.llm.adapters.openai_chat import OpenAIChatClient
from agentic_erp_assistant.llm.tools import DEFAULT_TOOLS, ToolCallResult


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
    """The acceptance criterion: no provider-specific type reaches this module.

    ``llm.tools`` is allowed alongside the typing surface because it is our own
    vendor-neutral declaration of what a tool is -- the tool-calling port has to
    name it in a signature. It imports nothing from this package and nothing
    from a provider, so the direction the rest of this test protects is intact.
    """
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
    assert modules <= {
        "typing",
        "collections.abc",
        "agentic_erp_assistant.llm.tools",
    }


# --------------------------------------------------------------------------
# The second, optional surface: asking the model which tool to use
# --------------------------------------------------------------------------


class FakeToolCallingClient(FakeClient):
    """A provider that can also route. No SDK, no network."""

    def call_with_tools(self, messages, *, tools, temperature: float, on_delta=None):
        return ToolCallResult.from_tool_call(
            tool_name=tools[0].name, arguments={"milestone_id": "M2"}
        )


def test_a_text_only_client_is_still_a_valid_provider() -> None:
    """The reason this is a second protocol and not a wider first one."""
    client = FakeClient()

    assert isinstance(client, LargeLanguageModelClient)
    assert not isinstance(client, ToolCallingClient)


def test_a_client_can_satisfy_both_surfaces() -> None:
    client = FakeToolCallingClient()

    assert isinstance(client, LargeLanguageModelClient)
    assert isinstance(client, ToolCallingClient)


def test_the_real_adapter_satisfies_the_tool_calling_port() -> None:
    """Structurally, with no inheritance and no change to the adapter."""
    client = OpenAIChatClient(api_key="test-key", model="fake-model-1")

    assert isinstance(client, ToolCallingClient)
    assert isinstance(client, LargeLanguageModelClient)


def test_call_with_tools_returns_one_choice_and_never_two_outcomes() -> None:
    client: ToolCallingClient = FakeToolCallingClient()

    result = client.call_with_tools(
        [{"role": "user", "content": "How is M2 tracking?"}],
        tools=DEFAULT_TOOLS,
        temperature=0.0,
    )

    assert result.tool_name == DEFAULT_TOOLS[0].name
    assert result.content is None


# --------------------------------------------------------------------------
# on_delta: the port may be asked to stream, and may decline
# --------------------------------------------------------------------------


class StreamingClient:
    """A client that streams content fragments, then returns the whole thing --
    the contract on_delta promises: still one normalized result at the end."""

    model_name = "fake-streaming-1"

    def complete(
        self, messages, *, temperature: float, on_delta=None
    ) -> CompletionResponse:
        for fragment in ("Hel", "lo"):
            if on_delta is not None:
                on_delta(fragment)
        return {
            "text": "Hello",
            "model": self.model_name,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def call_with_tools(self, messages, *, tools, temperature: float, on_delta=None):
        return ToolCallResult.from_content("Hello")


def test_a_client_may_ignore_on_delta_and_remain_a_valid_port() -> None:
    """FakeClient never declared on_delta at all; a caller that never passes it
    is still calling a conforming client."""
    client: LargeLanguageModelClient = FakeClient()

    response = client.complete(
        [{"role": "user", "content": "hi"}], temperature=0.0
    )

    assert response["text"] == "ok"


def test_a_client_that_streams_still_returns_the_complete_normalized_result() -> (
    None
):
    client: LargeLanguageModelClient = StreamingClient()
    seen: list[str] = []

    response = client.complete(
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        on_delta=seen.append,
    )

    assert seen == ["Hel", "lo"]
    assert response["text"] == "Hello"


def test_on_delta_is_never_passed_tool_call_arguments() -> None:
    """The contract call_with_tools makes: content fragments only, never the
    tool-call arguments a model chose instead of answering."""
    client: ToolCallingClient = FakeToolCallingClient()
    seen: list[str] = []

    result = client.call_with_tools(
        [{"role": "user", "content": "How is M2 tracking?"}],
        tools=DEFAULT_TOOLS,
        temperature=0.0,
        on_delta=seen.append,
    )

    assert seen == []
    assert result.tool_name is not None
