"""The adapter is a contract with a provider, so the wire is what is tested.

Every test here runs through an injected ``httpx.MockTransport``: no network, no
key, no cost, and no dependency on a provider's uptime. That is not a compromise
made to keep tests fast -- the thing under test genuinely *is* the shape of the
request and the classification of the reply. A live call would exercise OpenAI's
availability and tell us nothing about either.

What is asserted, in order: the request body (schema, model, temperature, roles),
the success path (text, usage, model, stop reason), every documented failure
mode, and the construction-time configuration checks.
"""

import ast
import json
from pathlib import Path

import httpx
import pytest

from agentic_erp_assistant.llm.client import (
    EVIDENCE_PREAMBLE,
    RESPONSE_FORMAT_NAME,
    OpenAIChatClient,
)
from agentic_erp_assistant.llm.ports import (
    ClientConfigurationError,
    LargeLanguageModelClient,
    Message,
    ProviderAuthError,
    TransientProviderError,
)
from agentic_erp_assistant.llm.prompts import build_messages
from agentic_erp_assistant.llm.schemas import EvidenceSnippet, GroundedAnswer

MODEL = "test-model-1"
API_KEY = "sk-test-not-a-real-key"

ANSWER_JSON = json.dumps(
    {
        "answer": "Sprint 12 closed 31 of 34 points.",
        "citations": [{"source_id": "sprint-12-report.md", "locator": "3.2"}],
        "grounded": True,
        "confidence": 0.9,
        "refusal_reason": None,
    }
)

USAGE = {
    "prompt_tokens": 812,
    "completion_tokens": 57,
    "total_tokens": 869,
    # A nested block the port's two-field Usage has no room for. It is here to
    # prove last_usage keeps what the provider actually reported.
    "prompt_tokens_details": {"cached_tokens": 640},
}


def success_body(*, model: str = MODEL, content: str = ANSWER_JSON) -> dict:
    """A minimal but realistically-shaped Chat Completions reply."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": USAGE,
    }


class Recorder:
    """A mock transport handler that keeps every request it was given."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses) or [httpx.Response(200, json=success_body())]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self._responses[min(len(self.requests) - 1, len(self._responses) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def body(self) -> dict:
        """The decoded JSON body of the single recorded request."""
        assert len(self.requests) == 1, f"expected one request, got {len(self.requests)}"
        return json.loads(self.requests[0].content)


def make_client(recorder: Recorder, **overrides) -> OpenAIChatClient:
    """Build a fully-injected client: no environment involved."""
    settings = {"api_key": API_KEY, "model": MODEL}
    settings.update(overrides)
    return OpenAIChatClient(transport=httpx.MockTransport(recorder), **settings)


@pytest.fixture
def messages() -> list[Message]:
    """A real four-block prompt, so role folding is tested on the real shape."""
    return build_messages(
        "How did sprint 12 go?",
        [
            EvidenceSnippet(
                source_id="sprint-12-report.md",
                locator="3.2",
                text="Closed 31 of 34 committed points.",
            )
        ],
    )


# --------------------------------------------------------------------------
# The request body
# --------------------------------------------------------------------------


def test_response_format_carries_the_schema_byte_for_byte(messages) -> None:
    """The acceptance criterion: the emitted schema, not a copy of it."""
    recorder = Recorder()
    with make_client(recorder) as client:
        client.complete(messages, temperature=0.0)

    response_format = recorder.body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == RESPONSE_FORMAT_NAME
    assert response_format["json_schema"]["schema"] == GroundedAnswer.model_json_schema()


def test_strict_is_false_so_the_schema_never_has_to_be_rewritten(messages) -> None:
    """Strict mode would demand every field in ``required``; GroundedAnswer has
    defaults, so honoring it would mean shipping a second, drifting schema."""
    recorder = Recorder()
    with make_client(recorder) as client:
        client.complete(messages, temperature=0.0)

    assert recorder.body["response_format"]["json_schema"]["strict"] is False


def test_the_model_and_temperature_are_sent_as_configured(messages) -> None:
    recorder = Recorder()
    with make_client(recorder, model="some-other-model") as client:
        client.complete(messages, temperature=0.7)

    assert recorder.body["model"] == "some-other-model"
    assert recorder.body["temperature"] == 0.7


def test_the_key_travels_in_the_header_and_not_in_the_body(messages) -> None:
    recorder = Recorder()
    with make_client(recorder) as client:
        client.complete(messages, temperature=0.0)

    assert recorder.requests[0].headers["authorization"] == f"Bearer {API_KEY}"
    assert API_KEY not in recorder.requests[0].content.decode()


def test_it_posts_to_the_chat_completions_endpoint(messages) -> None:
    recorder = Recorder()
    with make_client(recorder) as client:
        client.complete(messages, temperature=0.0)

    assert str(recorder.requests[0].url) == (
        "https://api.openai.com/v1/chat/completions"
    )


def test_evidence_is_relabeled_developer_and_stays_its_own_message(messages) -> None:
    """The role has to be folded -- the API knows four roles -- but folding it
    into the neighbouring turn would erase the boundary the role existed for."""
    recorder = Recorder()
    with make_client(recorder) as client:
        client.complete(messages, temperature=0.0)

    wire = recorder.body["messages"]
    assert [message["role"] for message in wire] == [
        "system",
        "developer",
        "user",
        "developer",
    ]

    evidence_block = wire[3]["content"]
    assert evidence_block.startswith(EVIDENCE_PREAMBLE)
    assert "[sprint-12-report.md#3.2]" in evidence_block

    # The user's turn is untouched: no evidence appended, nothing prefixed.
    assert wire[2]["content"] == "How did sprint 12 go?"


def test_the_other_roles_pass_through_unchanged() -> None:
    recorder = Recorder()
    conversation: list[Message] = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "prior reply"},
    ]
    with make_client(recorder) as client:
        client.complete(conversation, temperature=0.0)

    assert recorder.body["messages"] == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "prior reply"},
    ]


# --------------------------------------------------------------------------
# The success path
# --------------------------------------------------------------------------


def test_a_successful_call_returns_the_normalized_response(messages) -> None:
    recorder = Recorder(httpx.Response(200, json=success_body(model="served-model-2")))
    with make_client(recorder) as client:
        response = client.complete(messages, temperature=0.0)

    assert response["text"] == ANSWER_JSON
    # What answered, not what was asked for: an alias can resolve to a snapshot.
    assert response["model"] == "served-model-2"
    assert response["stop_reason"] == "stop"
    assert response["usage"] == {"input_tokens": 812, "output_tokens": 57}


def test_the_returned_text_is_raw_and_not_parsed_for_the_caller(messages) -> None:
    """Validation is a guardrail decision with a trace entry; not this layer's."""
    recorder = Recorder(
        httpx.Response(200, json=success_body(content="I am not JSON at all."))
    )
    with make_client(recorder) as client:
        response = client.complete(messages, temperature=0.0)

    assert response["text"] == "I am not JSON at all."


def test_last_usage_is_the_providers_own_object_verbatim(messages) -> None:
    recorder = Recorder()
    with make_client(recorder) as client:
        assert client.last_usage is None
        client.complete(messages, temperature=0.0)
        assert client.last_usage == USAGE


def test_last_usage_is_a_copy_so_a_caller_cannot_edit_the_record(messages) -> None:
    recorder = Recorder(
        httpx.Response(200, json=success_body()),
        httpx.Response(200, json=success_body()),
    )
    with make_client(recorder) as client:
        client.complete(messages, temperature=0.0)
        first = client.last_usage
        assert first is not None
        first["prompt_tokens"] = 0

        client.complete(messages, temperature=0.0)
        assert client.last_usage == USAGE


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_rate_limits_and_server_errors_are_transient(messages, status: int) -> None:
    recorder = Recorder(httpx.Response(status, json={"error": {"message": "later"}}))
    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError):
            client.complete(messages, temperature=0.0)


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_an_auth_error(messages, status: int) -> None:
    recorder = Recorder(httpx.Response(status, json={"error": {"message": "no"}}))
    with make_client(recorder) as client:
        with pytest.raises(ProviderAuthError):
            client.complete(messages, temperature=0.0)


@pytest.mark.parametrize("status", [400, 404, 422])
def test_other_client_errors_are_definitive_not_retryable(
    messages, status: int
) -> None:
    """A malformed request, an unknown model, a rejected temperature: repeating
    any of them unchanged only burns the retry budget."""
    recorder = Recorder(httpx.Response(status, json={"error": {"message": "nope"}}))
    with make_client(recorder) as client:
        with pytest.raises(ProviderAuthError):
            client.complete(messages, temperature=0.0)


def test_an_error_message_carries_the_status_and_request_id(messages) -> None:
    recorder = Recorder(
        httpx.Response(
            500,
            json={"error": {"message": "upstream exploded"}},
            headers={"x-request-id": "req_abc123"},
        )
    )
    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError) as failure:
            client.complete(messages, temperature=0.0)

    detail = str(failure.value)
    assert "500" in detail
    assert "req_abc123" in detail
    assert "upstream exploded" in detail


def test_a_connect_timeout_is_transient(messages) -> None:
    recorder = Recorder(httpx.ConnectTimeout("too slow"))
    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError):
            client.complete(messages, temperature=0.0)


def test_a_read_timeout_is_transient(messages) -> None:
    recorder = Recorder(httpx.ReadTimeout("still waiting"))
    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError):
            client.complete(messages, temperature=0.0)


def test_a_transport_failure_is_transient(messages) -> None:
    recorder = Recorder(httpx.ConnectError("connection refused"))
    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError):
            client.complete(messages, temperature=0.0)


def test_an_unreadable_body_is_transient(messages) -> None:
    recorder = Recorder(httpx.Response(200, content=b"<html>gateway</html>"))
    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError):
            client.complete(messages, temperature=0.0)


def test_a_reply_with_no_choices_is_transient(messages) -> None:
    recorder = Recorder(httpx.Response(200, json={"model": MODEL, "choices": []}))
    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError):
            client.complete(messages, temperature=0.0)


def test_null_content_is_a_failure_rather_than_an_empty_answer(messages) -> None:
    body = success_body()
    body["choices"][0]["message"]["content"] = None
    body["choices"][0]["finish_reason"] = "content_filter"
    recorder = Recorder(httpx.Response(200, json=body))

    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError) as failure:
            client.complete(messages, temperature=0.0)

    assert "content_filter" in str(failure.value)


def test_a_missing_usage_block_fails_rather_than_reporting_zero(messages) -> None:
    """Zero cost is a number the budget layer would believe."""
    body = success_body()
    del body["usage"]
    recorder = Recorder(httpx.Response(200, json=body))

    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError):
            client.complete(messages, temperature=0.0)
        assert client.last_usage is None


def test_an_unusable_usage_block_fails(messages) -> None:
    body = success_body()
    body["usage"] = {"prompt_tokens": "many", "completion_tokens": 1}
    recorder = Recorder(httpx.Response(200, json=body))

    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError):
            client.complete(messages, temperature=0.0)


# --------------------------------------------------------------------------
# Configuration, read at construction
# --------------------------------------------------------------------------


def test_a_missing_key_raises_auth_error_before_anything_is_sent() -> None:
    recorder = Recorder()
    with pytest.raises(ProviderAuthError) as failure:
        OpenAIChatClient(transport=httpx.MockTransport(recorder))

    assert "OPENAI_API_KEY" in str(failure.value)
    assert recorder.requests == []


def test_a_missing_model_is_a_configuration_error_not_an_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nobody rejected anything -- a local file is incomplete."""
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    recorder = Recorder()

    with pytest.raises(ClientConfigurationError) as failure:
        OpenAIChatClient(transport=httpx.MockTransport(recorder))

    assert "OPENAI_MODEL" in str(failure.value)
    assert recorder.requests == []


def test_a_blank_value_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OPENAI_MODEL=`` in a copied .env.example is the common mistake."""
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    with pytest.raises(ClientConfigurationError):
        OpenAIChatClient(transport=httpx.MockTransport(Recorder()))


def test_the_environment_supplies_both_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.setenv("OPENAI_MODEL", "model-from-the-environment")
    recorder = Recorder()

    with OpenAIChatClient(transport=httpx.MockTransport(recorder)) as client:
        assert client.model_name == "model-from-the-environment"
        client.complete([{"role": "user", "content": "hi"}], temperature=0.0)

    assert recorder.body["model"] == "model-from-the-environment"


def test_no_model_is_assumed_when_the_environment_is_empty() -> None:
    """There is no default model anywhere in the adapter, by design."""
    source = Path("src/agentic_erp_assistant/llm/client.py").read_text(
        encoding="utf-8"
    )
    assert "gpt-" not in source


def test_transport_and_http_client_together_is_a_caller_bug() -> None:
    with pytest.raises(ValueError):
        OpenAIChatClient(
            api_key=API_KEY,
            model=MODEL,
            transport=httpx.MockTransport(Recorder()),
            http_client=httpx.Client(),
        )


# --------------------------------------------------------------------------
# Lifecycle and conformance
# --------------------------------------------------------------------------


def test_it_satisfies_the_provider_port() -> None:
    client = make_client(Recorder())
    assert isinstance(client, LargeLanguageModelClient)
    client.close()


def test_leaving_the_with_block_closes_the_pool_it_opened() -> None:
    with make_client(Recorder()) as client:
        assert not client._http.is_closed
    assert client._http.is_closed


def test_an_injected_client_is_left_open_because_the_caller_owns_it() -> None:
    borrowed = httpx.Client(transport=httpx.MockTransport(Recorder()))
    with OpenAIChatClient(api_key=API_KEY, model=MODEL, http_client=borrowed):
        pass

    assert not borrowed.is_closed
    borrowed.close()


def test_no_other_llm_module_reaches_for_the_network() -> None:
    """The adapter boundary, checked rather than trusted.

    Constraint: no provider-specific dependency outside the adapter layer. An
    ``httpx`` or ``dotenv`` import appearing in ports, schemas or prompts would
    mean the boundary had quietly moved.
    """
    package = Path("src/agentic_erp_assistant/llm")
    offenders = []

    for module in sorted(package.glob("*.py")):
        if module.name == "client.py":
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in {"httpx", "dotenv", "openai", "requests"}:
                    offenders.append(f"{module.name}: {name}")

    assert offenders == []
