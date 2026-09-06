"""The embeddings adapter, proven against an injected transport.

Every test here runs through ``httpx.MockTransport``: no network, no key, no
cost. What is under test is the request/response contract -- does the body carry
the model and the inputs, does a 429 become the retryable error, does an
out-of-order reply get put back in order -- and an injected transport is the only
way to assert all of that without billing an account.
"""

import httpx
import pytest

from agentic_erp_assistant.llm.ports import (
    ClientConfigurationError,
    ProviderAuthError,
    TransientProviderError,
)
from agentic_erp_assistant.llm.retry import retry_with_backoff
from agentic_erp_assistant.rag.embeddings import (
    MAX_INPUTS_PER_REQUEST,
    OpenAIEmbeddingsClient,
)
from agentic_erp_assistant.rag.ports import EmbeddingsPort

API_KEY = "sk-test-not-a-real-key"
MODEL = "text-embedding-3-small"

SETTINGS = {"api_key": API_KEY, "model": MODEL}


class Recorder:
    """A transport handler that remembers what it was asked and answers with a
    scripted reply."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("the client made an unexpected extra request")
        return self._responses.pop(0)


def reply(
    *vectors: list[float],
    indices: list[int] | None = None,
    model: str = MODEL,
    usage: dict | None = {"prompt_tokens": 42, "total_tokens": 42},
) -> httpx.Response:
    order = indices if indices is not None else list(range(len(vectors)))
    body = {
        "object": "list",
        "model": model,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in zip(order, vectors, strict=True)
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def client(recorder: Recorder) -> OpenAIEmbeddingsClient:
    return OpenAIEmbeddingsClient(
        transport=httpx.MockTransport(recorder), **SETTINGS
    )


def failing(error: Exception) -> Recorder:
    class Raiser(Recorder):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            raise error

    return Raiser()


# -- the happy path ----------------------------------------------------------


def test_one_request_carries_the_whole_batch() -> None:
    """Batching is the cost decision: one round trip and one rate-limit slot."""
    recorder = Recorder(reply([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]))

    with client(recorder) as embedder:
        batch = embedder.embed(["alpha", "beta", "gamma"])

    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.url.path.endswith("/embeddings")
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"

    import json

    payload = json.loads(request.content)
    assert payload == {"model": MODEL, "input": ["alpha", "beta", "gamma"]}

    assert batch.vectors == ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))
    assert batch.dimensions == 2
    assert batch.prompt_tokens == 42
    assert batch.model == MODEL
    assert len(batch) == 3


def test_the_reply_is_sorted_by_index_not_taken_in_arrival_order() -> None:
    """The caller pairs vectors with chunks positionally.

    A reordering the adapter did not undo would attach every passage to the
    wrong vector, and nothing downstream could detect it -- a mis-paired vector
    is still a perfectly valid vector.
    """
    recorder = Recorder(
        reply([9.0, 9.0], [1.0, 1.0], [5.0, 5.0], indices=[2, 0, 1])
    )

    with client(recorder) as embedder:
        batch = embedder.embed(["a", "b", "c"])

    assert batch.vectors == ((1.0, 1.0), (5.0, 5.0), (9.0, 9.0))


def test_the_served_model_is_reported_when_it_differs() -> None:
    recorder = Recorder(reply([1.0], model="text-embedding-3-small-2026-01"))
    with client(recorder) as embedder:
        assert embedder.embed(["a"]).model == "text-embedding-3-small-2026-01"


def test_an_empty_batch_costs_nothing() -> None:
    recorder = Recorder()
    with client(recorder) as embedder:
        batch = embedder.embed([])

    assert recorder.requests == []
    assert batch.vectors == () and batch.dimensions == 0 and batch.prompt_tokens == 0


# -- inputs the caller got wrong ---------------------------------------------


def test_a_blank_input_is_refused_before_the_request() -> None:
    recorder = Recorder()
    with client(recorder) as embedder, pytest.raises(ValueError, match="blank"):
        embedder.embed(["alpha", "   ", "gamma"])
    assert recorder.requests == []


def test_an_oversized_batch_is_refused_locally() -> None:
    """A 400 would be classified as definitive -- true, and unhelpful."""
    recorder = Recorder()
    with client(recorder) as embedder, pytest.raises(ValueError, match="split the batch"):
        embedder.embed(["x"] * (MAX_INPUTS_PER_REQUEST + 1))
    assert recorder.requests == []


# -- the failure matrix ------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        (429, TransientProviderError),
        (500, TransientProviderError),
        (503, TransientProviderError),
        (401, ProviderAuthError),
        (403, ProviderAuthError),
        (400, ProviderAuthError),
        (404, ProviderAuthError),
    ],
)
def test_each_status_maps_to_the_right_error(
    status: int, expected: type[Exception]
) -> None:
    recorder = Recorder(httpx.Response(status, text="upstream said no"))
    with client(recorder) as embedder, pytest.raises(expected):
        embedder.embed(["alpha"])


def test_a_timeout_is_retryable() -> None:
    recorder = failing(httpx.ReadTimeout("too slow"))
    with client(recorder) as embedder, pytest.raises(TransientProviderError, match="timed out"):
        embedder.embed(["alpha"])


def test_a_transport_failure_is_retryable() -> None:
    recorder = failing(httpx.ConnectError("no route"))
    with client(recorder) as embedder, pytest.raises(
        TransientProviderError, match="transport failure"
    ):
        embedder.embed(["alpha"])


def test_an_error_message_carries_the_request_id_and_never_the_key() -> None:
    recorder = Recorder(
        httpx.Response(500, text="boom", headers={"x-request-id": "req_123"})
    )
    with client(recorder) as embedder:
        with pytest.raises(TransientProviderError) as raised:
            embedder.embed(["alpha"])

    assert "req_123" in str(raised.value)
    assert API_KEY not in str(raised.value)


# -- replies that are not usable ---------------------------------------------


def test_a_short_reply_is_a_transient_failure() -> None:
    recorder = Recorder(reply([1.0, 0.0]))
    with client(recorder) as embedder, pytest.raises(TransientProviderError, match="for 2 inputs"):
        embedder.embed(["alpha", "beta"])


def test_duplicate_indices_are_refused() -> None:
    recorder = Recorder(reply([1.0], [2.0], indices=[0, 0]))
    with client(recorder) as embedder, pytest.raises(
        TransientProviderError, match="expected 0..1"
    ):
        embedder.embed(["alpha", "beta"])


def test_vectors_of_two_widths_in_one_batch_are_refused() -> None:
    """They cannot go into one collection, and a similarity across them would be
    meaningless rather than merely wrong."""
    recorder = Recorder(reply([1.0, 0.0], [1.0, 0.0, 0.0]))
    with client(recorder) as embedder, pytest.raises(
        TransientProviderError, match="different\n?\\s*widths"
    ):
        embedder.embed(["alpha", "beta"])


def test_a_missing_usage_block_is_a_failure_not_a_zero() -> None:
    """Zero cost is a number the budget layer would believe."""
    recorder = Recorder(reply([1.0], usage=None))
    with client(recorder) as embedder, pytest.raises(TransientProviderError, match="usage"):
        embedder.embed(["alpha"])


def test_an_unreadable_body_is_a_transient_failure() -> None:
    recorder = Recorder(httpx.Response(200, text="not json"))
    with client(recorder) as embedder, pytest.raises(TransientProviderError, match="unreadable"):
        embedder.embed(["alpha"])


# -- configuration -----------------------------------------------------------


def test_a_missing_key_is_a_credential_problem() -> None:
    with pytest.raises(ProviderAuthError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingsClient(
            model=MODEL, transport=httpx.MockTransport(Recorder())
        )


def test_a_missing_embedding_model_is_a_configuration_problem() -> None:
    """Different type from a missing key on purpose: nobody rejected anything,
    and the choice is baked into every stored vector."""
    with pytest.raises(ClientConfigurationError, match="OPENAI_EMBEDDING_MODEL"):
        OpenAIEmbeddingsClient(
            api_key=API_KEY, transport=httpx.MockTransport(Recorder())
        )


def test_a_blank_value_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "   ")
    with pytest.raises(ClientConfigurationError):
        OpenAIEmbeddingsClient(transport=httpx.MockTransport(Recorder()))


def test_the_environment_supplies_both_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "model-from-the-environment")
    recorder = Recorder(reply([1.0]))

    with OpenAIEmbeddingsClient(transport=httpx.MockTransport(recorder)) as embedder:
        assert embedder.model_name == "model-from-the-environment"
        embedder.embed(["alpha"])

    import json

    assert json.loads(recorder.requests[0].content)["model"] == (
        "model-from-the-environment"
    )


def test_transport_and_http_client_together_are_a_caller_bug() -> None:
    with pytest.raises(ValueError, match="not both"):
        OpenAIEmbeddingsClient(
            **SETTINGS,
            transport=httpx.MockTransport(Recorder()),
            http_client=httpx.Client(),
        )


def test_a_borrowed_client_is_not_closed() -> None:
    borrowed = httpx.Client(transport=httpx.MockTransport(Recorder()))
    with OpenAIEmbeddingsClient(**SETTINGS, http_client=borrowed):
        pass
    assert not borrowed.is_closed
    borrowed.close()


# -- the port and the retry engine -------------------------------------------


def test_the_client_satisfies_the_port() -> None:
    with client(Recorder()) as embedder:
        assert isinstance(embedder, EmbeddingsPort)


def test_the_existing_retry_engine_already_handles_this_client() -> None:
    """Reusing the provider error types is what buys this: no second retry path."""
    recorder = Recorder(httpx.Response(503, text="later"), reply([1.0, 2.0]))
    waits: list[float] = []

    with client(recorder) as embedder:
        batch = retry_with_backoff(
            lambda: embedder.embed(["alpha"]),
            max_attempts=2,
            sleep=waits.append,
        )

    assert batch.vectors == ((1.0, 2.0),)
    assert len(recorder.requests) == 2 and waits
