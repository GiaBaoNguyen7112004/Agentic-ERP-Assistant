"""The streaming half of the adapter: read as an SSE body, and land in the
same shapes _send already produces.

Every test runs through httpx.MockTransport returning a
``text/event-stream`` body -- no network, no key, and the same reason the
non-streaming tests do: what is under test is whether this code reads the
wire correctly, not whether OpenAI is up.
"""

import json

import httpx
import pytest

from agentic_erp_assistant.llm.adapters.openai_chat import OpenAIChatClient
from agentic_erp_assistant.llm.ports import ProviderAuthError, TransientProviderError
from agentic_erp_assistant.llm.tools import GET_PROJECT_STATUS_TOOL

MODEL = "test-model-1"
API_KEY = "sk-test-not-a-real-key"

USAGE = {"prompt_tokens": 100, "completion_tokens": 20}


def sse(*chunks: dict, done: bool = True) -> bytes:
    """Encode chunks as a Chat Completions SSE body."""
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def content_chunk(fragment: str | None, *, finish_reason: str | None = None) -> dict:
    return {
        "id": "chatcmpl-test",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "delta": {"content": fragment} if fragment is not None else {},
                "finish_reason": finish_reason,
            }
        ],
    }


def usage_chunk() -> dict:
    """The extra final chunk stream_options: {include_usage: true} asks for --
    empty choices, the usage block instead."""
    return {"id": "chatcmpl-test", "model": MODEL, "choices": [], "usage": USAGE}


class StreamingRecorder:
    """A mock transport handler serving one pre-built streaming response."""

    def __init__(self, body: bytes | None = None, *, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self._body = body if body is not None else sse(content_chunk("ok"), usage_chunk())
        self._status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self._status,
            content=self._body,
            headers={"content-type": "text/event-stream"},
        )

    @property
    def body(self) -> dict:
        assert len(self.requests) == 1
        return json.loads(self.requests[0].content)


def make_client(recorder: StreamingRecorder, **overrides) -> OpenAIChatClient:
    settings = {"api_key": API_KEY, "model": MODEL}
    settings.update(overrides)
    return OpenAIChatClient(transport=httpx.MockTransport(recorder), **settings)


# --------------------------------------------------------------------------
# Content streamed in fragments
# --------------------------------------------------------------------------


def test_content_in_three_fragments_calls_on_delta_three_times_in_order() -> None:
    body = sse(
        content_chunk("The "),
        content_chunk("vendor "),
        content_chunk("slipped.", finish_reason="stop"),
        usage_chunk(),
    )
    recorder = StreamingRecorder(body)
    seen: list[str] = []

    with make_client(recorder) as client:
        response = client.complete(
            [{"role": "user", "content": "How is M2?"}],
            temperature=0.0,
            on_delta=seen.append,
        )

    assert seen == ["The ", "vendor ", "slipped."]
    assert response["text"] == "The vendor slipped."
    assert response["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert response["stop_reason"] == "stop"
    assert client.last_usage == USAGE


def test_the_request_body_carries_stream_and_stream_options() -> None:
    recorder = StreamingRecorder()

    with make_client(recorder) as client:
        client.complete(
            [{"role": "user", "content": "hi"}], temperature=0.0, on_delta=lambda t: None
        )

    assert recorder.body["stream"] is True
    assert recorder.body["stream_options"] == {"include_usage": True}


def test_without_on_delta_the_request_does_not_stream() -> None:
    from agentic_erp_assistant.llm.adapters.openai_chat import OpenAIChatClient as _C

    class NonStreamRecorder:
        def __init__(self) -> None:
            self.requests: list[httpx.Request] = []

        def __call__(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(
                200,
                json={
                    "model": MODEL,
                    "choices": [
                        {
                            "message": {"content": "ok", "role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": USAGE,
                },
            )

    recorder = NonStreamRecorder()
    client = _C(api_key=API_KEY, model=MODEL, transport=httpx.MockTransport(recorder))

    client.complete([{"role": "user", "content": "hi"}], temperature=0.0)

    body = json.loads(recorder.requests[0].content)
    assert "stream" not in body


# --------------------------------------------------------------------------
# A tool call streamed as fragments
# --------------------------------------------------------------------------


def test_a_streamed_tool_call_produces_one_result_and_never_calls_on_delta() -> None:
    body = sse(
        {
            "id": "chatcmpl-test",
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "get_project_status", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": '{"mile'}}]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'stone_id'}}]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": '": "M2"}'}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        usage_chunk(),
    )
    recorder = StreamingRecorder(body)
    seen: list[str] = []

    with make_client(recorder) as client:
        result = client.call_with_tools(
            [{"role": "user", "content": "How is M2?"}],
            tools=[GET_PROJECT_STATUS_TOOL],
            temperature=0.0,
            on_delta=seen.append,
        )

    assert seen == []
    assert result.tool_name == "get_project_status"
    assert result.arguments == {"milestone_id": "M2"}


# --------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------


def test_no_usage_chunk_is_a_transient_failure() -> None:
    body = sse(content_chunk("ok", finish_reason="stop"))  # no usage_chunk()
    recorder = StreamingRecorder(body)

    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError, match="usage"):
            client.complete(
                [{"role": "user", "content": "hi"}],
                temperature=0.0,
                on_delta=lambda t: None,
            )


def test_a_429_mid_setup_is_transient() -> None:
    recorder = StreamingRecorder(b"", status=429)

    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError):
            client.complete(
                [{"role": "user", "content": "hi"}],
                temperature=0.0,
                on_delta=lambda t: None,
            )


def test_a_401_is_a_definitive_rejection() -> None:
    recorder = StreamingRecorder(b"", status=401)

    with make_client(recorder) as client:
        with pytest.raises(ProviderAuthError):
            client.complete(
                [{"role": "user", "content": "hi"}],
                temperature=0.0,
                on_delta=lambda t: None,
            )


def test_an_unreadable_chunk_is_a_transient_failure() -> None:
    recorder = StreamingRecorder(b"data: {not json\n\ndata: [DONE]\n\n")

    with make_client(recorder) as client:
        with pytest.raises(TransientProviderError, match="unreadable"):
            client.complete(
                [{"role": "user", "content": "hi"}],
                temperature=0.0,
                on_delta=lambda t: None,
            )
