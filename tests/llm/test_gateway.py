"""The assembled path, tested as an assembly.

Steps 2 and 7 are already proven in isolation. What this file proves is that
they still fire when wired together, and in the right order: the budget check
above the client, the retry around it, the schema validation below it, and a
telemetry record on every path that spent money.

Most tests drive the real ``OpenAIChatClient`` over ``httpx.MockTransport``,
because "the request never reached the transport" is only meaningful against a
client that would really have sent one. One test uses a bare port-conforming
fake, to show the gateway depends on the port and not on the adapter.
"""

import json

import httpx
import pytest
from pydantic import ValidationError

from agentic_erp_assistant.llm.adapters.openai_chat import OpenAIChatClient
from agentic_erp_assistant.llm.gateway import (
    WHOLE_DOCUMENT,
    ContextWindowExceeded,
    LLMGateway,
)
from agentic_erp_assistant.llm.ports import (
    CompletionResponse,
    ProviderAuthError,
    TransientProviderError,
)
from agentic_erp_assistant.llm.schemas import EvidenceSnippet, GroundedAnswer
from agentic_erp_assistant.llm.telemetry import InMemoryTelemetry
from agentic_erp_assistant.llm.tools import LIST_RISKS_TOOL, ToolCallResult

MODEL = "gpt-4o"  # priced in pricing.py, so cost assertions are real
API_KEY = "sk-test-not-a-real-key"
QUESTION = "What is the refund window?"
EVIDENCE = {"doc-1": "Refunds close after 30 days."}

USAGE = {"prompt_tokens": 812, "completion_tokens": 57, "total_tokens": 869}

GROUNDED_JSON = json.dumps(
    {
        "answer": "Refunds close after 30 days.",
        "citations": [{"source_id": "doc-1", "locator": "full"}],
        "grounded": True,
        "confidence": 0.9,
        "refusal_reason": None,
    }
)

UNGROUNDED_JSON = json.dumps(
    {
        "answer": "Refunds close after 30 days.",
        "citations": [],
        "grounded": True,  # claimed, unbacked: schemas.py must reject this
        "confidence": 0.9,
        "refusal_reason": None,
    }
)


def reply(content: str = GROUNDED_JSON, *, usage: dict | None = USAGE) -> dict:
    body = {
        "id": "chatcmpl-test",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return body


class Recorder:
    """Mock transport handler: records requests, replays scripted responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses) or [httpx.Response(200, json=reply())]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        return self._responses[index]


class FixedCounter:
    """A tokenizer stand-in, so budget arithmetic is exact rather than incidental."""

    def __init__(self, tokens: int = 100) -> None:
        self.tokens = tokens

    def count_tokens(self, text: str, *, model: str) -> int:
        return self.tokens

    def count_message_tokens(self, messages, *, model: str) -> int:
        return self.tokens


class FakePortClient:
    """A client that satisfies the port and knows nothing about HTTP."""

    model_name = "fake-model-1"

    def __init__(self, text: str = GROUNDED_JSON) -> None:
        self.calls = 0
        self._text = text

    def complete(self, messages, *, temperature: float) -> CompletionResponse:
        self.calls += 1
        return {
            "text": self._text,
            "model": self.model_name,
            "stop_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }


def make_gateway(
    recorder: Recorder,
    *,
    context_window: int = 128_000,
    **overrides,
) -> tuple[LLMGateway, InMemoryTelemetry, OpenAIChatClient]:
    telemetry = InMemoryTelemetry()
    client = OpenAIChatClient(
        api_key=API_KEY, model=MODEL, transport=httpx.MockTransport(recorder)
    )
    settings = {
        "telemetry": telemetry,
        "sleep": lambda _: None,
        "jitter": lambda: 1.0,
    }
    settings.update(overrides)
    gateway = LLMGateway(client, context_window=context_window, **settings)
    return gateway, telemetry, client


# --------------------------------------------------------------------------
# Step 2 of answer(): the budget check, above the client
# --------------------------------------------------------------------------


def test_an_oversized_request_never_reaches_the_transport() -> None:
    """The acceptance criterion: refused locally, before it can cost anything."""
    recorder = Recorder()
    gateway, telemetry, client = make_gateway(recorder, context_window=64)

    with pytest.raises(ContextWindowExceeded):
        gateway.answer(QUESTION, EVIDENCE)
    client.close()

    assert recorder.requests == []
    assert client.last_usage is None
    assert [record.outcome for record in telemetry.records] == ["budget_exceeded"]


def test_a_refused_request_is_recorded_with_no_spend() -> None:
    recorder = Recorder()
    gateway, telemetry, client = make_gateway(
        recorder, context_window=200, counter=FixedCounter(500), output_reserve=10
    )
    extra = client.estimate_extra_tokens(structured=True)

    with pytest.raises(ContextWindowExceeded) as failure:
        gateway.answer(QUESTION, EVIDENCE)
    client.close()

    record = telemetry.records[0]
    assert record.estimated_input_tokens == 500 + extra
    assert (record.input_tokens, record.output_tokens) == (0, 0)
    assert record.attempts == 0
    assert record.cost_usd == 0.0  # priced model, nothing spent
    assert "exceeds" in str(failure.value)


def test_a_request_that_fills_the_window_exactly_is_sent() -> None:
    """Strictly greater than: equality fits, and refusing it would be a second
    invisible margin on top of output_reserve."""
    recorder = Recorder()
    extra = OpenAIChatClient(api_key=API_KEY, model=MODEL).estimate_extra_tokens(
        structured=True
    )

    gateway, _, client = make_gateway(
        recorder,
        context_window=110 + extra,
        counter=FixedCounter(100),
        output_reserve=10,
    )

    gateway.answer(QUESTION, EVIDENCE)
    client.close()

    assert len(recorder.requests) == 1


def test_one_token_over_the_window_is_refused() -> None:
    recorder = Recorder()
    gateway, _, client = make_gateway(
        recorder,
        context_window=109,
        counter=FixedCounter(100),
        output_reserve=10,
    )

    with pytest.raises(ContextWindowExceeded):
        gateway.answer(QUESTION, EVIDENCE)
    client.close()

    assert recorder.requests == []


def test_the_reserve_is_part_of_the_budget() -> None:
    """A prompt that fits exactly still fails once the model starts answering."""
    recorder = Recorder()
    gateway, _, client = make_gateway(
        recorder,
        context_window=100,
        counter=FixedCounter(100),
        output_reserve=1,
    )

    with pytest.raises(ContextWindowExceeded):
        gateway.answer(QUESTION, EVIDENCE)
    client.close()


def test_the_context_window_has_no_default() -> None:
    """Assuming one model's window while running another is the bug this
    whole check exists to prevent."""
    client = OpenAIChatClient(
        api_key=API_KEY, model=MODEL, transport=httpx.MockTransport(Recorder())
    )

    with pytest.raises(TypeError):
        LLMGateway(client)  # type: ignore[call-arg]
    client.close()


# --------------------------------------------------------------------------
# The happy path, and what gets recorded
# --------------------------------------------------------------------------


def test_a_valid_reply_is_returned_as_a_grounded_answer() -> None:
    recorder = Recorder()
    gateway, _, client = make_gateway(recorder)

    answer = gateway.answer(QUESTION, EVIDENCE)
    client.close()

    assert isinstance(answer, GroundedAnswer)
    assert answer.grounded is True
    assert answer.citations[0].source_id == "doc-1"


def test_telemetry_uses_the_providers_counts_not_the_estimate() -> None:
    recorder = Recorder()
    gateway, telemetry, client = make_gateway(recorder, counter=FixedCounter(100))
    extra = client.estimate_extra_tokens(structured=True)

    gateway.answer(QUESTION, EVIDENCE)
    client.close()

    record = telemetry.records[0]
    assert record.outcome == "answered"
    assert record.estimated_input_tokens == 100 + extra
    assert (record.input_tokens, record.output_tokens) == (812, 57)
    assert record.attempts == 1
    assert record.model == MODEL
    assert record.latency_seconds >= 0.0


def test_the_recorded_cost_comes_from_the_reviewed_rate_table() -> None:
    recorder = Recorder()
    gateway, telemetry, client = make_gateway(recorder)

    gateway.answer(QUESTION, EVIDENCE)
    client.close()

    expected = 812 / 1_000_000 * 2.50 + 57 / 1_000_000 * 10.00
    assert telemetry.records[0].cost_usd == pytest.approx(expected)


def test_an_unpriced_model_still_answers() -> None:
    """A rate table that has not caught up must not fail a working request.

    Note which model gets priced: the one the response says served the request,
    not the alias that was asked for. An alias resolving to a dated snapshot is
    routine, and pricing the alias would price something that did not answer.
    """
    unreviewed = "a-model-nobody-reviewed"
    body = reply()
    body["model"] = unreviewed
    recorder = Recorder(httpx.Response(200, json=body))
    telemetry = InMemoryTelemetry()
    client = OpenAIChatClient(
        api_key=API_KEY,
        model=unreviewed,
        transport=httpx.MockTransport(recorder),
    )
    gateway = LLMGateway(client, context_window=128_000, telemetry=telemetry)

    answer = gateway.answer(QUESTION, EVIDENCE)
    client.close()

    assert answer.grounded is True
    assert telemetry.records[0].cost_usd is None
    assert telemetry.unpriced_calls == 1


def test_whole_document_evidence_is_cited_by_a_stable_tag() -> None:
    recorder = Recorder()
    gateway, _, client = make_gateway(recorder)

    gateway.answer(QUESTION, EVIDENCE)
    client.close()

    body = json.loads(recorder.requests[0].content)
    evidence_block = body["messages"][3]["content"]
    assert f"[doc-1#{WHOLE_DOCUMENT}]" in evidence_block
    assert "Refunds close after 30 days." in evidence_block


def test_located_snippets_are_passed_through_unchanged() -> None:
    """Retrieval will hand over real chunk locators rather than whole documents."""
    recorder = Recorder()
    gateway, _, client = make_gateway(recorder)

    gateway.answer(
        QUESTION,
        [
            EvidenceSnippet(
                source_id="policy.md", locator="3.2", text="Refunds close after 30 days."
            )
        ],
    )
    client.close()

    body = json.loads(recorder.requests[0].content)
    assert "[policy.md#3.2]" in body["messages"][3]["content"]


def test_the_gateway_works_with_any_port_conforming_client() -> None:
    """It depends on the port; the adapter is one implementation of it."""
    telemetry = InMemoryTelemetry()
    client = FakePortClient()
    gateway = LLMGateway(client, context_window=128_000, telemetry=telemetry)

    answer = gateway.answer(QUESTION, EVIDENCE)

    assert answer.grounded is True
    assert client.calls == 1
    assert telemetry.records[0].input_tokens == 10


def test_a_client_that_does_not_satisfy_token_estimating_still_estimates() -> None:
    """TokenEstimating is optional, the same way UsageReporting is: a client
    written before it existed is still a valid one, merely estimated a
    little low exactly as it always was."""
    telemetry = InMemoryTelemetry()
    counter = FixedCounter(100)
    gateway = LLMGateway(
        FakePortClient(), context_window=128_000, telemetry=telemetry, counter=counter
    )

    gateway.answer(QUESTION, EVIDENCE)

    assert telemetry.records[0].estimated_input_tokens == 100


# --------------------------------------------------------------------------
# Step 2's guardrail, firing inside the assembly
# --------------------------------------------------------------------------


def test_an_unbacked_answer_is_rejected_by_the_schema() -> None:
    recorder = Recorder(httpx.Response(200, json=reply(UNGROUNDED_JSON)))
    gateway, telemetry, client = make_gateway(recorder)

    with pytest.raises(ValidationError):
        gateway.answer(QUESTION, EVIDENCE)
    client.close()

    assert [record.outcome for record in telemetry.records] == ["invalid_schema"]


def test_a_rejected_reply_is_still_recorded_as_spent() -> None:
    """It was generated and billed; a log that omits it understates the run."""
    recorder = Recorder(httpx.Response(200, json=reply(UNGROUNDED_JSON)))
    gateway, telemetry, client = make_gateway(recorder)

    with pytest.raises(ValidationError):
        gateway.answer(QUESTION, EVIDENCE)
    client.close()

    record = telemetry.records[0]
    assert (record.input_tokens, record.output_tokens) == (812, 57)
    assert record.cost_usd is not None and record.cost_usd > 0
    assert record.detail is not None


def test_prose_instead_of_json_is_a_validation_failure_not_a_retry() -> None:
    recorder = Recorder(httpx.Response(200, json=reply("Sure! Here is the answer.")))
    gateway, telemetry, client = make_gateway(recorder)

    with pytest.raises(ValidationError):
        gateway.answer(QUESTION, EVIDENCE)
    client.close()

    assert len(recorder.requests) == 1  # not resampled
    assert telemetry.records[0].outcome == "invalid_schema"


# --------------------------------------------------------------------------
# Step 7's retry, firing inside the assembly
# --------------------------------------------------------------------------


def test_transient_failures_are_retried_until_one_succeeds() -> None:
    delays: list[float] = []
    recorder = Recorder(
        httpx.Response(500, json={"error": {"message": "boom"}}),
        httpx.Response(503, json={"error": {"message": "still boom"}}),
        httpx.Response(200, json=reply()),
    )
    gateway, telemetry, client = make_gateway(
        recorder, sleep=delays.append, jitter=lambda: 1.0
    )

    answer = gateway.answer(QUESTION, EVIDENCE)
    client.close()

    assert answer.grounded is True
    assert len(recorder.requests) == 3
    assert delays == [0.5, 1.0]
    assert telemetry.records[0].attempts == 3


def test_exhausted_retries_raise_and_are_recorded() -> None:
    recorder = Recorder(httpx.Response(500, json={"error": {"message": "boom"}}))
    gateway, telemetry, client = make_gateway(recorder, max_attempts=3)

    with pytest.raises(TransientProviderError):
        gateway.answer(QUESTION, EVIDENCE)
    client.close()

    record = telemetry.records[0]
    assert record.outcome == "provider_failure"
    assert record.attempts == 3
    assert len(recorder.requests) == 3


def test_a_failed_run_records_the_tokens_the_provider_already_billed() -> None:
    """The one case last_usage exists for: attempts were billed, and the
    exception replaced the response that would have reported them."""
    recorder = Recorder(
        # Generated and billed, then rejected for a content filter...
        httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "content_filter",
                    }
                ],
                "usage": USAGE,
            },
        ),
        # ...then the retry never gets a usable reply either.
        httpx.Response(500, json={"error": {"message": "boom"}}),
    )
    gateway, telemetry, client = make_gateway(recorder, max_attempts=2)

    with pytest.raises(TransientProviderError):
        gateway.answer(QUESTION, EVIDENCE)
    client.close()

    record = telemetry.records[0]
    assert record.outcome == "provider_failure"
    assert (record.input_tokens, record.output_tokens) == (812, 57)
    assert record.cost_usd is not None and record.cost_usd > 0


def test_an_auth_failure_is_not_retried_inside_the_gateway_either() -> None:
    recorder = Recorder(httpx.Response(401, json={"error": {"message": "no"}}))
    gateway, telemetry, client = make_gateway(recorder)

    with pytest.raises(ProviderAuthError):
        gateway.answer(QUESTION, EVIDENCE)
    client.close()

    assert len(recorder.requests) == 1
    assert telemetry.records[0].outcome == "provider_failure"
    assert telemetry.records[0].attempts == 1


def test_a_blank_question_never_reaches_the_provider() -> None:
    recorder = Recorder()
    gateway, telemetry, client = make_gateway(recorder)

    with pytest.raises(ValueError):
        gateway.answer("   ", EVIDENCE)
    client.close()

    assert recorder.requests == []
    assert telemetry.records == []


# --------------------------------------------------------------------------
# decide(): the same four steps, asking a different question
# --------------------------------------------------------------------------


def tool_call_reply(name: str, arguments: dict) -> dict:
    return {
        "id": "chatcmpl-test",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": USAGE,
    }


def test_a_decision_comes_back_as_the_call_the_model_chose() -> None:
    recorder = Recorder(
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"}))
    )
    gateway, _, _ = make_gateway(recorder)

    decision = gateway.decide("What could go wrong on atlas?")

    assert decision.tool_name == "list_risks"
    assert decision.arguments == {"project_id": "atlas"}
    assert decision.content is None


def test_the_planner_request_carries_the_offered_functions() -> None:
    recorder = Recorder(
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"}))
    )
    gateway, _, _ = make_gateway(recorder)

    gateway.decide("What could go wrong on atlas?")

    body = json.loads(recorder.requests[0].content)
    offered = {tool["function"]["name"] for tool in body["tools"]}
    assert "search_project_documents" in offered
    assert "create_risk" in offered
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is False


def test_decide_sends_the_production_contract_by_default() -> None:
    """The default sends PLANNER_CONTRACT byte-for-byte -- eval/routing.py's
    baseline (ADR 0020) imports it rather than copying it precisely so this
    stays true."""
    from agentic_erp_assistant.llm.prompts import PLANNER_CONTRACT

    recorder = Recorder(
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"}))
    )
    gateway, _, _ = make_gateway(recorder)

    gateway.decide("What could go wrong on atlas?")

    body = json.loads(recorder.requests[0].content)
    developer_block = next(m["content"] for m in body["messages"] if m["role"] == "developer")
    assert developer_block == PLANNER_CONTRACT


def test_decide_sends_a_different_contract_when_the_gateway_carries_one() -> None:
    candidate = "a candidate contract, not the production one"
    recorder = Recorder(
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"}))
    )
    gateway, _, _ = make_gateway(recorder, planner_contract=candidate)

    gateway.decide("What could go wrong on atlas?")

    body = json.loads(recorder.requests[0].content)
    developer_block = next(m["content"] for m in body["messages"] if m["role"] == "developer")
    assert developer_block == candidate


def test_allow_tools_false_forces_the_wire_choice_and_says_so_in_telemetry() -> None:
    """ADR 0019: how ``engine/nodes.py`` ends a turn's planning loop after a
    mutating tool has already succeeded -- by withholding the option, not by
    asking in prose."""
    recorder = Recorder(httpx.Response(200, json=reply("Recorded R-6 against orion.")))
    gateway, telemetry, _ = make_gateway(recorder)

    decision = gateway.decide("What could go wrong on atlas?", allow_tools=False)

    assert decision.tool_name is None
    body = json.loads(recorder.requests[0].content)
    assert body["tool_choice"] == "none"
    assert "tools" in body  # still offered -- see the port's docstring
    assert "tool_choice=none" in telemetry.records[0].detail


def test_observations_reach_the_model_in_their_own_block() -> None:
    """The reason a second decision can differ from the first."""
    from agentic_erp_assistant.state.tool_outcome import ToolOutcome

    recorder = Recorder(httpx.Response(200, json=reply("Two risks are open.")))
    gateway, _, _ = make_gateway(recorder)

    gateway.decide(
        "What could go wrong on atlas?",
        observations=(
            ToolOutcome(
                tool_name="list_risks",
                status="ok",
                summary="2 open",
                source_ids=("atlas",),
            ),
        ),
    )

    body = json.loads(recorder.requests[0].content)
    blocks = [message["content"] for message in body["messages"]]
    assert any("list_risks -> ok: 2 open" in block for block in blocks)


def test_content_with_no_call_is_the_answer_route() -> None:
    recorder = Recorder(httpx.Response(200, json=reply("Nothing is at risk.")))
    gateway, _, _ = make_gateway(recorder)

    decision = gateway.decide("Anything at risk?")

    assert decision.tool_name is None
    assert decision.content == "Nothing is at risk."


def test_a_routing_call_is_recorded_as_its_own_kind_of_spend() -> None:
    """A cost report that grouped routing under 'answered' would be wrong about
    what the money bought."""
    recorder = Recorder(
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"}))
    )
    gateway, telemetry, _ = make_gateway(recorder)

    gateway.decide("What could go wrong on atlas?")

    assert [record.outcome for record in telemetry.records] == ["routed"]
    assert telemetry.records[0].detail is not None
    assert "list_risks" in telemetry.records[0].detail


def test_a_decision_too_large_to_send_is_refused_before_it_costs_anything() -> None:
    recorder = Recorder()
    gateway, telemetry, _ = make_gateway(
        recorder, context_window=100, counter=FixedCounter(99), output_reserve=10
    )

    with pytest.raises(ContextWindowExceeded):
        gateway.decide("What could go wrong on atlas?")

    assert recorder.requests == []
    assert [record.outcome for record in telemetry.records] == ["budget_exceeded"]


def test_a_failed_decision_is_retried_and_then_recorded() -> None:
    recorder = Recorder(
        httpx.Response(503, text="upstream down"),
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"})),
    )
    gateway, telemetry, _ = make_gateway(recorder)

    decision = gateway.decide("What could go wrong on atlas?")

    assert decision.tool_name == "list_risks"
    assert len(recorder.requests) == 2
    assert [record.outcome for record in telemetry.records] == ["routed"]
    assert telemetry.records[0].attempts == 2


def test_a_text_only_client_cannot_route_and_says_so_before_sending() -> None:
    """A deployment mistake, not a turn that failed."""
    gateway = LLMGateway(FakePortClient(), context_window=128_000)

    with pytest.raises(TypeError, match="ToolCallingClient"):
        gateway.decide("What could go wrong on atlas?")


# --------------------------------------------------------------------------
# call_tools(): the same four steps, against a prompt somebody else built
# --------------------------------------------------------------------------


def test_a_second_caller_gets_the_same_four_steps_decide_gets() -> None:
    """The memory layer asks a different question in a different prompt. The
    reason it goes through here rather than reaching the client is that skipping
    this method would skip the budget check, the retry and the cost record."""
    recorder = Recorder(
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"}))
    )
    gateway, telemetry, _ = make_gateway(recorder)

    result = gateway.call_tools(
        [
            {"role": "system", "content": "You are a memory policy."},
            {"role": "user", "content": "What was worth keeping?"},
        ],
        tools=(LIST_RISKS_TOOL,),
    )

    assert result.tool_name == "list_risks"
    assert [record.outcome for record in telemetry.records] == ["routed"]


def test_call_tools_sends_the_prompt_it_was_given_and_builds_none_of_it() -> None:
    """Building one is where the roles are decided -- which block is instruction
    and which is data -- so each caller makes that decision in the open."""
    recorder = Recorder(
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"}))
    )
    gateway, _, _ = make_gateway(recorder)

    gateway.call_tools(
        [{"role": "user", "content": "the only block"}], tools=(LIST_RISKS_TOOL,)
    )

    body = json.loads(recorder.requests[0].content)
    assert [message["content"] for message in body["messages"]] == ["the only block"]


def test_call_tools_refuses_a_prompt_that_cannot_fit_before_sending() -> None:
    recorder = Recorder(
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"}))
    )
    gateway, telemetry, _ = make_gateway(recorder)
    gateway.context_window = 32

    with pytest.raises(ContextWindowExceeded):
        gateway.call_tools(
            [{"role": "user", "content": "words " * 200}], tools=(LIST_RISKS_TOOL,)
        )

    assert recorder.requests == []
    assert [record.outcome for record in telemetry.records] == ["budget_exceeded"]


def test_decide_is_call_tools_with_the_planner_prompt() -> None:
    """The refactor's whole claim: one path, two callers, no second place where
    a request escapes the four steps."""
    recorder = Recorder(
        httpx.Response(200, json=tool_call_reply("list_risks", {"project_id": "atlas"}))
    )
    gateway, _, _ = make_gateway(recorder)

    gateway.decide("What could go wrong on atlas?")

    body = json.loads(recorder.requests[0].content)
    from agentic_erp_assistant.llm.prompts import PLANNER_CONTRACT

    blocks = [message["content"] for message in body["messages"]]
    assert PLANNER_CONTRACT in blocks
    assert "What could go wrong on atlas?" in blocks


# --------------------------------------------------------------------------
# stream: the gateway streams to a sink it was built with
# --------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.resets = 0

    def delta(self, text: str) -> None:
        self.deltas.append(text)

    def reset(self) -> None:
        self.resets += 1
        self.deltas.clear()


class FakeStreamingClient:
    """A bare port-conforming client that actually streams -- for gateway-
    level behavior (reset-on-retry, which field gets extracted) without
    going through real HTTP or SSE, which the adapter's own tests already
    cover."""

    model_name = "fake-streaming-1"

    def __init__(self, *outcomes: object) -> None:
        self.calls = 0
        self._outcomes = list(outcomes)

    def _next(self) -> object:
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        return outcome

    def complete(self, messages, *, temperature: float, on_delta=None):
        outcome = self._next()
        if isinstance(outcome, Exception):
            raise outcome
        text = outcome
        if on_delta is not None:
            for character in text:
                on_delta(character)
        return {
            "text": text,
            "model": self.model_name,
            "stop_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    def call_with_tools(self, messages, *, tools, temperature: float, on_delta=None):
        outcome = self._next()
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, ToolCallResult):
            return outcome
        content = outcome
        if on_delta is not None:
            for character in content:
                on_delta(character)
        return ToolCallResult.from_content(content)


def test_a_recording_sink_receives_only_the_answer_field_text() -> None:
    client = FakeStreamingClient(GROUNDED_JSON)
    sink = RecordingSink()
    gateway = LLMGateway(client, context_window=128_000, stream=sink)

    answer = gateway.answer(QUESTION, EVIDENCE)

    assert "".join(sink.deltas) == answer.answer == "Refunds close after 30 days."
    assert "doc-1" not in "".join(sink.deltas)


def test_a_failed_attempt_then_a_success_resets_the_sink_exactly_once() -> None:
    client = FakeStreamingClient(
        TransientProviderError("boom"), GROUNDED_JSON
    )
    sink = RecordingSink()
    gateway = LLMGateway(
        client,
        context_window=128_000,
        stream=sink,
        sleep=lambda _: None,
        jitter=lambda: 1.0,
    )

    answer = gateway.answer(QUESTION, EVIDENCE)

    assert sink.resets == 1
    assert "".join(sink.deltas) == answer.answer
    assert client.calls == 2


def test_decide_streams_content_for_a_no_tool_reply() -> None:
    client = FakeStreamingClient("Nothing is at risk.")
    sink = RecordingSink()
    gateway = LLMGateway(client, context_window=128_000, stream=sink)

    decision = gateway.decide("Anything at risk?")

    assert decision.tool_name is None
    assert "".join(sink.deltas) == "Nothing is at risk."


def test_decide_streams_nothing_for_a_tool_call() -> None:
    client = FakeStreamingClient(
        ToolCallResult.from_tool_call(
            tool_name="list_risks", arguments={"project_id": "atlas"}
        )
    )
    sink = RecordingSink()
    gateway = LLMGateway(client, context_window=128_000, stream=sink)

    decision = gateway.decide("What could go wrong on atlas?")

    assert decision.tool_name == "list_risks"
    assert sink.deltas == []


def test_a_gateway_without_a_sink_never_passes_on_delta() -> None:
    """FakePortClient's complete() has no on_delta parameter at all; if the
    gateway ever passed the keyword regardless of whether a sink is bound,
    this would raise TypeError instead of answering."""
    client = FakePortClient()
    gateway = LLMGateway(client, context_window=128_000)  # stream=None, the default

    answer = gateway.answer(QUESTION, EVIDENCE)

    assert answer.grounded is True
