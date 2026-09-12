"""Does the budget estimate actually track the invoice? (gap-plan.md gap 12,
Phase K2, closing E4 from the manual walkthrough.)

Deliberately outside ``tests/llm/``: that directory's own ``conftest.py``
neutralizes ``OPENAI_API_KEY`` and ``.env`` loading for every test under it,
by design ("nothing here should be able to reach the network even by
accident") -- exactly what this test needs to do on purpose. Marked ``live``
(``pyproject.toml``) rather than ``postgres``'s pattern of a fixture that
skips: there is no server to be "not up" here, only a key to be missing, so
the skip is a plain check at the top of each test.

    uv run pytest -m live -q
"""

import os

import pytest
from dotenv import load_dotenv

from agentic_erp_assistant.llm.adapters.openai_chat import OpenAIChatClient
from agentic_erp_assistant.llm.gateway import LLMGateway
from agentic_erp_assistant.llm.telemetry import InMemoryTelemetry
from agentic_erp_assistant.state.evidence import EvidenceSnippet

pytestmark = pytest.mark.live

PLANNER_DRIFT_TOLERANCE = 0.05
"""Measured on 2026-09-12 against gpt-4o-2024-08-06 (Phase K1's estimate,
which now counts the tools array): a fixed planner call landed at +1.3%
drift, deterministically -- input tokens for a fixed prompt do not vary
run to run, only the model's own reply does. Comfortably inside 5%; the
remainder is tiktoken's local count approximating the provider's own
tokenizer rather than matching it byte for byte, the documented cost of not
calling a token-counting endpoint per request before every real one (ADR
0001 rejects that trade explicitly)."""

ANSWER_DRIFT_TOLERANCE = 0.10
"""Measured the same day, same model, for a fixed answering call: +6.9%
drift, just as deterministic. Wider than the planner's bound, and this is a
finding, not a shrug: decomposed live (base messages + the full wrapped
``response_format`` payload, both counted exactly as sent, compact JSON
confirmed byte-for-byte against what ``httpx`` actually puts on the wire),
133 of ~1,923 tokens are still unaccounted for. Nothing in the request
explains them -- ``strict: false`` is set precisely so the model is not
asked to satisfy a compiled constrained grammar, yet OpenAI's structured
outputs still appear to carry some fixed processing overhead invisible to
any byte the client sends. Undocumented by OpenAI as of this writing, so
closing this gap further would mean baking in a guessed constant -- exactly
what ``llm/pricing.py`` refuses to do for cost, and what this estimate must
refuse to do for tokens. 10% is the honest bound this measurement supports;
tightening it later needs a citation, not an adjustment."""


def _skip_without_key() -> None:
    load_dotenv(override=False)
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        pytest.skip(
            "no OPENAI_API_KEY: set it in .env to run the live estimate-drift check"
        )


def _assert_within_tolerance(record, tolerance: float) -> None:
    assert record.input_tokens > 0, "no invoice to compare the estimate against"
    drift = abs(record.estimated_input_tokens - record.input_tokens) / record.input_tokens
    assert drift < tolerance, (
        f"estimated={record.estimated_input_tokens} actual={record.input_tokens} "
        f"drift={drift:.1%} (tolerance {tolerance:.0%})"
    )


def test_a_live_planner_calls_estimate_tracks_the_invoice() -> None:
    """decide(): the tool-offering call K1 was written for."""
    _skip_without_key()
    telemetry = InMemoryTelemetry()
    client = OpenAIChatClient()
    gateway = LLMGateway(client, context_window=128_000, telemetry=telemetry)

    gateway.decide("What is the status of milestone M2?")
    client.close()

    _assert_within_tolerance(telemetry.records[0], PLANNER_DRIFT_TOLERANCE)


def test_a_live_answering_calls_estimate_tracks_the_invoice() -> None:
    """answer(): the structured-output call K1 was written for. See
    ANSWER_DRIFT_TOLERANCE's docstring for why this bound is wider than the
    planner's, and why it is not tighter."""
    _skip_without_key()
    telemetry = InMemoryTelemetry()
    client = OpenAIChatClient()
    gateway = LLMGateway(client, context_window=128_000, telemetry=telemetry)

    gateway.answer(
        "Why is milestone M2 late?",
        [
            EvidenceSnippet(
                source_id="status-report-2026-09",
                locator="§2.2",
                text=(
                    "Milestone M2 is two days late due to 41 reconciliation "
                    "exceptions in the cost-centre mapping."
                ),
            )
        ],
    )
    client.close()

    _assert_within_tolerance(telemetry.records[0], ANSWER_DRIFT_TOLERANCE)
