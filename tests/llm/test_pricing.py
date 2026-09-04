"""The rates are the evidence behind every dollar figure, so the tests pin them.

Two failures matter here and neither announces itself: a cost that silently uses
one rate for both directions, and a cost quietly returned for a model nobody
priced. Both are asserted below.
"""

from datetime import date

import pytest

from agentic_erp_assistant.llm.pricing import (
    MODEL_RATES,
    PRICING_CHECKED_ON,
    PRICING_SOURCE,
    UnknownModelRateError,
    estimate_cost_usd,
)
from agentic_erp_assistant.llm.prompts import build_messages
from agentic_erp_assistant.llm.schemas import EvidenceSnippet
from agentic_erp_assistant.llm.tokenizer import count_message_tokens

MODEL = "gpt-4o"
UNPRICED_MODEL = "gpt-4o-mini"

A_MILLION = 1_000_000


def test_a_million_each_way_costs_the_published_rate() -> None:
    """The named acceptance criterion: $2.50 in + $10.00 out per 1M."""
    cost = estimate_cost_usd(MODEL, input_tokens=A_MILLION, output_tokens=A_MILLION)

    assert cost == pytest.approx(12.50)


def test_input_and_output_are_priced_at_their_own_rates() -> None:
    """Catches the classic slip the combined figure above would not.

    Averaging the two rates still yields $12.50 for a million each way. Pricing
    each direction alone is what proves they were kept apart.
    """
    assert estimate_cost_usd(
        MODEL, input_tokens=A_MILLION, output_tokens=0
    ) == pytest.approx(2.50)
    assert estimate_cost_usd(
        MODEL, input_tokens=0, output_tokens=A_MILLION
    ) == pytest.approx(10.00)


@pytest.mark.parametrize("factor", [2, 10, 1000])
def test_cost_scales_linearly_on_both_axes(factor: int) -> None:
    """The named acceptance criterion."""
    base = estimate_cost_usd(MODEL, input_tokens=1_000, output_tokens=500)

    scaled = estimate_cost_usd(
        MODEL, input_tokens=1_000 * factor, output_tokens=500 * factor
    )

    assert scaled == pytest.approx(base * factor)


def test_an_unpriced_model_raises_instead_of_returning_a_number() -> None:
    """The named acceptance criterion: never zero, never a guessed price."""
    with pytest.raises(UnknownModelRateError, match=UNPRICED_MODEL):
        estimate_cost_usd(UNPRICED_MODEL, input_tokens=1_000, output_tokens=1_000)


def test_the_unknown_rate_error_cannot_be_swallowed_as_a_key_error() -> None:
    """A caller's broad `except KeyError` must not absorb a missing price."""
    assert not issubclass(UnknownModelRateError, KeyError)
    assert not issubclass(UnknownModelRateError, LookupError)


def test_no_tokens_cost_nothing() -> None:
    assert estimate_cost_usd(MODEL, input_tokens=0, output_tokens=0) == 0.0


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [(-1, 0), (0, -1), (-1, -1)],
)
def test_negative_token_counts_are_a_caller_bug(
    input_tokens: int, output_tokens: int
) -> None:
    """Left alone, a negative cost quietly cancels a real one out of a total."""
    with pytest.raises(ValueError, match="negative"):
        estimate_cost_usd(
            MODEL, input_tokens=input_tokens, output_tokens=output_tokens
        )


def test_the_table_carries_its_provenance() -> None:
    """Deliberately not a staleness time-bomb -- the date is for a reviewer."""
    assert isinstance(PRICING_CHECKED_ON, date)
    assert PRICING_CHECKED_ON <= date.today()
    assert PRICING_SOURCE.startswith("https://")


def test_every_entry_has_real_rates() -> None:
    assert MODEL_RATES
    for model, rate in MODEL_RATES.items():
        assert rate.input_usd_per_million > 0, model
        assert rate.output_usd_per_million > 0, model


def test_the_table_is_read_only() -> None:
    """The rates are reviewed, not edited at runtime."""
    with pytest.raises(TypeError):
        MODEL_RATES["gpt-4o"] = None  # type: ignore[index]


def test_a_dollar_figure_traces_back_to_a_real_request() -> None:
    """Real prompt -> real token count -> a dated rate. The whole point."""
    messages = build_messages(
        "When does sprint 12 close?",
        [
            EvidenceSnippet(
                source_id="doc-12",
                locator="3.2",
                text="Sprint 12 closes on 30 September.",
            )
        ],
    )
    tokens = count_message_tokens(messages, model=MODEL)

    cost = estimate_cost_usd(MODEL, input_tokens=tokens, output_tokens=0)

    assert cost == pytest.approx(tokens / A_MILLION * 2.50)
    assert 0 < cost < 0.01
