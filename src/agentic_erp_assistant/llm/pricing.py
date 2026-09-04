"""What a request costs in money, from rates a human reviewed and dated.

``tokenizer.py`` says how many tokens a request will take. This says what that
costs, and it is a **config artifact rather than a constant to trust
indefinitely**: published rates change on the vendor's schedule, not ours, so the
table carries the date it was last checked and the page it was read from.

Two things this module refuses to do:

* Blend input and output into one rate. On the model priced here, output bills at
  four times input -- a single blended ``$/1k`` figure is wrong by a factor that
  depends on how talkative the model happened to be.
* Price a model nobody reviewed. An unknown model raises. A silent fallback price
  looks authoritative and is not, which is the worst thing a cost figure can be,
  and it is how a real bill surprises a team.
"""

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "estimate_cost_usd",
    "MODEL_RATES",
    "ModelRate",
    "PRICING_CHECKED_ON",
    "PRICING_SOURCE",
    "UnknownModelRateError",
]


PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
"""Where the rates below were read from."""

PRICING_CHECKED_ON = date(2026, 8, 28)
"""When they were last read.

This date is the point of the file. If it is old, the numbers below may be
fiction, and the only way that becomes visible is a reviewer noticing it here.
"""


@dataclass(frozen=True)
class ModelRate:
    """What one model charges, per million tokens, in USD.

    A dataclass rather than a validated model: these are literals in source that
    a human reviews against a published page. There is nothing to check at
    runtime that review does not already catch.
    """

    input_usd_per_million: float
    output_usd_per_million: float


# Input and output are separately priced, and the gap is large -- output costs 4x
# input here. Anything that collapses them into one number is wrong.
MODEL_RATES: Mapping[str, ModelRate] = MappingProxyType(
    {
        "gpt-4o": ModelRate(input_usd_per_million=2.50, output_usd_per_million=10.00),
    }
)
"""The reviewed rates. Read-only: the table is reviewed, not edited at runtime."""


class UnknownModelRateError(Exception):
    """No reviewed rate exists for the requested model.

    Deliberately not a :class:`KeyError` or :class:`LookupError` subclass. Callers
    routinely wrap dictionary work in ``except KeyError``, and one of them would
    swallow this and carry on with no price -- which is the exact failure the
    raise exists to prevent. Nothing catches this by accident.
    """


def estimate_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate what a call costs in USD, from the dated table above.

    The result is deliberately **not** rounded: rounding belongs at the display
    edge, and rounding here would accumulate error across the many small calls a
    single run makes.

    ``float`` rather than ``Decimal`` because this is a reporting estimate summed
    over a run, not money being moved. ``Decimal`` would advertise an exactness
    that the token count underneath it does not have.

    Args:
        model: Must have a reviewed entry in :data:`MODEL_RATES`.
        input_tokens: Prompt tokens, e.g. from ``count_message_tokens``.
        output_tokens: Completion tokens.

    Raises:
        UnknownModelRateError: No reviewed rate for ``model``. Never a guess,
            never zero, never the nearest entry.
        ValueError: A token count is negative. That is a caller bug, and left
            alone it would produce a negative cost that quietly cancels a real
            one out of a total.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            f"token counts must not be negative: "
            f"input_tokens={input_tokens}, output_tokens={output_tokens}"
        )

    try:
        rate = MODEL_RATES[model]
    except KeyError:
        raise UnknownModelRateError(
            f"no reviewed rate for model {model!r}; add one to MODEL_RATES in "
            f"{__name__} from {PRICING_SOURCE} rather than pricing it by guess"
        ) from None

    return (
        input_tokens / 1_000_000 * rate.input_usd_per_million
        + output_tokens / 1_000_000 * rate.output_usd_per_million
    )
