"""What one model call cost, recorded whether or not the answer was any good.

``tokenizer.py`` estimates a request before it is sent. ``pricing.py`` turns
token counts into money. This module is where those two meet the thing that
actually happened, one record per call, so a run can be reviewed afterwards
rather than guessed at.

Three choices shape the record:

* **The estimate and the real count are both kept.** They are different numbers
  for good reasons -- the port's ``developer`` and ``evidence`` roles are folded
  before they reach the wire, and no local tokenizer knows what a provider adds
  server-side. Keeping only one of them means never finding out that the
  estimate has drifted, which is the number the budget check is made of.
* **A call that failed is still recorded.** A reply rejected by the schema, or a
  retry that ran out of attempts, was still generated and still billed. A
  telemetry log that only holds successes reports a run as cheaper than it was.
* **An unpriced model records ``cost_usd=None``, not zero.** ``pricing.py``
  refuses to guess a rate and this module refuses to invent one, but a missing
  row in a rate table is not a reason to fail a request that worked. ``None``
  says "not known"; ``0.0`` would say "free", and one of those is a lie.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable

from agentic_erp_assistant.llm.pricing import UnknownModelRateError, estimate_cost_usd

__all__ = [
    "InMemoryTelemetry",
    "ModelCallRecord",
    "now",
    "Outcome",
    "price",
    "TelemetrySink",
]

logger = logging.getLogger(__name__)


Outcome = Literal[
    "answered",  # a reply came back and satisfied the schema
    "routed",  # a decision came back: one tool call, or content to answer with
    "invalid_schema",  # a reply came back and did not
    "budget_exceeded",  # refused locally; nothing was sent
    "provider_failure",  # the call never produced a usable reply
]
"""How a call ended.

A closed set rather than a free string: these are the buckets a cost report
groups by, and an unrecognized label would silently become its own category.
"""


@dataclass(frozen=True)
class ModelCallRecord:
    """One call's cost, latency and outcome.

    Frozen: a record describes something that already happened, and editing it
    after the fact would make the log a worse witness than no log.
    """

    model: str
    """The model that answered, as the provider reported it."""

    outcome: Outcome
    """How it ended. See :data:`Outcome`."""

    estimated_input_tokens: int
    """What ``tokenizer.py`` predicted before sending. The budget check's number."""

    input_tokens: int
    """What the provider billed for the prompt. ``0`` when nothing was sent."""

    output_tokens: int
    """What the provider billed for the completion. ``0`` when nothing was sent."""

    cost_usd: float | None
    """The priced total, or ``None`` when no reviewed rate exists for the model."""

    latency_seconds: float
    """Wall clock for the whole attempt sequence, retries and waiting included."""

    attempts: int
    """How many times the provider was called. ``0`` for a request refused locally."""

    occurred_at: datetime
    """When the call finished, in UTC."""

    detail: str | None = None
    """One line of context for a non-``answered`` outcome: the validation error,
    the provider's message, or why the price is unknown."""

    @property
    def estimate_error(self) -> int | None:
        """Estimate minus reality, or ``None`` when nothing was sent.

        The number that says whether the budget check can still be trusted. A
        consistently negative value means requests are being waved through that
        should have been refused.
        """
        if self.input_tokens == 0:
            return None
        return self.estimated_input_tokens - self.input_tokens


def price(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float | None, str | None]:
    """Price a call, or say why it could not be priced.

    Returns:
        ``(cost, detail)``. ``cost`` is ``None`` when the model has no reviewed
        rate, and ``detail`` then carries the reason -- surfaced in the record
        rather than raised, because a rate table that has not caught up with a
        model choice must not fail a request that otherwise worked.
    """
    try:
        return estimate_cost_usd(
            model, input_tokens=input_tokens, output_tokens=output_tokens
        ), None
    except UnknownModelRateError as error:
        logger.warning("cost unknown for model %r: %s", model, error)
        return None, f"cost unknown: {error}"


@runtime_checkable
class TelemetrySink(Protocol):
    """Where records go.

    A protocol with one method, because the real sink is the trace store that
    ``trace/`` will own, and this module must not import it -- the dependency
    would point outward from the core. Until then :class:`InMemoryTelemetry` is
    enough to assert against in tests.
    """

    def record(self, record: ModelCallRecord) -> None:
        """Persist one record. Must not raise: telemetry never fails a request."""
        ...


@dataclass
class InMemoryTelemetry:
    """Keeps records in a list. The default sink, and the one tests read."""

    records: list[ModelCallRecord] = field(default_factory=list)

    def record(self, record: ModelCallRecord) -> None:
        self.records.append(record)

    @property
    def total_cost_usd(self) -> float:
        """What the priced calls came to.

        Unpriced calls are skipped rather than counted as zero -- and
        :attr:`unpriced_calls` is how a reader learns the total is incomplete.
        """
        return sum(r.cost_usd for r in self.records if r.cost_usd is not None)

    @property
    def unpriced_calls(self) -> int:
        """How many records carry no price. A non-zero value makes
        :attr:`total_cost_usd` a lower bound, not a total."""
        return sum(1 for r in self.records if r.cost_usd is None)


def now() -> datetime:
    """Timezone-aware UTC, in one place so records cannot disagree."""
    return datetime.now(UTC)
