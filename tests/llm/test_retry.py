"""The backoff schedule is a promise, so it is asserted rather than described.

Every test injects ``sleep`` and ``jitter``. Nothing here waits: the delays are
recorded into a list and compared exactly. A retry suite that actually slept
would take seconds to prove arithmetic, and would be the first thing someone
deletes when the suite gets slow.

The other half of what is proven here is a negative: that a failure which is not
:class:`TransientProviderError` never causes a second call. That one matters
more than the schedule -- retrying a rejected key or a malformed prompt spends
real money to arrive at the same answer.
"""

import random

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.llm.ports import (
    ClientConfigurationError,
    ProviderAuthError,
    TransientProviderError,
)
from agentic_erp_assistant.llm.retry import retry_with_backoff
from agentic_erp_assistant.llm.schemas import GroundedAnswer

NO_JITTER = 1.0
"""A jitter factor of 1.0 leaves the nominal schedule visible in the assertions."""


class Operation:
    """A callable that fails a chosen number of times, then succeeds.

    Records its own call count, so "was this retried?" is answerable without
    reading the delay list.
    """

    def __init__(self, *, failures: int, error: Exception | None = None) -> None:
        self.calls = 0
        self._failures = failures
        self._error = error
        self.raised: list[Exception] = []

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self._failures:
            error = self._error or TransientProviderError(
                f"503 from test-model (attempt {self.calls})"
            )
            self.raised.append(error)
            raise error
        return "result"


def sequence(*factors: float):
    """A jitter that returns each factor in turn, then repeats the last."""
    drawn = []

    def jitter() -> float:
        value = factors[min(len(drawn), len(factors) - 1)]
        drawn.append(value)
        return value

    jitter.drawn = drawn  # type: ignore[attr-defined]
    return jitter


# --------------------------------------------------------------------------
# The schedule
# --------------------------------------------------------------------------


def test_a_successful_call_is_not_delayed_at_all() -> None:
    delays: list[float] = []
    operation = Operation(failures=0)

    result = retry_with_backoff(
        operation, sleep=delays.append, jitter=lambda: NO_JITTER
    )

    assert result == "result"
    assert operation.calls == 1
    assert delays == []


def test_the_documented_schedule_doubles_up_to_the_attempt_budget() -> None:
    """The unit's own example: four attempts at a one-second base."""
    delays: list[float] = []
    operation = Operation(failures=99)

    with pytest.raises(TransientProviderError):
        retry_with_backoff(
            operation,
            max_attempts=4,
            base_delay_seconds=1.0,
            sleep=delays.append,
            jitter=lambda: NO_JITTER,
        )

    assert delays == [1.0, 2.0, 4.0]
    assert operation.calls == 4


def test_the_default_budget_is_four_attempts_at_half_a_second() -> None:
    delays: list[float] = []

    with pytest.raises(TransientProviderError):
        retry_with_backoff(
            Operation(failures=99), sleep=delays.append, jitter=lambda: NO_JITTER
        )

    assert delays == [0.5, 1.0, 2.0]


def test_retrying_stops_as_soon_as_the_operation_succeeds() -> None:
    delays: list[float] = []
    operation = Operation(failures=2)

    result = retry_with_backoff(
        operation,
        max_attempts=4,
        base_delay_seconds=1.0,
        sleep=delays.append,
        jitter=lambda: NO_JITTER,
    )

    assert result == "result"
    assert operation.calls == 3
    assert delays == [1.0, 2.0]


def test_the_delay_is_capped_and_stays_capped() -> None:
    delays: list[float] = []

    with pytest.raises(TransientProviderError):
        retry_with_backoff(
            Operation(failures=99),
            max_attempts=6,
            base_delay_seconds=1.0,
            max_delay_seconds=4.0,
            sleep=delays.append,
            jitter=lambda: NO_JITTER,
        )

    assert delays == [1.0, 2.0, 4.0, 4.0, 4.0]


def test_one_attempt_never_sleeps() -> None:
    delays: list[float] = []
    operation = Operation(failures=1)

    with pytest.raises(TransientProviderError):
        retry_with_backoff(
            operation, max_attempts=1, sleep=delays.append, jitter=lambda: NO_JITTER
        )

    assert operation.calls == 1
    assert delays == []


# --------------------------------------------------------------------------
# Jitter
# --------------------------------------------------------------------------


def test_the_jitter_factor_scales_every_delay() -> None:
    delays: list[float] = []

    with pytest.raises(TransientProviderError):
        retry_with_backoff(
            Operation(failures=99),
            max_attempts=4,
            base_delay_seconds=1.0,
            sleep=delays.append,
            jitter=lambda: 0.5,
        )

    assert delays == [0.5, 1.0, 2.0]


def test_each_delay_draws_its_own_jitter() -> None:
    """One draw reused across the run would keep callers in lockstep, which is
    the entire failure the jitter exists to prevent."""
    delays: list[float] = []
    jitter = sequence(1.0, 0.25, 0.5)

    with pytest.raises(TransientProviderError):
        retry_with_backoff(
            Operation(failures=99),
            max_attempts=4,
            base_delay_seconds=1.0,
            sleep=delays.append,
            jitter=jitter,
        )

    assert jitter.drawn == [1.0, 0.25, 0.5]  # type: ignore[attr-defined]
    assert delays == [1.0, 0.5, 2.0]


def test_the_cap_is_applied_before_the_jitter_not_after() -> None:
    """Capping afterwards would pin late attempts to exactly max_delay_seconds
    and re-synchronize everyone who was waiting."""
    delays: list[float] = []

    with pytest.raises(TransientProviderError):
        retry_with_backoff(
            Operation(failures=99),
            max_attempts=4,
            base_delay_seconds=1.0,
            max_delay_seconds=2.0,
            sleep=delays.append,
            jitter=lambda: 0.5,
        )

    # Capped to [1, 2, 2], then halved -- not [0.5, 1.0, 2.0].
    assert delays == [0.5, 1.0, 1.0]


def test_the_default_jitter_spreads_across_the_whole_window() -> None:
    """Full jitter: uniform over [0, cap), not a wobble around the nominal."""
    delays: list[float] = []
    random.seed(20260903)

    with pytest.raises(TransientProviderError):
        retry_with_backoff(
            Operation(failures=99),
            max_attempts=5,
            base_delay_seconds=1.0,
            sleep=delays.append,
        )

    nominal = [1.0, 2.0, 4.0, 8.0]
    assert len(delays) == 4
    assert all(0.0 <= delay < cap for delay, cap in zip(delays, nominal))
    # Not every draw sits in the top half; a half-window jitter could not
    # produce a delay below cap/2.
    assert any(delay < cap / 2 for delay, cap in zip(delays, nominal))


# --------------------------------------------------------------------------
# What is never retried
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ProviderAuthError("401 from test-model"),
        ClientConfigurationError("OPENAI_MODEL is not set."),
        ValueError("a bug in the operation"),
        RuntimeError("something else entirely"),
    ],
    ids=["auth", "configuration", "value", "runtime"],
)
def test_a_non_transient_failure_is_never_retried(error: Exception) -> None:
    delays: list[float] = []
    operation = Operation(failures=99, error=error)

    with pytest.raises(type(error)) as failure:
        retry_with_backoff(
            operation, sleep=delays.append, jitter=lambda: NO_JITTER
        )

    assert failure.value is error
    assert operation.calls == 1
    assert delays == []


def test_a_schema_validation_failure_is_never_retried() -> None:
    """The acceptance criterion, spelled out: a model that answered with an
    ungrounded claim will answer with one again, at four times the cost."""
    delays: list[float] = []
    calls = 0

    def parse_a_bad_answer() -> GroundedAnswer:
        nonlocal calls
        calls += 1
        return GroundedAnswer(answer="Sprint 12 went well.", grounded=True, confidence=0.9)

    with pytest.raises(ValidationError):
        retry_with_backoff(
            parse_a_bad_answer, sleep=delays.append, jitter=lambda: NO_JITTER
        )

    assert calls == 1
    assert delays == []


def test_a_keyboard_interrupt_passes_straight_through() -> None:
    """A user pressing ctrl-c is not a provider hiccup to wait out."""
    delays: list[float] = []
    calls = 0

    def interrupted() -> None:
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        retry_with_backoff(interrupted, sleep=delays.append, jitter=lambda: NO_JITTER)

    assert calls == 1
    assert delays == []


# --------------------------------------------------------------------------
# Exhaustion, observability, and the budget itself
# --------------------------------------------------------------------------


def test_exhaustion_reraises_the_last_failure_not_the_first() -> None:
    """The trace wants the failure that ended the run, with its own status and
    request id -- not a stale copy of how the run started."""
    operation = Operation(failures=99)

    with pytest.raises(TransientProviderError) as failure:
        retry_with_backoff(
            operation,
            max_attempts=3,
            sleep=lambda _: None,
            jitter=lambda: NO_JITTER,
        )

    assert failure.value is operation.raised[-1]
    assert "attempt 3" in str(failure.value)


def test_the_exhausted_error_records_how_many_attempts_were_spent() -> None:
    with pytest.raises(TransientProviderError) as failure:
        retry_with_backoff(
            Operation(failures=99),
            max_attempts=3,
            sleep=lambda _: None,
            jitter=lambda: NO_JITTER,
        )

    assert any("3 attempts" in note for note in failure.value.__notes__)


def test_on_retry_sees_every_retry_before_it_is_slept_off() -> None:
    """The hook the trace layer will use: this module never imports trace."""
    observed: list[tuple[int, float, str]] = []
    order: list[str] = []

    def record(attempt: int, delay: float, error: TransientProviderError) -> None:
        observed.append((attempt, delay, type(error).__name__))
        order.append("retry")

    with pytest.raises(TransientProviderError):
        retry_with_backoff(
            Operation(failures=99),
            max_attempts=3,
            base_delay_seconds=1.0,
            sleep=lambda _: order.append("sleep"),
            jitter=lambda: NO_JITTER,
            on_retry=record,
        )

    assert observed == [
        (1, 1.0, "TransientProviderError"),
        (2, 2.0, "TransientProviderError"),
    ]
    assert order == ["retry", "sleep", "retry", "sleep"]


def test_a_successful_run_notifies_nobody() -> None:
    observed: list[int] = []

    retry_with_backoff(
        Operation(failures=0),
        sleep=lambda _: None,
        jitter=lambda: NO_JITTER,
        on_retry=lambda attempt, delay, error: observed.append(attempt),
    )

    assert observed == []


@pytest.mark.parametrize(
    "budget",
    [
        {"max_attempts": 0},
        {"max_attempts": -1},
        {"base_delay_seconds": -1.0},
        {"max_delay_seconds": -0.5},
    ],
    ids=["zero-attempts", "negative-attempts", "negative-base", "negative-cap"],
)
def test_an_invalid_budget_is_a_caller_bug(budget: dict) -> None:
    operation = Operation(failures=0)

    with pytest.raises(ValueError):
        retry_with_backoff(operation, sleep=lambda _: None, **budget)

    assert operation.calls == 0
