"""The window arithmetic, on its own, against a clock the test moves by hand.

The gateway tests prove the limit fires in the right place in the pipeline.
These prove the counter underneath it is a sliding log and not a fixed window --
which is invisible from the gateway until the boundary case that a fixed window
gets wrong actually happens.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agentic_erp_assistant.tools.limits import InMemoryRateLimiter, RateLimiter
from agentic_erp_assistant.tools.registry import RateLimitPolicy

WHEN = datetime(2026, 9, 5, 9, 30, tzinfo=timezone.utc)

ACTOR = "pm@example.com"
TOOL = "get_project_status"

ONE_PER_MINUTE = RateLimitPolicy(max_calls=1, per_seconds=60.0)
THREE_PER_MINUTE = RateLimitPolicy(max_calls=3, per_seconds=60.0)


@pytest.fixture
def limiter() -> InMemoryRateLimiter:
    return InMemoryRateLimiter()


def at(seconds: float) -> datetime:
    return WHEN + timedelta(seconds=seconds)


def test_the_default_satisfies_the_port() -> None:
    """Structurally, so a Redis-backed one can replace it without the gateway
    knowing which it has."""
    assert isinstance(InMemoryRateLimiter(), RateLimiter)


def test_an_untouched_key_has_its_whole_budget(limiter: InMemoryRateLimiter) -> None:
    assert limiter.check(ACTOR, TOOL, THREE_PER_MINUTE, WHEN) is None


def test_checking_does_not_count(limiter: InMemoryRateLimiter) -> None:
    """A check with a side effect makes "asked and refused" indistinguishable
    from "ran", and the gateway checks before it knows whether it will run."""
    for _ in range(10):
        limiter.check(ACTOR, TOOL, ONE_PER_MINUTE, WHEN)

    assert limiter.check(ACTOR, TOOL, ONE_PER_MINUTE, WHEN) is None


def test_the_budget_is_spent_exactly_at_the_declared_count(
    limiter: InMemoryRateLimiter,
) -> None:
    for _ in range(3):
        assert limiter.check(ACTOR, TOOL, THREE_PER_MINUTE, WHEN) is None
        limiter.record(ACTOR, TOOL, THREE_PER_MINUTE, WHEN)

    assert limiter.check(ACTOR, TOOL, THREE_PER_MINUTE, WHEN) == 60.0


def test_the_wait_is_until_the_oldest_call_ages_out(
    limiter: InMemoryRateLimiter,
) -> None:
    """Not until some window resets. The distinction is the whole reason this
    is a log: a fixed window knows when it rolls over, which is not the same
    moment capacity returns, and the number it reports is therefore a guess."""
    limiter.record(ACTOR, TOOL, ONE_PER_MINUTE, WHEN)

    assert limiter.check(ACTOR, TOOL, ONE_PER_MINUTE, at(20)) == 40.0
    assert limiter.check(ACTOR, TOOL, ONE_PER_MINUTE, at(59)) == 1.0


def test_waiting_exactly_as_long_as_told_is_enough(
    limiter: InMemoryRateLimiter,
) -> None:
    limiter.record(ACTOR, TOOL, ONE_PER_MINUTE, WHEN)
    wait = limiter.check(ACTOR, TOOL, ONE_PER_MINUTE, at(10))

    assert wait == 50.0
    assert limiter.check(ACTOR, TOOL, ONE_PER_MINUTE, at(10 + wait)) is None


def test_calls_age_out_one_at_a_time_rather_than_all_at_once(
    limiter: InMemoryRateLimiter,
) -> None:
    """The behaviour a fixed window cannot express, and the reason it allows a
    double burst across its boundary: here the third call still holds the
    budget after the first two have expired."""
    limiter.record(ACTOR, TOOL, THREE_PER_MINUTE, at(0))
    limiter.record(ACTOR, TOOL, THREE_PER_MINUTE, at(1))
    limiter.record(ACTOR, TOOL, THREE_PER_MINUTE, at(50))

    # 61s in, the first two have aged out and the third has not.
    assert limiter.check(ACTOR, TOOL, THREE_PER_MINUTE, at(61)) is None
    limiter.record(ACTOR, TOOL, THREE_PER_MINUTE, at(61))
    limiter.record(ACTOR, TOOL, THREE_PER_MINUTE, at(62))

    assert limiter.check(ACTOR, TOOL, THREE_PER_MINUTE, at(63)) == 47.0


def test_the_log_never_grows_past_the_budget(limiter: InMemoryRateLimiter) -> None:
    """Bounded by ``max_calls`` per key, pruned on every touch -- otherwise a
    counter kept for auditability becomes a leak."""
    for second in range(500):
        limiter.record(ACTOR, TOOL, THREE_PER_MINUTE, at(second))

    assert len(limiter._calls[(ACTOR, TOOL)]) <= THREE_PER_MINUTE.max_calls


def test_two_actors_hold_separate_budgets(limiter: InMemoryRateLimiter) -> None:
    limiter.record(ACTOR, TOOL, ONE_PER_MINUTE, WHEN)

    assert limiter.check("lead@example.com", TOOL, ONE_PER_MINUTE, WHEN) is None


def test_two_tools_hold_separate_budgets(limiter: InMemoryRateLimiter) -> None:
    limiter.record(ACTOR, TOOL, ONE_PER_MINUTE, WHEN)

    assert limiter.check(ACTOR, "list_risks", ONE_PER_MINUTE, WHEN) is None


def test_the_same_sequence_against_the_same_clock_refuses_the_same_calls() -> None:
    """Determinism, stated as a test rather than as a comment: nothing here
    reads a clock or draws a random number, so two identical runs agree."""
    runs = []
    for _ in range(2):
        limiter = InMemoryRateLimiter()
        waits = []
        for second in (0, 1, 2, 30, 61, 62):
            wait = limiter.check(ACTOR, TOOL, THREE_PER_MINUTE, at(second))
            waits.append(wait)
            if wait is None:
                limiter.record(ACTOR, TOOL, THREE_PER_MINUTE, at(second))
        runs.append(waits)

    assert runs[0] == runs[1]
