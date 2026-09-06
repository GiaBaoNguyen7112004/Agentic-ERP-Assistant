"""Retry as a policy, not as a reflex.

An adapter deliberately does not retry: it classifies a failure and returns.
This module is the other half of that split -- it decides whether a failure is
worth another attempt, and how long to wait first. Keeping them apart means the
decision is in one readable place, with a budget, instead of being a
``try/except`` that has quietly grown inside every call site.

Two rules do the work here.

**Exactly one failure type is retried, and the caller names it.** The
``except`` clause is bound to a single type -- :class:`TransientProviderError`
by default -- so nothing else is even considered: not a rejected key, not a
schema-validation failure, not a bug in the operation. That is the whole
enforcement: a broad catch with an ``isinstance`` filter afterwards is one edit
away from swallowing something it should not, and a prompt that produces
invalid output produces it just as invalidly the fourth time, having burned
three times the money and four times the latency to prove it.

``retry_on`` exists because the tool boundary has the same problem and deserves
the same answer. A tool handler raises
:class:`~agentic_erp_assistant.tools.models.TransientToolError`, which is not a
provider failure and must not be made a subclass of one just to reuse this
loop. A second retry loop written next door would be the alternative, and it
would be the one that drifts -- different jitter, different logging, a budget
nobody audits. One parameter is cheaper than two implementations. It stays a
single type, never a tuple: "which failures are worth repeating" is a decision
that should be readable at the call site, and a tuple is where that decision
starts collecting members nobody re-examined.

**The wait is exponential and jittered.** Doubling gives an overloaded provider
room to recover. The jitter is what stops every concurrent caller from waking at
the same instant and re-triggering the same rate limit together -- a fixed
schedule turns one rate limit into a synchronized herd, which is a worse pattern
of load than the one that caused the 429.
"""

import logging
import random
import time
from collections.abc import Callable

from agentic_erp_assistant.llm.ports import TransientProviderError

__all__ = ["retry_with_backoff"]

logger = logging.getLogger(__name__)


def _validate(
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> None:
    """Reject nonsense budgets at the door.

    ``max_attempts=0`` is the one worth being loud about: a loop that runs zero
    times would return ``None`` from a function annotated to return ``T``, and
    the failure would surface somewhere far away as a missing attribute rather
    than here as a caller bug.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    if base_delay_seconds < 0:
        raise ValueError(
            f"base_delay_seconds must not be negative, got {base_delay_seconds}"
        )
    if max_delay_seconds < 0:
        raise ValueError(
            f"max_delay_seconds must not be negative, got {max_delay_seconds}"
        )


def retry_with_backoff[T](
    operation: Callable[[], T],
    *,
    max_attempts: int = 4,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
    sleep: Callable[[float], object] = time.sleep,
    jitter: Callable[[], float] = random.random,
    on_retry: Callable[[int, float, Exception], object] | None = None,
    retry_on: type[Exception] = TransientProviderError,
) -> T:
    """Call ``operation``, retrying only the one failure type named by ``retry_on``.

    The delay before attempt *n* is::

        min(base_delay_seconds * 2 ** (n - 2), max_delay_seconds) * jitter()

    -- the exponential is capped first and jittered second, so the jitter always
    spreads callers across the whole ``[0, cap]`` band. Jittering first and
    capping after would pin every late attempt to exactly ``max_delay_seconds``,
    which reintroduces the lockstep the jitter exists to break.

    ``jitter`` returns a *factor*, and the default is :func:`random.random` --
    full jitter, uniform over the entire window rather than a half-window
    wobble around the nominal delay. A fresh draw is taken for every delay.

    Args:
        operation: A zero-argument callable. Bind arguments with
            ``functools.partial`` or a closure; keeping the signature at zero
            arguments means this function never has to know what it is calling.
        max_attempts: Total attempts, not retries. ``1`` disables retrying and
            never sleeps.
        base_delay_seconds: The first delay, before jitter.
        max_delay_seconds: The ceiling the doubling is clamped to.
        sleep: How to wait. Injectable so a test can record the schedule
            instead of living through it -- proving the exact delays is the
            point, and a suite that actually slept for seconds would be quietly
            deleted by whoever runs it next.
        jitter: The factor generator, in ``[0.0, 1.0)``. Injectable for the same
            reason: a schedule that depends on the global random state is not a
            schedule anyone can assert on.
        retry_on: The single exception type worth another attempt. Defaults to
            :class:`TransientProviderError`, so provider calls need not say it.
            A tool gateway passes
            :class:`~agentic_erp_assistant.tools.models.TransientToolError`.
        on_retry: Called as ``(attempt, delay, error)`` just before each sleep,
            where ``attempt`` is the attempt that just failed. Present so the
            runtime can record every retry in the trace without this module
            importing the trace layer -- the dependency would point the wrong
            way, and a retry that leaves no trace record is exactly the kind of
            silent cost this project is supposed to make visible.

    Returns:
        Whatever ``operation`` returns, from the first attempt that succeeds.

    Raises:
        retry_on: Every attempt failed transiently. The **last**
            failure is re-raised unchanged, carrying the provider's own status,
            request id and message, with a note recording how many attempts were
            spent. It is deliberately not wrapped in a new exception type:
            callers above already know this type from the port or the tool
            boundary, and wrapping would bury the original message one
            ``__cause__`` deeper in the trace.
        ValueError: The retry budget itself is invalid.
        Exception: Anything else ``operation`` raises, on the first attempt,
            untouched.
    """
    _validate(max_attempts, base_delay_seconds, max_delay_seconds)

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except retry_on as error:
            if attempt == max_attempts:
                error.add_note(
                    f"gave up after {max_attempts} "
                    f"attempt{'s' if max_attempts != 1 else ''}"
                )
                raise

            delay = (
                min(base_delay_seconds * 2 ** (attempt - 1), max_delay_seconds)
                * jitter()
            )

            # The exception type and the wait, never the prompt: this line runs
            # on every rate limit, and a log that repeats request content is a
            # log nobody can safely keep.
            logger.warning(
                "attempt %d/%d failed transiently (%s); retrying in %.3fs",
                attempt,
                max_attempts,
                type(error).__name__,
                delay,
            )
            if on_retry is not None:
                on_retry(attempt, delay, error)
            sleep(delay)

    # Unreachable: the loop either returns or raises on the final attempt. Kept
    # so the function has no implicit ``None`` path for a reader to wonder about.
    raise AssertionError("retry loop exited without returning or raising")
