"""How many calls one actor has made to one tool, and how long until the next.

A protocol and one in-memory implementation, the same pair -- for the same
reason -- as :mod:`agentic_erp_assistant.tools.audit`: the gateway has to be
able to count a call without knowing whether the counter is a dict in this
process or a shared store behind four of them. A dict is not the answer for a
real deployment (two workers would grant two budgets), and the seam is what
lets that change without the execution path changing with it.

The policy itself is not here. It is declared on
:class:`~agentic_erp_assistant.tools.registry.ToolDefinition` with every other
per-tool budget, and passed in on each call. A limiter that stored its own
limits would be a second control plane, and the one a reviewer does not read.

A sliding log, not a fixed window
---------------------------------

Each key keeps the timestamps of its recent calls and drops the ones that have
aged out. A fixed window -- a counter reset every ``per_seconds`` -- is cheaper
and wrong in two ways that matter here. It allows a double burst across the
boundary: the full budget at the end of one window and the full budget at the
start of the next, back to back, which is exactly the runaway-loop shape the
limit exists to stop. And it cannot answer "when may I try again" honestly; it
knows when the window resets, not when capacity returns, so the
``retry_after_seconds`` it reports is a guess the caller then acts on. The log
answers exactly: capacity returns when the oldest call in the window ages out.

The cost is bounded and small -- at most ``max_calls`` timestamps per key,
pruned on every touch and trimmed on every write. The trim is what makes that
bound structural rather than a property of callers behaving: this limiter
cannot be talked into holding a window's worth of timestamps by a caller that
records without checking first.

Determinism
-----------

Nothing here reads a clock. ``now`` is supplied by the caller, which is the
gateway's injected clock, so "the budget was spent and the wait is 47 seconds"
is a thing a test asserts rather than races. There is no jitter and no
randomness: two identical call sequences against the same clock produce the
same refusals.
"""

from collections import deque
from datetime import datetime, timedelta
from threading import Lock
from typing import Protocol, runtime_checkable

from agentic_erp_assistant.tools.registry import RateLimitPolicy

__all__ = ["InMemoryRateLimiter", "RateLimiter"]


@runtime_checkable
class RateLimiter(Protocol):
    """Somewhere the calls one actor has made can be counted and asked about.

    Two methods rather than one ``consume`` that both checks and increments,
    because the gateway does the two at different points on purpose: it checks
    before asking a human to approve anything, and it counts only once the call
    is actually about to run. A combined method would force the count to happen
    at whichever of those two moments the signature picked.
    """

    def check(
        self,
        actor: str,
        tool_name: str,
        policy: RateLimitPolicy,
        now: datetime,
    ) -> float | None:
        """Seconds the actor must wait, or ``None`` if the call may proceed.

        Must not raise and must not count the call. A limiter whose check has
        a side effect makes "asked and refused" indistinguishable from "ran".
        """
        ...

    def record(
        self,
        actor: str,
        tool_name: str,
        policy: RateLimitPolicy,
        now: datetime,
    ) -> None:
        """Count one call against the actor's budget for this tool."""
        ...


class InMemoryRateLimiter:
    """A dict of timestamp logs, and the default. Enough for one process.

    Locked, unlike :class:`~agentic_erp_assistant.tools.audit.InMemoryAuditLog`,
    because this is the one piece of shared mutable state on the execution path
    and the gateway already runs handlers on a thread pool. An audit log that
    interleaves two appends loses nothing; a counter that interleaves a prune
    and an append hands out a budget nobody declared.
    """

    def __init__(self) -> None:
        self._calls: dict[tuple[str, str], deque[datetime]] = {}
        self._lock = Lock()

    def check(
        self,
        actor: str,
        tool_name: str,
        policy: RateLimitPolicy,
        now: datetime,
    ) -> float | None:
        with self._lock:
            calls = self._live(actor, tool_name, policy, now)
            if len(calls) < policy.max_calls:
                return None
            # Capacity returns when the oldest call still in the window ages
            # out of it -- not when some window resets. Strictly positive:
            # anything at or before the cutoff was just pruned.
            oldest = calls[0]
            return (oldest + timedelta(seconds=policy.per_seconds) - now).total_seconds()

    def record(
        self,
        actor: str,
        tool_name: str,
        policy: RateLimitPolicy,
        now: datetime,
    ) -> None:
        with self._lock:
            calls = self._live(actor, tool_name, policy, now)
            calls.append(now)
            # Only the most recent ``max_calls`` entries can ever hold a budget
            # of that size, so anything older is answering no question. Trimmed
            # here rather than left to the window, so the log is bounded by the
            # policy and not by how fast a caller can call.
            #
            # It only ever fires for a caller that records without checking --
            # which the gateway does not do, because :meth:`check` runs first
            # and refuses. For such a caller this is marginally lenient: the
            # over-budget call it never asked about stops counting sooner than
            # a strict reading would have it. Discipline about that belongs at
            # the gateway, where the refusal can actually be returned.
            while len(calls) > policy.max_calls:
                calls.popleft()

    def _live(
        self,
        actor: str,
        tool_name: str,
        policy: RateLimitPolicy,
        now: datetime,
    ) -> deque[datetime]:
        """The calls still inside the window, with the aged-out ones dropped.

        Pruned on read rather than on a timer: there is no background task in
        this runtime, and a log that only shrinks when someone looks at it is
        both correct and the cheapest thing that can be.
        """
        calls = self._calls.setdefault((actor, tool_name), deque())
        cutoff = now - timedelta(seconds=policy.per_seconds)
        while calls and calls[0] <= cutoff:
            calls.popleft()
        return calls
