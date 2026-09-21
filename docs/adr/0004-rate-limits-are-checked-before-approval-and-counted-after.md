# 0004 — Rate limits are registry policy, checked before approval and counted after

**Status:** Accepted (2026-09-05)

## Context

The gateway was already one ordered path — find, validate, permit, approve,
audit, execute, trace — with every rule read off a `ToolDefinition` so that no
policy exists in a code path and in no document. A per-actor, per-tool call
budget is a fifth rule of that kind, and adding it raises three questions the
diff cannot answer on its own: what shape the counter has, where in the order it
sits, and what one call costs.

The forcing case is a stuck agent. A model in a loop calling `list_risks` forty
times, or a client replaying an approved `create_risk`, produces load that every
existing check waves through: the scope is held, the arguments are valid, and
the approval — for the write — was genuinely given. Nothing in the pipeline was
counting.

## Decision

`RateLimitPolicy(max_calls, per_seconds)` is declared on `ToolDefinition` beside
`RetryPolicy`. The counter lives behind a `RateLimiter` protocol in
`tools/limits.py`, defaulting to `InMemoryRateLimiter` — the same
port-plus-in-memory-default pair as `AuditSink`, for the same reason: a dict is
right for one worker and wrong for four, and the gateway should not know which
it is in. A refused call comes back as the new `ToolStatus` member
`rate_limited`, carrying `retry_after_seconds`, and leaves both a `rate_limited`
trace event and an audit row.

Three sub-decisions do the actual work.

**There is no spelling for "unlimited".** `rate_limit` has no `None` and no
sentinel; it defaults to `DEFAULT_RATE_LIMIT` (30/60s), and `create_risk`
declares the tighter `WRITE_RATE_LIMIT` (5/60s). A tool added a year from now
inherits a real budget whether or not its author thought about one. This is the
argument `required_scope` already makes by rejecting a blank string, applied to
a field where the tempting default is "no limit".

The write's budget is tight but is *not* the primary gate — approval is. It is
the backstop for what approval cannot see: an approve-then-resubmit loop
replaying one decision. Five a minute is more writes than a person approves in a
minute, so it only ever fires on something that is not a person.

**The window is a sliding log, not a fixed window.** Each `(actor, tool)` key
holds the timestamps of its recent calls, pruned on every touch and trimmed to
`max_calls` on every write. A fixed counter reset every `per_seconds` is cheaper
and wrong twice over. It permits a double burst across the boundary — the full
budget at the end of one window and the full budget at the start of the next,
back to back, which is precisely the runaway shape the limit exists to stop. And
it cannot answer "when may I try again" honestly: it knows when its window
resets, which is not the moment capacity returns, so the `retry_after_seconds`
it reports is a guess the caller then acts on. The log answers exactly —
capacity returns when the oldest call in the window ages out —  and
`tests/tools/test_limits.py` asserts that waiting precisely as long as the
refusal asked for is enough.

**The check sits above the approval gate; the count sits below it.** These are
the same policy at two moments, and splitting them is the decision.

The check runs after permission and before approval, so a human is never asked
to decide a call that policy will refuse whatever they say. Spending an
approver's attention on a foregone refusal is the expensive mistake, and an
approver who is asked to rule on calls that then fail anyway stops reading the
ones that matter.

The count runs at the moment the handler is about to execute. A call that merely
stops for a human therefore costs nothing, which is what makes a budget of one
usable at all — charged at the check, the approve-then-resubmit round trip would
consume two units and a limit of one could never be exercised. For the same
reason, a call refused for a missing scope or invented arguments costs nothing:
quota is consumption of a backend, and a denial consumed no backend. If denials
counted, a model firing calls it is not entitled to could lock its own actor out
of the tools it *is*.

**One call is one unit, retries included.** `RetryPolicy` already bounds
attempts. Charging per attempt would make an actor's real budget a function of
how flaky the backend was that minute — which is the one property a limit
described as deterministic must not have.

## Consequences

`rate_limited` is a distinct `ToolStatus` rather than a flavour of `denied`,
because the two differ on permanence: a missing scope will still be missing next
time, a spent budget refills. Collapsed into one status the graph would either
abandon a call that would succeed in forty seconds, or re-offer one that will be
refused forever. `retry_after_seconds` — declared in an earlier step against
exactly this use — is now required on `rate_limited` and optional on
`transient_failure`, since a limiter always knows the answer and a timeout may
not.

The gateway does **not** retry a rate-limited call. The wait is reported upward
for the caller to schedule. Retrying inside the turn would spend the attempt
budget proving a limit that is, by construction, still in force.

`_write_audit_row` now records a rate-limited call whatever the tool, breaking
the previous rule that only approval-gated tools produce rows. The rule existed
so writes would not be buried under one row per read, and that still holds for
successful reads. But a limit that fires silently on reads hides exactly the
runaway pattern it was added to catch, and that pattern deserves a row wherever
it appears.

Nothing is enforced across processes. `InMemoryRateLimiter` means two workers
grant two budgets, and the docstring says so. The port is the mitigation, not
the fix; a shared-store implementation is a drop-in and needs no change on the
execution path.

The limiter reads no clock of its own — `now` is supplied by the gateway's
injected clock — so "the budget was spent and the wait is 47 seconds" is
asserted rather than raced. There is no jitter anywhere in it.

## Alternatives considered

**Count in `gateway.py` directly, as the step description literally asked.**
Rejected in placement only, not in substance: the *check* is in `gateway.py`, in
the ordered pipeline where it belongs. Only the storage moved behind a port —
one concern per module, and the same seam the audit sink already has. A dict
field on the gateway would have made the single-process assumption permanent.

**A fixed window.** Rejected: burst at the boundary, and a dishonest
`retry_after_seconds`. See above.

**Token bucket.** Rejected as more machinery than the case needs. It smooths
bursts better, but its state (a fractional token count plus a last-refill
timestamp) is harder to explain to a reviewer and harder to assert on than a
list of times, and nothing here needs sub-window smoothing.

**Check and count in one `consume()` call.** Rejected — it forces the count to
happen at whichever of the two moments the signature picks. Counting at the
check charges calls that stop for a human; counting at execution means the check
has to happen there too, below the approval gate, and approvers get asked about
calls that cannot run.

**`rate_limit: RateLimitPolicy | None = None`.** Rejected: an optional limit is
an unlimited default wearing a type annotation, and the tool that most needs a
budget is the one somebody adds in a hurry.

**Reuse `denied`, or `transient_failure`, rather than a new status.** Rejected.
`denied` loses the wait and makes a recoverable refusal look permanent.
`transient_failure` is worse: it is the one status the retry engine acts on, so
a rate-limited call would immediately be retried into the limit that just
refused it.

**Count per attempt rather than per call.** Rejected: non-deterministic by
construction. See above.
