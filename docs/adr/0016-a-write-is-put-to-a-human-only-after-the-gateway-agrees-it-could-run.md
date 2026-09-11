# 0016 — A write is put to a human only after the gateway agrees it could run

**Status:** Accepted (2026-09-11)

## Context

ADR 0004 records the rule the gateway already keeps for rate limits: a human
must never be asked to decide a call that policy will refuse whatever they
say. The transition table keeps the mirror-image rule for approval itself —
`assert_transition` refuses to let an unapproved, mutating call enter
`call_tool` — but nothing kept the *asking* on the same side of the gate as
the checking.

`think()` routed a mutating decision straight to `request_approval` on the
planner's say-so alone. `ToolGateway`'s steps 1–4 — the tool exists, its
arguments validate, the actor holds the scope, there is budget left — ran only
*after* a human answered, when `execute()` was finally called on `call_tool`.
An actor without `project.risk.write` who asked the assistant to record a risk
therefore produced a pause: a card in an approver's queue, asking them to rule
on a call that would come back `denied` the instant they said yes. The
approver's "yes" changed nothing except which of two people learned the call
was refused, and when.

This is fact 9 in the end-to-end code plan and the first of the two gaps the
plan commits to closing before the web layer lands, precisely because a
browser surfaces the queue to a real person — a scripted test never noticed
because nothing was watching the wasted round trip.

## Decision

`ToolGateway` gains `preflight(request) -> ToolOutcome`: steps 1–5 of
`execute()` (find, validate, scope, project, budget → approval), refactored
into one shared `_checks()` both methods call, with the handler never reached
and the budget never counted (counted at execution, per ADR 0004, not at the
check). On a gated tool `preflight` returns exactly what those steps produce —
a refusal, or `approval_required` when every check passes. There is no honest
outcome for an ungated tool or an already-approved call, so those raise
`ValueError` rather than pretend to answer.

`think()` calls `preflight` on the `request_approval` route before advancing
the state. Anything other than `approval_required` ends the turn there, as a
routed `fail` with `tool_failure`, the outcome recorded as an observation, and
a trace event naming what refused it. Only `approval_required` still advances
to `request_approval`, exactly as before.

The transition guard is untouched. `assert_transition` still refuses an
unapproved, mutating call at `call_tool`, and `execute()` still runs every
check again at execution time — an entitlement revoked between the pause and
the decision is caught there, at resume, not trusted to what `preflight` saw
minutes or days earlier.

The preflight request is built with `approval="not_required"`, not
`state.approval`. `state.approval` describes the *previous* tool call this
turn made, if any; a turn that made one approved write earlier in the same
run and then decides on a second, unrelated write must not have that second
call read as pre-approved because the first one was. Nothing has been decided
about *this* call yet, so the honest value is the field's own default — the
same one a fresh `ToolRequest` carries when nobody has said anything.

## Consequences

One extra audit row per write, at pause time: the `approval_required` outcome
from `preflight` writes a gated-call row (`_write_audit_row` already fires for
any `approval_required` status) before the execution's own row lands on
approval. Two rows for one write is not a regression to explain away — it is
the same shape the escalated-read path already produces, and a reviewer
reading `audit_rows` for a trace id now sees the request being checked and
the request being executed as two distinct facts, which they are.

An actor with no scope for a write, or a project mismatch on the call, now
fails the turn immediately with no pause at all — verified against the
manual-test handbook's tomas scenario (A9): no approver is ever shown the
call, and `pauses` gets no row for it. This is a behavior change a reviewer
walking the handbook needs to expect, not a regression: v1 without this ADR
paused first and refused second, and the handbook has been rewritten to match
the corrected order.

`preflight` is now part of `ToolGatewayPort`, so every fake gateway in the
test suite that stands in for the real one needs a `preflight` method to
satisfy `think()`'s new call on the `request_approval` route — a small,
one-time tax on the test doubles, paid in this change.

## Alternatives considered

**Route the write through `call_tool` and let the gateway refuse it there.**
Rejected. The transition guard exists specifically to keep an unapproved
write out of `call_tool`; weakening it to gain a pre-check would trade one
invariant for the very thing it protects.

**Check the scope (and the other gated facts) in the planner, before it
chooses `request_approval`.** Rejected. The decision layer does not import
the tool registry, by design (ADR 0006) — the model chooses a route and a
tool name, and the registry is the only authority on what that name is
allowed to do. Duplicating the checks there would be a second, drifting copy
of the rule the gateway already owns.

**Check in the web layer, before the approval card is even rendered.**
Rejected. The gate would hold only for the one caller that remembered to add
it — `scripts/run_turn.py`, a future second frontend, a replay tool — and not
for the engine itself, which is where CLAUDE.md's approval rule actually
lives.

**Cache the gateway's answer from `preflight` and reuse it at `resume`,
skipping `execute()`'s re-check.** Rejected, and this is the reason
`execute()` still re-runs every check rather than trusting the earlier
`preflight`: an entitlement can be revoked, or a project reassigned, in the
time between a pause and a decision, and the call that finally runs must be
checked against the world as it is at that moment, not the world as it was
when the human was first asked.
