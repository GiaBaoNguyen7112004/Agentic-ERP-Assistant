# 0003 — Compaction is an allow-list, and the summary is best-effort

**Status:** Accepted (2026-09-04)

## Context

A conversation cannot grow forever, so at some point it is compressed. The
natural implementation is to hand the transcript to a model and keep what comes
back.

That implementation quietly defeats hard constraint #4 in CLAUDE.md — no write
executes without an explicit, recorded approval. A summarizer optimizes for
readable prose, and "there is an unapproved request to delete record rec-9
waiting on a human" reads like housekeeping. Once it is compressed away, the next
turn does not know an approval is pending. Nothing raised, nothing logged, and
the control is gone. The same is true of a guardrail finding in `safety_flags`.

The compression step must therefore not be the thing deciding what is important.

## Decision

`compact_conversation` is an allow-list over six named fields — `user_goal`,
`accepted_facts`, `citations`, `pending_approvals`, `safety_flags`,
`unresolved_questions` — declared in `PRESERVED_FIELDS`. Every one of them
present in the input survives into the output. Everything else is dropped.

Three sub-decisions follow, and each one is the part that actually does the work.

**The list is an allow-list, not a deny-list.** A deny-list fails open: a field
added upstream next month passes through a rule that was never told about it. An
allow-list fails closed — the new field is dropped, and keeping it requires a
diff someone reviews. Since the fields at stake are safety state, failing closed
is the only acceptable direction.

**The six values are preserved, not validated.** This module does not check that
`citations` holds well-formed `Citation` objects, or that `pending_approvals`
matches `ApprovalRequest`, though both types exist one package over. Validation
is a way for compaction to *fail*, and every way it can fail is a way an approval
can vanish. A malformed approval that raises here is an approval that did not
survive. Whatever shape an upstream layer created comes out the other side;
policing that shape is that layer's job, and it has already had its chance.

The one normalization is that a top-level `list` becomes a `tuple`, so the record
is genuinely immutable rather than a frozen dataclass wrapping a mutable list.
Elements are untouched. Values that are not lists pass through unconverted —
`tuple()` over a string would explode it into characters.

**The summary is best-effort; the six fields are not.** The summarizer is
injected behind a `Summarizer` protocol and defaults to `structural_summary`,
which is deterministic and needs no provider. A summarizer that raises — or
returns a non-string — is logged and replaced. A provider outage cannot cost a
preserved field. This is the one place in the runtime where a broad
`except Exception` is the safer behaviour, because the alternative loses state
that cannot be recovered.

## Consequences

The guarantee is structural rather than behavioural: it holds because of the
shape of the code, not because the summarizer is usually good at its job. That is
what makes it defensible.

`compact_conversation` accepts `Mapping[str, object]` rather than one of this
project's `extra="forbid"` models, which reads as a step backwards from the
repository's "state is typed and explicit" rule. It is not. A strict model would
*reject* an input containing `raw_transcript`, and rejecting is not dropping —
the allow-list requires accepting unknown fields in order to discard them.
Compaction is precisely the boundary where loose accumulated state becomes typed,
and the output is a frozen `CompactedConversation`. The preserved fields are
typed `object | None` because this module preserves rather than validates; that
is the same decision as above, visible in the annotation.

`as_state()` exists so the acceptance criterion can be asserted honestly. Asking
whether a dataclass has a `raw_transcript` attribute proves nothing — it never
could have. Asking whether the emitted mapping contains the key is a real test.

**An open gap, recorded rather than hidden.** Compaction guarantees
`pending_approvals` and `safety_flags` survive; `ContextBuilder` (ADR 0001) may
then exclude them for budget, and the guarantee is void. Today the mitigation is
priority: those fields take the top band when they become candidates. That is a
convention, not an enforcement. The proper fix is a "must include" concept in the
builder — a candidate whose exclusion is an error rather than a plan — and it is
deliberately not in this step.

A summary derived from a transcript is generated text that may carry instructions
someone placed in a project document. When it enters a prompt it goes in as
history or evidence, never as policy — the same boundary `llm/prompts.py` draws
for retrieved passages.

## Alternatives considered

**Summarize everything and keep the summary.** Rejected — the failure this ADR
exists to prevent.

**A deny-list ("drop the transcript, keep the rest").** Rejected: fails open on
every field nobody has thought of yet. `tests/context/test_compact.py` asserts
this as a property over invented field names, so a future refactor toward a
deny-list fails a test rather than passing quietly.

**Validate into `Citation` and `ApprovalRequest` on the way through.** Rejected.
It is the most tempting option, because the types are right there and the project
values typed state everywhere else. But it converts a malformed approval into a
lost approval, which is the exact outcome the module is built to prevent. The
spec's own example, `{"tool": "delete_record", "id": "rec-9"}`, does not match
`ApprovalRequest` — it has neither `reason` nor `arguments_summary` — so a
validating implementation would fail the acceptance criterion on the first
example it was given.

**Mandatory LLM summarization.** Rejected: it couples state preservation to
provider availability, so an outage becomes data loss.

**Putting `CompactedConversation` in `state/`.** Deferred. It is plausibly the
first citizen of the state model CLAUDE.md plans, but creating that package today
for one type with one caller is premature. It migrates when the runtime's state
model lands.
