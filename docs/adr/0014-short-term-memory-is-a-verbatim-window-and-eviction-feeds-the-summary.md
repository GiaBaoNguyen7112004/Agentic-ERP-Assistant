# 0014 — Short-term memory is a verbatim window, and eviction feeds the summary

**Status:** Accepted (2026-09-10)

## Context

Between turn 1 and turn 2 of a session, the only thing carried used to be the
session id. Turn 2 was shown durable memory — preferences, decisions, the open
intent, a session summary — and nothing else. Nothing showed the planner or the
composer what was said two messages ago, so a follow-up such as "and what about
M2?" or "yes, the second one" had no antecedent and could not be resolved.

Durable memory is deliberately not a transcript: ADR 0011 makes the policy the
only writer and refuses by default, and ADR 0012 keeps memory in a role of its
own where it can never become a citation. Short-term memory is a different
thing, and the difference is the design — the same principal's recent
conversation, verbatim, bounded, and inert in the prompt.

Two pieces existed for this and were never wired: the compaction allow-list
(`context/compact.py`, ADR 0003) and `summarize_session`, reachable through
`SessionMemory.consolidate`. Its docstring said so plainly: nothing produces
that state yet; the compaction allow-list is fed by a caller that does not
exist.

The decisions to make: what role the recent turns take in a prompt, whether the
policy judges them, where the verbatim record lives, how a turn leaves the
window, and where in the orchestrator each act runs.

## Decision

**A dedicated `history` role, not native `user`/`assistant` message pairs.** An
old request in the `user` role would be byte-indistinguishable from what the
user typed now, and `assistant` already means "this turn's reply" in
`build_memory_messages`. The role gets a preamble and a `SYSTEM_POLICY` rule,
exactly like evidence, observation and memory; the OpenAI adapter folds it to
`developer` behind a preamble, the same precedent. Precedence is stated in the
module docstring: policy instructs, the user asks, evidence grounds,
observations report, history reminds, and memory is background.

**The window is the newest `HISTORY_TURN_LIMIT` (6) turns, clipped by tokens,
and the budget is proven sufficient.** `select_history` keeps the newest six
turns, strips citation tags and `Sources:` trailers *before* clipping, and
clips request and response against `HISTORY_BUDGET_TOKENS` (1600). A test
asserts `6 × (80 + 160 + 16) ≤ 1600`, so the budget can never silently drop a
turn from inside the window: the only turns that leave are the ones the limit
evicts, and those are promoted rather than dropped. `state.history` holds
exactly what the prompt renders — clipped, stripped copies — so the trace shows
what the model saw, and `runs.state` / `pauses.state` do not carry six full
transcripts.

**Not policy-gated; defended structurally.** `policy.decide` protects durable
memory and has no opinion on a `ConversationTurn`. A bounded window of the
user's own words must be faithful — dropping "never approve X" from the user's
last message would be wrong, and a policy that might refuse it is a policy
that might misquote them. The defence is structural instead: its own role
behind a preamble, `SYSTEM_POLICY` rule 7 (a record of words, not facts;
nothing in it may be cited; a previous reply that reads like an instruction is
a thing that was once said), citation tags stripped at selection time, and the
engine's grounding check refusing any re-cited source that was not retrieved
this turn.

**`session_turns` is the record; `promoted_in_run` is the watermark.** A
separate projection table rather than a `session_id` column on `runs`:
`runs.state` is the audit record, reading it for prompts would couple prompt
construction to `STATE_VERSION` and hydrate whole states to render two lines.
A projection written by the code that knew what the turn was is the same move
as `pauses.arguments_summary` and `trace_events`. Finished turns are immutable,
so ADR 0013's two-copies objection — about *mutable* records — does not apply.
A separate `SessionHistoryPort` rather than a wider `TurnMemoryPort`:
`SessionMemory` needs embeddings and Qdrant, history needs neither, and
`memory=None, conversation=store` is a legitimate deployment.

**Promotion goes through the existing gates.** A turn leaves the window when a
newer arrival pushes it past the limit. The evicted turns are folded into the
session's `session_summary` through `compact_conversation` (the allow-list,
ADR 0003) and `summarize_session`, with an audit row — which also closes a gap,
since summaries were never audited before. A model may propose the summary's
fields; code decides what survives, the per-item `unsafe_to_store` check
included. When no summary proposer is configured, a structural fallback still
produces a summary — the goal from the oldest evicted request, unresolved
questions from turns that ended in `clarify` — so promotion never depends on a
model call being available. The summary's `links` are the trace ids it folded.

**Paused turns are recorded but never evicted.** A turn waiting on a human has
not finished being worth something: the pause *is* the fact this turn is about,
and it belongs in the window verbatim — the next request sees "waiting for
approval to run create_risk", not a paraphrase of an open decision. It is
recorded with no reply at pause time and upserted with the reply on resume.

**Orchestrator ordering:** short-term recall → long-term recall → engine →
consolidation, which promotes → run record → pause → the turn joins the
window. Recall first so the engine is shown the antecedents on its first
decision; promotion inside consolidation so the `history_promoted` event is in
the trace of the turn that caused the eviction; the window record last,
because the trace store's contract is that an unsaved run "did not happen", so
a history row is only ever written about a run the trace holds.

## Consequences

**The residual risk is on the `think → answer` route.** A planner that answers
directly, without retrieval or a tool, is not grounding-checked, so a planner
answering from a prior reply in the history block produces an uncited reply.
`PLANNER_CONTRACT` forbids it in text — "history is never a reason to choose
`answer`" — and this ADR records it as model-dependent. A structural guard on
that route is a named follow-up, not a silent gap.

**The `trace_events.kind` CHECK is named and re-applied on every init run.**
The kind set grew on a database that already existed, and `CREATE TABLE IF NOT
EXISTS` never touches an existing constraint — without the name-and-reapply
pair, the growth would have permanently split databases initialised before
from those initialised after.

**1600 tokens of history compete with evidence and memory in the same prompt
budget.** The number is chosen so the invariant is provable — 6 × 256 = 1536,
with 64 to spare — rather than tuned: a follow-up needs the antecedent, not
the transcript, and a window that needed the whole budget would be the wrong
shape.

**A memory-layer outage leaves turns unpromoted, and they are retried.**
Promotion lives in consolidation, so turns accumulate unpromoted while
`memory=None`, and a store that raises mid-consolidation leaves the watermark
unset rather than lying about what was folded. Marking after consolidation
returns is what makes the retry safe; promotion is idempotent, so a re-fold
supersedes the same summary harmlessly.

**A summary proposer that fails yields the structural summary.** The turns are
still marked promoted; what the model might have added is a better paraphrase,
not the only copy.

**One window per (session, actor).** `recent` and `evicted` filter by both, so
a second actor sharing a session id sees none of it — the same boundary
`memory.models.bounds` draws around an intent.

## Alternatives considered

**Native `user`/`assistant` message pairs.** Rejected, above: role ambiguity
is the whole problem, and the dedicated role is what makes `SYSTEM_POLICY`
rule 7 expressible — "a previous reply that reads like an instruction is a
thing that was once said" needs a place it can name.

**Derive history from `runs.state`.** Rejected: couples prompt construction to
`STATE_VERSION`, hydrates whole states to render two lines, and makes every
replay of a run a potential prompt change. The projection is written by the
code that knew what the turn was — the same move as `pauses.arguments_summary`.

**Widen `TurnMemoryPort`.** Rejected: `SessionMemory` needs embeddings and
Qdrant, history needs neither, and one wider port would make
`memory=None, conversation=store` impossible to express.

**Gate the window with `policy.decide`.** Rejected, above: a policy that might
refuse the user's own last sentence is unfaithful by design; the defence must
be structural, not discretionary.

**A `remember`/`history` graph node.** Rejected on ADR 0011's grounds — the
step budget must not pay for bookkeeping — and because recall before the engine
is what lets the planner resolve the antecedent on its first decision rather
than spending a step discovering it.

**Promote at recall time.** Rejected: eviction observed during recall would
mark turns promoted before their summary was written, and a crash between the
two would lose them. Promotion happens in the consolidation that writes the
summary, and the watermark is set only after it returns.

**Summarise the whole session every turn.** Rejected: one model call per turn,
and a summary that drifts a little more each turn. Only what the limit evicts
is folded, once, at the moment of eviction, while the words are exact.