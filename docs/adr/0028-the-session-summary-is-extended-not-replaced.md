# 0028 — The session summary is extended, not replaced

**Status:** Accepted (2026-09-15)

## Context

ADR 0014's promotion folds each batch of turns evicted from the short-term
window into the session's one `session_summary` record, superseding the
previous. The dev database (actor `priya`, project `atlas`, session
`sess-f38d543c27b3`, 2026-09-14/15) showed what that actually produced. The
session held 12 turns; the window (`HISTORY_TURN_LIMIT = 6`) evicted one turn
on each of six arrivals, so six promotions ran, each writing one summary that
superseded the last. Read in order:

| run that promoted | evicted turn's request | resulting summary statement |
|---|---|---|
| run-147f1e5b… | "give me project status" | `Goal: give me project status.` |
| run-b207276f… | "milestone" | `Goal: milestone. Open: milestone.` |
| run-9ee1c3f9… | "how many do you have >?" | `Goal: how many do you have >?.` |
| run-b6dc7387… | "how many milestones project having ?" | `Goal: how many milestones project having ?. Open: How many milestones does the project have?.` |
| run-2fc5fc1f… | "currrent status of milestone 1" | `Goal: currrent status of milestone 1.` |
| run-b0408b5d… | (the "i mean M1" turn) | `Goal: current status of milestone 1.` |

The user's own words for it: *"it seems just get the latest sentence"*. Every
statement is the structural projection of the single evicted turn — the goal
is that turn's raw request, typos included — and nothing accumulates.

Two rows prove the pipeline itself was healthy, which is what narrowed the
diagnosis. `model_calls` shows `chose propose_session_summary` in every
promoting run, so the proposer was not down. And the fourth statement's
`Open:` item — *"How many milestones does the project have?"* — is
grammar-corrected, while the evicted turn's raw request is *"how many
milestones project having ?"* and its route was `refuse` (structural
unresolved questions come only from `clarify` turns, verbatim). That item can
only have arrived through a validated model proposal. The model did its part
and left `user_goal` null — the honest answer for a batch that did not change
the session's goal. The code threw the previous summary away anyway.

The root cause was three layers, all confirmed in code:

1. **`conversation_state` never saw the previous summary.**
   `memory/service.py::_promote` fetched the live `session_summary` and did
   two things with it — handed it to the proposer as prompt context, and
   listed its id in `supersedes` — but never gave it to
   `memory/promotion.py::conversation_state`, the function that decides the
   new summary's content. Carrying content forward was entirely the model's
   job, under a contract that told it to do the opposite.
2. **The structural goal overwrote the previous goal whenever the proposal
   was null.** `structural_state` unconditionally sets
   `user_goal = oldest.request`; `conversation_state` overlaid the proposal's
   goal only `if proposal.user_goal`. A null goal — what the contract
   *instructs* the model to return when the batch did not change the goal —
   left the newest evicted request standing as the session's goal. This
   mechanism produced all six rows above.
3. **The contract and the code contradicted each other.**
   `PROMOTION_CONTRACT` said *"leave it unset if these turns did not change
   it"* — coherent only if unset means *keep the current one* — and the code
   did not keep it. Its docstring promised the model was shown the previous
   summary "so it can extend or correct it rather than starting over"; the
   extend half had never been implemented.

Everything downstream was correct: the store superseded idempotently, the
audit recorded write/update/forget faithfully, the `history_promoted` event
landed in the right trace. The defect was entirely in what
`conversation_state` was given.

## Decision

**Code carries the previous summary forward; the model proposes only the
delta.** The merge belongs in `conversation_state`, not in the model — this is
the project's own split (ADR 0011: the model proposes, code decides) — and the
dev DB shows why relying on the model is not merely suboptimal but
*incoherent*: the contract asks for a delta, so a cooperative model that
follows it guarantees content loss. Concretely:

- **`memory/summary.py::parse_summary`** recovers the previous summary's
  structured content by parsing its statement — the inverse of `_render`,
  built off `SUMMARY_SECTIONS`' own labels so parser and renderer cannot
  drift. This is the inverse of a renderer this repo owns, not a scrape of
  foreign text. The structured content of a summary exists nowhere else: the
  `memories` table has no per-section columns and `MemoryRecord` carries no
  payload besides the statement. Three deviations are accepted and pinned by
  tests: an item containing `"; "` parses as two (every word survives); a
  label inside an item splits early (the fragments accumulate); a statement
  clipped at `STATEMENT_MAX_CHARS` drops the marker — and, when `_render`
  glued `...` onto text cut mid-word, the possibly-incomplete final item as
  well, never carrying a half-fact that "reads as complete" (per `_render`'s
  own docstring). Foreign text (no known label) parses to `{}`: a summary this
  repo did not render is not carried forward on a guess.
- **Precedence, per field.** `user_goal`: the proposal's, else the previous
  summary's, else the structural guess (the oldest evicted request —
  unchanged, and still right for a session's *first* promotion, ADR 0014's
  fallback case). Each list field merges `[*structural (this batch),
  *proposed, *previous]`, whitespace-collapsed, deduplicated preserving first
  occurrence, capped at `MAX_ITEMS_PER_SECTION`. Newest content enters at the
  front, so when a section saturates at 3 the **oldest** item leaves: a
  sliding window, the honest semantics for a record capped at 400 characters
  — and `SUMMARY_SECTIONS`' priority order decides what falls off first.
  `pending_approvals` is never proposed and never carried: it is re-derived
  from the turns each promotion, because a superseded summary must not
  resurrect a settled approval.
- **The trace and audit say what happened.** The `MemoryDecision.reason` for
  an update that extended a previous summary names the run that wrote it —
  `session summary folded N evicted turn(s) over the summary from run <id>` —
  so a reviewer of the evidence can tell a first write from an extension
  without reading code (the same move as ADR 0024's decision reasons).
- **The contract stops asking for the impossible.** `PROMOTION_CONTRACT` keeps
  its `user_goal` sentence (now true) and gains one sentence: the current
  summary is carried forward automatically, propose only what these turns add
  or change, restate nothing. `build_promotion_messages`' docstring now says
  the previous summary is shown *for context*; the "extend or correct"
  promise moved from the model to the code, which is the whole fix.

## Alternatives rejected

- **Let the model restate the previous summary.** That was the defective
  behaviour; the contract forbids it; the 3-item caps would truncate
  restatements anyway; and it makes correctness depend on the one component
  the architecture deliberately distrusts.
- **Store the structured mapping** (a JSON column or a second record kind).
  Widens the store port, the schema, and every fake standing in for the port
  in engine tests, to avoid parsing a string this repo renders in the first
  place. Revisit only if parsing proves too brittle in practice, with
  evidence.
- **Re-derive the whole summary from `session_turns` each promotion.** The
  turns carry requests verbatim but not what the model judged durable —
  `accepted_facts` and `decisions` from earlier promotions exist only in the
  previous statement — and it drifts toward ADR 0014's rejected "summarise
  the whole session every turn".
- **Raise the caps so nothing falls off.** `STATEMENT_MAX_CHARS` is a
  load-bearing bound: memory competes with evidence for prompt space
  (`MEMORY_BUDGET_TOKENS = 400`). The sliding window inside it is the
  feature, not the bug.

## Consequences

- A summary can no longer regress to the newest evicted request, whatever the
  proposer does: a null proposal, a failed proposer, or no proposer at all
  all leave the previous content standing.
- A summary is capped — 3 items per section, 400 characters — so old content
  genuinely leaves rather than accumulating forever. What leaves first is
  decided by `SUMMARY_SECTIONS`' priority order (`Established` first, `Goal`
  last) and, within a section, by age.
- The three parse deviations are accepted, bounded, and pinned by tests. They
  cost nothing semantically: the fragments still render as facts, and a
  clipped statement loses at most its final item by choice.
- The dev database's damaged rows are not repaired, deliberately: every one
  is a `session_summary`, which is session-bounded
  (`SESSION_BOUNDED_KINDS`), so they can never be recalled outside their
  session, and that session is over. The superseded rows stay as evidence of
  the defect.

ADR 0014 stands — its structural fallback and its "fold only what the limit
evicts" are untouched; this ADR adds only the carry-forward. ADR 0011's
"the model proposes, code decides" gains its third half: what code also
carries.