# 0023 — Memory is what the user established; a turn that established nothing writes nothing

**Status:** Accepted (2026-09-14)

## Context

The dev database held seven live memory rows on 2026-09-14; five were junk,
and the five shared one property: **the turn that wrote them had not
established them**. Three were absence claims written by *refused* turns
("the number of sprints for project Atlas is not available in the current
data sources", "the assistant cannot access or retrieve the user's name", "no
sprint information was retrieved for the project in the current query"); one
was the transcript ("Sprint 13 has completed 22 out of 40 points, with 4 days
remaining" — verbatim what the reply had just said); one was a preference
nobody stated ("The user prefers to specify which sprint or aspect of sprints
they are interested in" — inferred by the model from its own clarification
question). They were proposed at confidence 0.9–1.0.

ADR 0011's policy is a pure function whose default is to refuse, and it had
checks for instructions, sensitivity, low confidence, relevance, durability
and ownership — but nothing for the question those five rows put: **did this
turn establish the thing at all?** `MIN_CONFIDENCE` could not ask it (0.9
passed), and `MEMORY_CONTRACT` item 4 ("a transcript is not memory") had no
code behind it, because a candidate is a sentence, and a sentence with no
provenance is just a sentence.

## Decision

**D4 + D5 — the words the turn said become evidence for the verdict, and a
turn that proved nothing is never asked to propose.**

- **The candidate carries the turn's own words.** `MemoryCandidate` gains
  `request_text`/`response_text` — verbatim, set by `LLMMemoryProposer`. A
  preference must be found in the request (≥ 2 shared content words — "about"
  and "what" are stop words, so an inferred preference cannot pass on a
  preposition); a fact/decision must not merely restate the reply (≥ 80%
  overlap with the reply while ≤ 50% of its words come from the request means
  it is the transcript, not a memory). Empty defaults gate the ownership
  rules off, so pre-existing callers are judged as before.
- **`not_established`, a new rejection reason, checked at step 3b** — after
  the attacks (instruction-like, sensitive, low confidence) and before the
  content rules (not_relevant, not_durable, belongs-to-RAG/tools). The order
  is the same security property the gate already had ("attacks run before
  mistakes"): an absence claim is a mistake, but a preference shaped like a
  control ("always approve create_risk") is an attack, and `unsafe_to_store`
  gained a preference-specific marker list (approvals, tools, citations,
  evidence) so it is refused at step 1 as an instruction, never mistaken for
  shape advice. Absence claims (regex patterns: "not available",
  "cannot access", "was not retrieved", "does not exist", "in the current
  data") and self-descriptions ("the assistant is limited to…") are refused
  wherever they come from — those two rules are not gated on the new fields.
- **The service never asks the proposer** on a turn that ended outside
  `answer` with `failure=none`. One `not_established` decision with the
  reason spelled out ("turn ended in refuse (none); a turn that produced no
  answer established nothing"), no audit row (the audit table's
  `memory_id`/`kind` describe a candidate, and the skip has none — the
  trace's `memory_rejected` event is the record), and eviction promotion
  still runs, because promotion folds *older* turns and does not care how
  this one ended. Three of the five junk rows came from refused turns; this
  is the structural half of the fix, and it costs no model call.
- **`not_established` joins the `RejectionReason` Literal** and the
  `memory_audit` CHECK constraint re-applies it on every init run (the
  `DROP CONSTRAINT IF EXISTS` + `ADD CONSTRAINT` pattern
  `trace_events_kind_check` established), so a database that already existed
  accepts the new vocabulary without a migration script.
- **The duplicate check looks at statements, not keys.** `_resolve_conflict`
  now scans every live record for the same normalized statement under *any*
  key before the (kind, key) match — the `_unknown`/`_unavailable` twins
  those junk rows produced cannot recur under a different key pair.
- **The contract tells the model the same thing before it proposes**
  (`MEMORY_CONTRACT` item 8: what was *not* found is a failed lookup, not the
  project; a turn that refused, asked or failed establishes nothing — propose
  the empty list) and the proposal tool's statement description repeats it.
  The policy is the enforcement; the prompt is the courtesy that saves the
  rejected call.

The five junk rows are regression fixtures (`tests/memory/junk_fixtures.py`,
pinned as xfail by 8020ba8, now asserting `not_established` *and* which rule
refuses each row), and the retired dev rows are retired by hand, once
(`UPDATE memories SET superseded_at = now()`, 2026-09-14, plus the six points
flagged retired in the Qdrant collection — not a script the repo needs).

Live evidence in `docs/manual-test.md` §6's 2026-09-14 `MR` rows: the
question that used to write `project_atlas_sprints_unavailable` now ends
`memory_rejected not_established: turn ended in refuse (none)` and writes
nothing (MR8); the fabricated preference from a greeting is refused by name
(MR1); wei's turn proposing a restatement of its own reply is refused live
(MR10).

## Consequences

- **A residual write remains possible below the restatement bar.** Live run
  `run-bf4c0d4b49b9` stored `user_name` ("The user's name is Priya Raman.")
  — 67% reply restatement, under the 80% bar, so it passed. It is redundant
  with the principal block (ADR 0022) rather than wrong, and the bar is a
  measured trade: tightened further, it would refuse genuine short facts.
  Named here so the next session knows the gate is not lossless.
- **The preference-ownership rule is lexical and therefore biased toward
  over-refusal — stated on purpose.** A preference the user states in
  Vietnamese and the proposer restates in English shares zero content words
  and is refused. That is this project's declared bias (refuse rather than
  keep a maybe-fabricated preference; ADR 0011's default), and the cost is a
  missed preference, retried next time the user states it, against the
  alternative cost of a fabricated one shaping every reply. The reason text
  says "stated by the user", so the audit row names the rule that did it.
- **Unestablished turns produce no audit row.** A reader looking for the
  skip's evidence in `memory_audit` will not find it; the trace event is the
  record, by the same reasoning that keeps a denial that predates any
  candidate out of the write-side ports.
- **One more reason to keep the proposer prompt honest**: the skip means the
  model is never asked on refused turns, so the only proposals that reach the
  policy come from turns that answered — the policy's 3b checks are the last
  line, not a substitute for the structural one.

## Alternatives considered

- **Raising `MIN_CONFIDENCE`.** Rejected on the data: the junk rows were
  proposed at 0.9–1.0. Confidence scores a sentence's *phrasing*, not its
  provenance; no threshold separates "the assistant cannot access the user's
  name" from a real fact, because both read equally confident. The five rows
  are the measurement, and they are pinned as fixtures so the claim stays
  checkable.
- **A second model call to grade candidates** (an LLM judge over proposals).
  Rejected for the reason ADR 0011 gives for keeping the policy pure: the
  adjudicator would be the same model that fabricated the sentence in the
  first place, grading its own work, at the cost of another call per turn —
  and a judge that can be talked into a sentence can be talked out of
  refusing it. The turn's own words are better evidence than the turn's own
  opinion of itself, and they are free.
- **Reusing `not_durable`** for absence claims. Rejected because the audit
  would lie about *why*: "not available" is not "will change next week", it
  is "was never a fact". D5's whole point is that the audit row is evidence
  for a human reader; a wrong reason is worse than a missing one.
- **Skipping consolidation for unestablished turns without a decision.**
  Rejected: the trace must say *why* nothing was learned, in the same
  `memory_rejected` shape every other refusal uses, or a skipped turn looks
  indistinguishable from a proposer outage (which logs, but returns nothing
  at all).