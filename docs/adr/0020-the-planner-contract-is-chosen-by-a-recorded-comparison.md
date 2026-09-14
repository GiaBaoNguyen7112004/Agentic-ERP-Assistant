# 0020 — The planner contract is chosen by a recorded comparison

**Status:** Accepted (2026-09-12)

## Context

The manual walkthrough (`docs/manual-test.md` §6, commit `638adbc`) recorded
R1 — "Why is milestone M2 late and by how much?" — taking the
retrieve-and-cite route once in four tries across a live session; the other
three times the planner called `get_project_status` alone, answered "two
days late," and never addressed the "why." No citation was ever asserted
without a retrieved passage behind it (constraint 3 held throughout), so this
was a routing inconsistency, not a grounding defect — but an incomplete
answer to half a compound question is still a real gap.

`docs/gap-plan.md`'s Phase L proposed measuring before tuning: three
candidate developer-block contracts, six cases fixed from the handbook, run
`N` times each through the same client, so a prompt edit's effect is read
off a number rather than a feeling. Phase J (`llm/prompts.py::
build_planner_messages`, ADR 0019) and Phase L1 (`LLMGateway.planner_contract`)
built the wiring; this ADR is the run and the decision.

## Decision

**The selection rule was written before the run, in
`eval/routing_prompts.py`'s module docstring, committed in `48c6987` —
before `evidence/routing/routing-comparison-2026-09-12.json` existed:**

> Highest overall match rate across all six cases wins. A tie goes to the
> fewer hallucinated tool calls, then to the shorter contract. `V1_DIRECT`
> keeps its place (no promotion) unless a candidate beats it on the R1 case
> specifically *and* loses on no other case.

**Three contracts, six cases, five repeats, ninety real gpt-4o calls, paced
4 seconds apart (a live dry run at no pause hit a 30k-tokens-per-minute
account limit partway through eighteen calls; paced, the full run completed
with zero rate-limit errors and zero rows lost):**

| Prompt | Match rate | Hallucinations | Cost | Contract length |
|---|---|---|---|---|
| `v1-direct` (production, unedited) | 25/30 = **83.3%** | 0 | $0.1854 | 2,087 chars |
| `v2-compound` (+ one sentence on rule 1) | 25/30 = **83.3%** | 0 | $0.1896 | 2,311 chars |
| `v3-evidence-first` (rules 1–2 restructured) | 25/30 = **83.3%** | 0 | $0.1886 | 2,291 chars |

Identical match rates, and identically composed: every prompt scored 5/5 on
five cases (T1, R10, T9/1, A1, A9) and **0/5 on R1**, every time, under every
contract. `route_distribution` shows no variance at all —
`{'call_tool:get_project_status': 5}` for R1 under all three prompts, with
no other route chosen even once across all fifteen R1 attempts.

**Neither candidate beat the baseline on R1, so neither is promoted.**
`PLANNER_CONTRACT` is unchanged; `V1_DIRECT` continues to resolve to it, and
`V2_COMPOUND`/`V3_EVIDENCE_FIRST` remain in `eval/routing_prompts.py` as the
record of what was tried. Per the rule stated above and in `docs/gap-plan.md`
D5: a prompt change with no measured effect is not kept.

**This is a more precise, and more sobering, finding than the one it
replaces.** The manual walkthrough's "1 route in 4" implied real
run-to-run variance a better-worded rule might resolve. This comparison
holds everything constant that the walkthrough could not — no prior
observations, no recalled memory, no session history, `temperature=0.0` —
and under those controlled conditions R1 is not 1-in-4, it is **0-in-15 for
every contract tried**: gpt-4o deterministically chooses
`get_project_status` for this exact question, regardless of which of the
three rule-wordings it is shown. Verified again outside the harness, through
the real engine (`scripts/run_turn.py --actor priya "Why is milestone M2
late and by how much?"`, three times, unedited `PLANNER_CONTRACT`): `think:
route_selected call_tool: called get_project_status` all three times. The
walkthrough's variance almost certainly came from session state this
comparison deliberately excludes — a recalled memory, a different phrasing on
a different attempt, or history from an earlier turn in the same session
(the walkthrough's own note says "asked 4 times total across the session,"
not four fresh, identical requests) — not from anything a rule 1 wording
change can reach.

## Consequences

Gap 13 stays open, but the record of it changes shape: from "an unmeasured
1-in-4" to "a measured, deterministic 0-in-15 that two different rule edits
both failed to move." `docs/e2e-code-plan.md` §5 is updated accordingly. The
follow-up this ADR points to is no longer "try a different sentence" — two
were tried, live, at real cost, and neither moved the number at all — it is
the structural fix `docs/gap-plan.md` §6 already named as the fallback: a
`documents_then_tool` route the planner can name explicitly for a compound
question, or a completeness check on the reply against the question's own
clauses, run after the fact rather than hoped for in the routing prompt.
That is out of scope for this ADR, which closes the comparison Phase L asked
for, not the routing problem itself.

The comparison harness (`eval/routing.py`, `eval/routing_prompts.py`,
`eval/routing_cases.py`, `scripts/run_routing_comparison.py`) is now proven
against a real account and stays in the repo as the regression gate for any
future candidate: a fourth contract is a fourth `RouterPrompt`, not a fourth
ad-hoc live session.

`evidence/routing/routing-comparison-2026-09-12.json` is committed alongside
this ADR — the full ninety rows, not just the summary table above.

**2026-09-14 wording change (the memory refactor, ADR 0022).** The
production `PLANNER_CONTRACT` gained its conversational rule — history is
the authority on a question about the conversation itself — between the old
rules 4 and 5. The named contract stays the production one, so the
comparison was re-run against the new wording
(`evidence/routing/routing-comparison-2026-09-14.json`, six cases ×
five repeats per contract, `principal=None` so the only diff from the
recorded baseline is the contract text). The result reproduces the baseline
exactly: 25/30 = 83.3% match rate, 0 hallucinations, on all three contracts;
the only mismatching case is still R1 (the compound question, 0/15, whose
structural closure is ADR 0021's, not a wording question); the declaration
match rate is 1.0 (30/30). Nothing dropped, so the choice recorded above
stands, and the conversational rule — about one class of question the six
cases do not contain — cost no case its expected route.

## Alternatives considered

Everything `docs/gap-plan.md`'s D5 already rejected before this ADR was
written (edit-in-place with two separate runs, one sample per pair, a
"policy-first" fourth candidate) stays rejected for the reasons recorded
there; this ADR does not reopen them.

**Promote a candidate anyway, since it "reads better."** Rejected outright:
the entire point of measuring first was to stop a prompt change from being
kept on the strength of how it reads rather than what it does. Both
candidates read at least as clearly as the baseline and neither routed a
single R1 case correctly; keeping either would be exactly the failure mode
D5 was written to prevent.

**Run more repeats to see if the 0/15 was itself unlucky.** Rejected for
now: `temperature=0.0` combined with a fixed prompt and no upstream state
produced *zero* variance across fifteen attempts per contract, not a close
distribution — a stronger signal than five more repeats would meaningfully
add to, at further real cost. If a future contract candidate ever produces a
non-zero R1 rate, repeats become the right lever to confirm it is not noise;
here there was none to confirm against.

**Investigate the walkthrough's original session for what actually varied.**
Left as the named follow-up rather than done here: reconstructing exact
prior-turn state (memory, history) from a session run days earlier is a
separate, forensic exercise, and this ADR's job was the comparison Phase L
asked for, not a retroactive audit of `638adbc`'s specific trace ids.
