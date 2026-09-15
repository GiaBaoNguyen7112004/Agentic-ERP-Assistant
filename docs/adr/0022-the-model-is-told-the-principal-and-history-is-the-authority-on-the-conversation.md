# 0022 — The model is told the principal, and history is the authority on the conversation

**Status:** Accepted (2026-09-14)

## Context

The memory layer looked broken from the chat, and the dev database proved the
symptoms were real (`refactor-memory-plan.md` §0, written after reading the
code *and* the rows):

- "What is my name and which project am I on?" was refused, although
  `AgentState.actor`/`project_code` (ADR 0017) and
  `User.display_name`/`role` (`data/users.json`) had held both facts the
  whole time — **no prompt ever told the model**. Every ERP tool takes a
  `project_id` argument the model had to guess, and `ask_clarification`'s
  description told it to ask when "no project" is named. The gateway *denied*
  a wrong-project call (ADR 0017) — the model was kept safe and kept ignorant
  at once.
- "What was my previous question?" was refused or clarified although
  `session_turns` held the turns and `history_recalled` fired. History **was**
  in the prompt; the prompts forbade using it. `SYSTEM_POLICY` rule 7 said
  history was "for understanding what the user is referring to, and for
  nothing else"; `PLANNER_CONTRACT` said "History is never a reason to choose
  answer without a retrieval or a tool call"; rule 4 routed anything "outside
  project delivery" to `refuse`. A question **about the conversation itself**
  had no legal route.
- The same two gaps compounded into "every turn re-asks for clarification":
  "what about sprints?" after an Atlas answer was refused as too vague,
  because the model was told not to lean on history and not told the project.

So the fix is not a new memory kind or a new tool — the state already knew
everything. What was missing was *telling the model*, and *permitting* one
class of answer the prompts had accidentally outlawed.

## Decision

**D1 — a principal block in the system role.** A frozen `Principal` value
(`llm/prompts.py`: actor, display name, role, project code, project name),
rendered by `render_principal` and appended to `SYSTEM_POLICY` at build time
(`system_content`). It goes in `system`, not a new role: it is standing
context about the session, not data to read, not a source to cite, and a
ninth `Role` would touch the adapter, the tokenizer and the wire protocol for
no gain. `SYSTEM_POLICY` itself stays a constant so `eval/routing.py` (ADR
0020) keeps sending byte-identical prompts unless it opts in. The block ends
by saying who the user is and which project this is may be stated without a
citation — session context, not a document claim. The gateway's project check
(ADR 0017) stays exactly as it is: the block makes the model *right*, the
gateway still makes it *safe*. Both halves are needed; the block is not a
replacement for the check.

**D2 — a conversational route, bounded by what it may claim.** Questions
about the conversation itself ("what did I ask?", "what did you say?"), a
greeting, a thank-you, and the acknowledgement of a stated preference are
answered directly from the `history`/`memory`/principal context — no tool, no
citation. The wording keeps ADR 0014's residual risk closed rather than
loosening it: **a claim about the project** still needs a retrieval or a tool
observation; **a claim about the conversation** never needed one, it only
needed permission. The bound is stated twice, because the two calls that must
agree are made separately: `SYSTEM_POLICY` rule 7 (the answering call) and
`PLANNER_CONTRACT` (the routing call) each name the conversation as the one
thing history is the authority on. `DECLARATION_CONTRACT` sends a
conversational question to `needs=()`, so ADR 0021's completeness redirect —
the structural guard on claims — simply never engages for it; a question that
*does* declare a need is held to it exactly as before. `REFUSE_TOOL`'s
description no longer invites routing such a question to refusal.

Both are prompt-layer changes plus one composition change (`composition/
turn.py` builds the `Principal` per turn from the user and the ERP project
name). No graph node, no state field, no tool, no schema change. Live
evidence in `docs/manual-test.md` §6's 2026-09-14 `MR` rows: `what is my
name…` answered with no citation (MR2), `what was my previous question?`
answered from history with no tool (MR5), `list the risks` routed to
`list_risks` with `project_id=atlas` filled by the model itself (the
`tests/live/test_conversational_routing.py` pair).

## Consequences

- The principal block grows every prompt by roughly 90 tokens. It is
  estimated correctly (it travels as ordinary system text), so the drift
  accounting in `tests/live/test_estimate_drift.py` is unaffected.
- Every prompt the comparison and the evals send is now built through
  `system_content`, and the block appears only when a `Principal` is passed —
  replays and the routing comparison send none, so ADR 0020's recorded
  baseline is unchanged (see its 2026-09 Consequences paragraph).
- "What about sprints?"-style **ellipsis** is *not* fixed by this ADR and was
  not expected to be: the conversational route covers the conversation, not
  resolving a pronoun onto the previous topic. MR4 shows the live model still
  refusing the elliptical follow-up twice; that sits next to intent
  inference (ADR 0014's open item), not behind this wording.
- The model can now *state* who the user is. That makes `not_established`'s
  self-description and restatement rules (ADR 0023) more important, not less:
  the model seeing the principal block proposes sentences about the user it
  never heard the user say. Live run `run-bf4c0d4b49b9` shows exactly that —
  a `user_name` fact slipped under the 80% restatement bar and was stored;
  it is redundant with the principal block and named in ADR 0023's
  Consequences.

## Alternatives considered

- **A ninth `Role`** (`principal`) between `system` and `user`. Rejected: it
  is standing context, not a channel with its own contract; a new role
  touches the adapter's role-folding, `TiktokenCounter`'s per-message framing
  test, `web/protocol.py`'s nine event types' sibling contract and every
  prompt-shape test — cost paid everywhere, benefit nowhere, since nothing
  ever needs to address the block separately (unlike `history` and `memory`,
  which have their own rules precisely because they are *data* the model
  might mis-trust).
- **Auto-filling `project_id` in the gateway** instead of telling the model
  the project. Rejected: it would work for the tool argument but leave "what
  is my name?" and "which project?" unanswered, and it would erase the
  evidence. ADR 0017's denial rows are the audit record that the model was
  wrong about the project; hiding the argument behind a silent fill would
  make a model that guesses wrong indistinguishable from one that guesses
  right. Telling the model and keeping the check keeps both the fix and the
  audit trail — the wrong-project denial (T14/T15 in `docs/manual-test.md`
  §4.3) still fires exactly as before.
- **Teaching the planner ellipsis resolution in the same pass** ("sprints"
  → the previous topic). Rejected for scope discipline: it is a routing
  capability with its own failure modes, it is adjacent to ADR 0014's
  task-in-flight work, and ADR 0020's method says contract wording changes
  are measured by the recorded comparison, not tuned to a walkthrough. Named
  as the follow-up MR4 records instead.