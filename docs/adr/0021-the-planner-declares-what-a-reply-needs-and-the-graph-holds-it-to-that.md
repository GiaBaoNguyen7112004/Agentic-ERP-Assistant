# 0021 — The planner declares what a reply needs, and the graph holds it to that

**Status:** Accepted (2026-09-12)

## Context

ADR 0020 measured gap 13 precisely and found no prompt wording fix for it.
"Why is milestone M2 late and by how much?" is two questions — the ERP holds
the number (`get_project_status` → `days_late: 2`), a project document holds
the reason (`status-report-2026-09` §2.2: the reconciliation exceptions) —
and a recorded three-contract comparison found gpt-4o routing it to the tool
alone, deterministically, 0/15 across two rule-wording candidates and five
repeats each. The reply that results is grounded (constraint 3 held every
time) and incomplete. ADR 0020's Consequences section named the follow-up
as structural, not a third prompt attempt: "a `documents_then_tool` route
the planner can name explicitly for a compound question, or a completeness
check on the reply against the question's own clauses, run after the fact
rather than hoped for in the routing prompt."

`docs/completeness-plan.md` is that follow-up, planned before implementation
and corrected twice against live behavior while building it (see Decision
and the deviations recorded in `git log` for the phases below). Three facts
about the graph, found while planning, decided its shape:

1. **Nothing typed said what a complete reply to a given question needed.**
   `PLANNER_CONTRACT`'s routing rules described the two kinds of fact in
   prose; the model applied one per call; nothing recorded that a question
   needed both. There was nothing for a check to check.
2. **`retrieve_and_answer` composed without this turn's own tool
   observations.** The one walkthrough run that answered R1 fully did so
   because the status report happens to restate the field itself — a
   tool-then-documents path was correct only by luck of the corpus.
3. **A planner that elected `answer` was believed.** `think` delivered the
   model's text with a `Sources:` trailer of observed ids and no check that
   the text rested on anything — the residual risk ADR 0014 already named
   for the `think → answer` route, and gap 1 in `docs/e2e-code-plan.md` §5.
   Gap 13 was one instance of gap 1.

## Decision

**The planner declares what a complete reply needs, through the same
function-calling channel it decides with, before anything runs — and a pure
function holds every attempt to answer or search to that declaration.**

- **A typed contract, not a routing rule.** `state/reply_contract.py::
  ReplyContract(needs: frozenset[ReplyNeed], document_query: str | None)`,
  where `ReplyNeed` is `document_passage` or `erp_field`. Declared by a
  dedicated function, `declare_reply_contract` (`llm/tools.py`), offered
  *alone* — the same way `propose_memories` is offered alone to the
  consolidation call and never to the planner mid-turn — with
  `tool_choice: "required"` so the result is always a call, never prose.
  `llm/prompts.py::DECLARATION_CONTRACT` is a separate developer block from
  `PLANNER_CONTRACT`: one tells the planner what to prefer when routing;
  this one asks it to declare, once, what its own eventual answer will need
  to point to.
- **Declared once per turn, by the orchestrator, not by `think`.**
  `AgentState.contract` is filled by `RunOrchestrator._declared`, in the
  same position as memory and history recall — after both, before the
  engine — under the same never-fail discipline: a declarer that raises
  leaves `contract=None` ("unchecked"), never fails the request.
  `Planner.declare` separately absorbs an *unreadable* declaration (prose,
  the wrong function, arguments the schema rejects, a query/needs mismatch)
  into `EMPTY_CONTRACT` — a genuine declaration of nothing needed — with a
  logged warning, so a garbled reply costs the turn its check, never its
  answer. `resume()` does not declare again: the paused state already
  carries what it had.
- **The check is a pure function over typed state.**
  `reasoning/completeness.py::assess(contract, state) -> Completeness` reads
  exactly two things: whether `state.evidence` is non-empty
  (`document_passage`) and whether any of `state.observations` has
  `status == "ok"` (`erp_field`) — deliberately including a mutating tool's
  own receipt, since a write's id is a live ERP fact the same way a read is.
  No reply text, no rationale, no second model call grading the first — the
  same discipline `reasoning/decision.py` already applies to `rationale`
  and `state/events.py` to `detail`.
- **A missing need is redirected once, in `REDIRECT_ORDER = ("erp_field",
  "document_passage")`** — the field before the passage, because
  `retrieve_and_answer` composes at the end of the turn and should already
  have the field in `observations` by then. `engine/nodes.py::think` applies
  this after ADR 0019's repeated-call guard and its belt-and-suspenders
  check, before the route branches: a decision to answer or retrieve with
  `erp_field` still missing is redirected by a second planner call
  (`tool_choice="required"`, `search_project_documents` withheld — the tool
  is left out of what is offered entirely, distinct from ADR 0019's
  `tool_choice="none"`, which still offers everything); a decision to answer
  with only `document_passage` missing is redirected straight to
  `retrieve_project_documents` with the contract's own query, and the
  planner's withheld text is kept as `AgentState.draft`. Each need is
  redirected at most once (`AgentState.redirected_needs`).
- **A redirect never delivers less than the turn would have without the
  check.** If the redirected search finds nothing, `retrieve_and_answer`
  delivers the withheld `draft` anyway — marked `failure="incomplete_reply"`
  (a new `FailureMode`, kept apart from `insufficient_evidence`: evidence
  was not lacking for what was said, something the planner itself declared
  necessary was never fetched) — rather than a bare refusal of a question
  the ERP had already half-answered. A model-chosen search that finds
  nothing still refuses exactly as it always has; the softer landing is
  only for a redirect the check itself made. *(Extended 2026-09-15, same
  rule, two more paths: a redirected search that finds passages which do
  not ground a reply -- the composer refuses, cites something never
  retrieved, or breaks its own schema by answering "grounded" with no
  citation, which is what a model that answered from memory or history
  rather than the passages sends back -- lands the same way. Session
  `sess-25f74ac93cd6` found the gap: a fact the user had established in
  memory, absent from every document, ended as `provider_failure` after the
  redirect instead of the memory-backed reply the planner already had. A
  provider that never answered is still a provider failure.)*
- **The composer sees this turn's own observations.** `llm/prompts.py::
  build_messages` grows a seventh block, `observation`, between `evidence`
  and `history` — the same block `build_planner_messages` and
  `build_memory_messages` already carried. `AnswerComposerPort.answer` and
  `LLMGateway.answer` thread `state.observations` through, and
  `retrieve_and_answer`'s `Sources:` trailer merges the composer's own
  citations with the observed ids, citations first, deduped. Without this,
  a redirected compound answer would compose from the passage alone and
  drop the field the tool already established — the same gap 13 named for
  a retrieval-only reply, moved to the composing side of it.
- **Two new trace events, one new failure mode.** `contract_declared`
  (orchestrator-emitted, alongside `memory_recalled`/`history_recalled`) and
  `contract_enforced` (node-emitted, the one member of this family charged
  to the step budget — a redirect is real work the turn is doing, not
  background filled in around it). `incomplete_reply` joins `FailureMode`.

**Two deviations from `docs/completeness-plan.md`'s written text, both found
by tracing every path before writing the branch, both recorded in the
commit that made them:**

1. **`ContractDeclarerPort` lives in `engine/orchestrator.py`, not
   `engine/ports.py`**, as the plan's own target layout said. `engine/
   ports.py`'s module docstring states its own promise directly — "the
   workflow's entire external surface is four protocols on one screen" —
   and `TurnMemoryPort`/`SessionHistoryPort` already explain why a
   declaration-shaped collaborator belongs beside them instead: the graph
   does not depend on a reply contract at all, `engine/nodes.py::think`
   reads `AgentState.contract` directly off the state, the same way it
   reads `evidence` or `observations`, never through a port. Following the
   plan's literal placement would have broken a promise the codebase
   already keeps for the identical case.
2. **The `document_passage` redirect is explicitly gated on
   `decision.route == "answer"`**, which the plan's pseudocode asserted in
   a comment ("only reachable with route == 'answer'") without the
   corresponding runtime check. As written, the redirect would also fire
   when `decision.route == "retrieve_project_documents"` and
   `document_passage` merely happened to be the next missing need in order
   (`erp_field` already satisfied) — a reachable case where the model had
   already, correctly, chosen to search with its own query. Firing there
   would discard that choice, substitute the contract's query for it, and
   set `draft` from `decision.message`, which is `None` on that route.
   Covered by
   `test_a_model_chosen_search_is_left_alone_even_with_a_passage_missing`.

## Consequences

Gap 13 (`docs/e2e-code-plan.md` §5) is closed structurally, not by prompt
wording — routing stays exactly as model-dependent as ADR 0020 measured it;
what changed is that an incomplete answer downstream of that routing no
longer reaches the user unmarked. Gap 1 (ADR 0014's residual risk on
`think → answer`) is closed for every turn whose declared contract names a
need; a turn that declares `needs=()` for a question that actually needed
something remains model-dependent — named as gap 16 in
`docs/completeness-plan.md` §6, with a concrete, deliberately-not-yet-built
fallback (a keyword rule that only *widens* a declaration, never narrows
one) if live evidence ever shows the need.

Cost: one more model call per turn from the point this shipped — a
declaration, ~1.1–1.4k input tokens on the live runs below. A turn whose
contract redirects spends one or two more calls than it would have (the
`erp_field` redirect's forced call, or the `document_passage` redirect's
retrieval) — never more than one extra step per need, bounded by
`redirected_needs`.

**Live-verified against the real OpenAI API and Postgres** (all trace ids
and the full declaration comparison are in `docs/manual-test.md` §6's
2026-09-12 rows, referencing this ADR):

- R1, three times: every run declares `{document_passage, erp_field}`,
  calls `get_project_status` on its own (no `erp_field` redirect needed
  against the live model — it routes the field correctly, exactly as ADR
  0020 found), tries to answer, gets redirected to `retrieve_project_documents`
  with the contract's own query, and now composes a reply stating both the
  two-day delay and the reconciliation-exception cause in one answer —
  `Sources:` lists both the document locator and `milestone-m2`. `failure=none`
  on every run.
- T1, R10, T9/1, A1 (approved) and A9: unchanged in route, reply shape and
  step count from their pre-ADR-0021 baselines. A1's live trace shows ADR
  0019's forced post-write answer and ADR 0021's completeness check
  composing without interfering (`needs=(none)` for a write — D1's own
  rejection of a needs field on tool arguments applies here too).
- The recorded declaration comparison (`evidence/routing/routing-comparison-
  2026-09-12-adr0021.json`, `scripts/run_routing_comparison.py`): the same
  90-row routing measurement ADR 0020 ran, reproducing its exact numbers
  (25/30 = 83.3% match per prompt, 0 hallucinations, R1 still
  `call_tool:get_project_status` 15/15 — confirming this ADR changed
  nothing about routing), plus 30 new declaration rows, one per case ×
  repeat, independent of which routing prompt was under comparison.
  **15 of 15 scored declarations matched exactly**: R1 declared
  `{document_passage, erp_field}` with a non-blank query every time; T1 and
  T9/1 declared `{erp_field}` alone every time. R10/A1/A9 declared
  `needs=(none)` every time, unscored by design (a refusal or a write has
  no reply-contract expectation to be right or wrong about).

`engine/transitions.py`'s edges, `tools/gateway.py`, and `rag/access.py`
are untouched by this plan — checked directly (`git diff --stat` against
the pre-Phase-N commit is empty for all three) and proven by every existing
approval-flow and access-control test still passing unchanged, plus A1/A9's
live traces matching their pre-ADR-0021 shape exactly.

## Alternatives considered

**A keyword rule over the question** ("why" → `document_passage`).
Rejected: not the model's own declaration, brittle across phrasings that
don't use the trigger words, and wrong exactly when the question is
unusual — which is when a rule is needed most. Kept as the named,
deliberately-not-built fallback (gap 16) if live evidence ever shows the
declaration itself unreliable, and scoped narrowly even then: it would only
ever *widen* a declaration, never narrow one, so it could not become a new
way to under-declare.

**A `needs` field on every tool's argument schema.** Rejected: it would
land in `arguments_summary`, the audit row, and the approver's card, none
of which are about what the *reply* needs — a write's arguments describe
the write, not a claim about completeness.

**Reading a JSON block from the content beside a tool call.** Rejected
outright: a prose channel, which ADR 0006 (function calling is the only
decision channel) exists to refuse.

**A `documents_then_tool` route the planner can name explicitly.** Rejected
as the primary mechanism, per ADR 0020's Consequences section: a new route
would need a new edge in `engine/transitions.py` for one question shape,
and would still need fact 2 (the composer not seeing observations) fixed to
actually work — fixing fact 2 turned out to be most of the answer on its
own, without a new route at all.

**A second model call grading the first** ("is this reply complete?").
Rejected: an unmeasured judge of an unmeasured answer, and exactly the kind
of behavior CLAUDE.md's constraint 6 keeps out of free-form prompt strings
when it should be typed state — the declaration-then-check split is the
typed alternative.

**Chaining ADR 0019's repeated-call guard onto the `erp_field` redirect's
own forced call.** Considered and left out of scope: the redirect fires at
most once per turn (bounded by `redirected_needs`), so a redirected call
that happened to repeat an earlier one would cost one wasted call, not an
unbounded loop — `MAX_STEPS` is still the hard backstop regardless. Adding
a second guard here for a case the existing bound already contains was not
worth the complexity.

**Failing the turn on an unmet contract, instead of delivering the draft or
the composed reply anyway.** Rejected: strictly worse for the user than the
turn would have been without this ADR at all, for a mechanism meant to add
to answers, not subtract from them. Marking the reply
`failure="incomplete_reply"` gives the trace an honest record without
costing the user the words the planner already had.
