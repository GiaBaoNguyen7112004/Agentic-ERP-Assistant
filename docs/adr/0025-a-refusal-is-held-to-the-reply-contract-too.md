# 0025 — A refusal is held to the reply contract too

**Status:** Accepted (2026-09-15)

## Context

Trace `run-e4feb394274f42c288be90ed37ad8c8e` (priya): a compound request
asking for the ERP's budget and open risks, a severity cross-check against
the risk register CSV, whether contingency covers the high-severity risk in
the Q3 budget summary PDF, and whether the sprint report shows schedule
slip from those risks. `declare_reply_contract` correctly named
`needs={document_passage, erp_field}`. Both ERP calls ran and succeeded
(`get_budget_summary`, `list_risks`). The fourth model call, with
`search_project_documents` offered alongside the rest, chose `refuse`:

> I cannot cross-check the open risks against the risk register CSV or
> check the Q3 budget summary PDF and the latest sprint report, as these
> documents are not accessible through the available tools.

Two of the three named documents (`risk-register`, `sprint-13-report`) are
`project.docs.read` — priya's own scope. Nothing was inaccessible; the
model had a working search tool and declined to use it. The turn ended
`failure=none`, `evidence=()`, with no `contract_enforced` event —
indistinguishable in the trace from a refusal that was actually correct.

ADR 0021 built exactly the mechanism to catch this kind of claim — "answer"
is never delivered with a declared need unmet, without at least one real
attempt to fetch it — but gated it to `decision.route in ("answer",
"retrieve_project_documents")`. `refuse` was left out because, at the time,
every refusal case in `eval/routing_cases.py` (A3: weather forecast in
Hanoi, A9: a write refused by an actor's own missing scope, caught by the
gateway not the planner) was a refusal that had nothing to do with a
declared document or field need — the contract in those cases correctly
declares no needs at all, so `assess()` finds nothing missing and the gate
was never exercised either way. This trace is the first case on record
where a contract named a real need and the model refused instead of trying
for it.

## Decision

**`refuse` joins `answer` and `retrieve_project_documents` as a route ADR
0021's completeness check holds to the turn's own declared contract.** A
refusal is exactly the kind of claim the check exists for — "nothing
available could support this" — and untested it is worth no more than an
ungrounded answer would be.

- `engine/nodes.py::think`'s contract gate widens to
  `("answer", "retrieve_project_documents", "refuse")`.
- The `erp_field` branch is unchanged in mechanism: a refusal with an
  un-redirected field missing is forced through the same second call
  (`tool_choice="required"`, `search_project_documents` withheld) that an
  answer or a search-first decision already gets. The model's own next
  choice is honored — it can still name the tool, ask back, or refuse a
  second time, now with the field satisfied or not.
- The `document_passage` branch widens from `decision.route == "answer"` to
  `decision.route in ("answer", "refuse")`. A refusal with the passage
  un-redirected is withheld and the turn moves to
  `retrieve_project_documents` with the contract's own `document_query` —
  the same redirect an answer gets, and gated the same way ADR 0021's
  second deviation gates the answer case: a decision that already chose
  `retrieve_project_documents` on its own is left alone regardless of which
  route triggered the check.
- **A refusal's message is never carried forward as `AgentState.draft`.**
  The answer case keeps `draft=decision.message` because a withheld answer
  is a real reply worth delivering, marked, if the redirected search finds
  nothing (ADR 0021: "a redirect never delivers less than the turn would
  have without the check"). A refusal's message is not a reply — it is the
  model's claim about *why* one is impossible, and that claim is exactly
  what is being tested. Setting `draft=None` for the refuse case means
  `retrieve_and_answer`'s no-passages branch takes its other path: an
  ordinary `refuse` marked `insufficient_evidence`, tested rather than
  merely repeated. `test_a_model_chosen_search_finding_nothing_still_
  refuses` already proved that path is correct for a model-chosen search
  with nothing found; the redirected refusal now reaches the identical
  code with the identical result, by construction rather than by
  coincidence.
- One redirect per need, unchanged: `AgentState.redirected_needs` is
  checked and set exactly as it is for `answer`, so a refusal whose need
  was already redirected once this turn (by an earlier `answer` or
  `refuse` attempt) is delivered as the terminal refusal it would have
  been without this ADR.
- The `contract_enforced` event for a withheld refusal records the model's
  own refusal reason in its detail (`"document_passage: refuse withheld,
  searching '...' (refusal reason: '...')"`), so a reviewer reading the
  trace sees exactly what claim was overridden and why.

**Rejected: withhold `refuse` from the offered tools whenever a need is
still unmet**, the same shape as ADR 0021's `erp_field` branch withholding
`search_project_documents`. Rejected because a request can be genuinely
out of scope (A3) even while carrying a contract that names a need — the
declaration step can mis-declare, and withholding `refuse` entirely would
make an honestly out-of-scope request un-refusable until the redirect
budget for every named need was spent. The one-redirect-per-need bound
already caps the cost of a wrong refusal at one extra model call and one
extra retrieval; that is cheaper than a request the model has no way to
decline.

**Rejected: a second, separate check specifically for refusals** (grading
the refusal reason's text against the contract with a second model call, or
pattern-matching the reason for phrases like "not accessible"). Both would
be exactly the self-report `reasoning/decision.py` and `reasoning/
completeness.py` already refuse to trust — a check on prose is not a check.
`assess()` already answers the only question that matters (`is evidence.
() empty and did nothing observe ok`) without reading a word the model
wrote.

## Consequences

A refusal that names a document by its file format ("the CSV", "the PDF")
without having searched for it no longer ends the turn unmodified — it is
redirected to search once, exactly as an answer would be, before the
refusal is allowed to stand. This does not, by itself, fix the trace that
motivated it: `search_project_documents`'s own description still lists only
"status reports, meeting notes, contracts" and says nothing about which
documents exist or which the model may read, so a model that reaches the
forced search still has no catalogue to reason from — that gap is
`context:`'s to close next (`multi-document-turn-plan.md` step 2), and a
single retrieval still cannot serve three separately-targeted document
questions (step 3, `rag:`). This ADR closes the part that is `engine:`'s to
close: a refusal is no longer a free pass around the contract a turn
declared for itself.

`eval/routing_cases.py`'s existing refusal cases (A3, A9) are unaffected —
both declare no needs, so `assess()` finds nothing missing and the gate
this ADR adds is never exercised for them. `docs/manual-test.md` gains a
case for the redirected-refusal path once `multi-document-turn-plan.md`'s
later steps make it reachable end to end with a real corpus and a real
model (today's fix is provable only against a fake or scripted planner,
per `tests/engine/test_completeness.py`'s new refusal cases; the live
re-run belongs with step 4 of that plan, alongside the re-recorded ADR
0020 comparison).

## Tests

`tests/engine/test_completeness.py`:
`test_an_unmet_passage_redirects_a_refusal_to_search`,
`test_an_unmet_field_redirects_a_refusal_too`,
`test_a_refusal_with_the_passage_already_redirected_stands`,
`test_a_refusal_that_already_meets_the_contract_is_not_redirected`,
`test_a_redirected_refusal_finding_nothing_ends_in_a_tested_refusal` (the
real engine, real transition table, a scripted `DecisionModel` honoring
`tool_choice`/`withhold` exactly as the provider does).
