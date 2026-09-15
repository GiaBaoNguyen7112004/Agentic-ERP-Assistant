# 0027 — Retrieval takes a bounded list of queries

**Status:** Accepted (2026-09-15)

## Context

ADR 0025 stopped the false refusal; ADR 0026 gave the model a catalogue to
search against. A live check against the real corpus (`scripts/run_turn.py
--actor priya`, same request as `run-e4feb394274f42c288be90ed37ad8c8e`)
after both fixes showed the model correctly choosing to search -- and then
running one query, `"risk severity in risk register"`. `retrieve_and_answer`
(`engine/nodes.py`) runs exactly one search per turn, ranked against one
query, truncated to `EVIDENCE_LIMIT` (4). Measured against the live index,
one query cannot serve the three documents this request names:

| query | top-4 passages |
|---|---|
| the contract's own compound query | `risk-register` rows R-3, R-6, R-8, R-1 -- no R-2, no sprint report, no budget passage |
| "risk register severity R-1 R-2" | R-1, R-8, R-2, R-6 |
| "contingency allocated for high-severity risks Q3 budget" | `status-report-2026-09#§4`, `steering-minutes-2026-08#§4.2`, R-8, R-6 |
| "sprint 13 schedule slip caused by risks" | `sprint-13-report#preamble`, `#§3.2`, `#§1.2`, `#§2` |

Three targeted queries cover the request cleanly; the single compound query
that best represents the whole question does not even return the sprint
report at all. The live check confirmed it: with one query, the sprint
report was never fetched, and the reply had nothing to say about schedule
slip. `engine/transitions.py` also forbids `retrieve_project_documents ->
retrieve_project_documents` by design ("a second retrieval pass is a real
design question... opening the edge before answering that invites a loop
with nothing but the step budget between it and the user") -- so the fix is
not a second pass, it is giving the one pass more than one thing to look for.

## Decision

**`search_project_documents` takes `queries: list[str]` (one to three, none
blank) instead of `query: str`, and `retrieve_and_answer` runs every one of
them in the same node, unions the hits, dedupes by citation tag, and
composes once.**

- `llm/tools.py::SearchProjectDocumentsArguments.queries`: `list[str]`,
  `min_length=1`, `max_length=SEARCH_QUERY_LIMIT` (3), plus a
  `model_validator` rejecting a blank entry the length constraint alone
  cannot express. `SEARCH_QUERY_LIMIT` is declared once, here -- the
  boundary that actually receives a call from the wire -- not duplicated
  as a second check in the decision layer: `reasoning/decision.py::
  ReasoningDecision.search_queries` validates only "non-empty on
  `retrieve_project_documents`, empty elsewhere, no blank entries," on the
  same grounds that module already gives for not validating
  `required_tool` against the tool registry -- a decision the schema has
  already bounded does not need a second, drifting copy of the bound.
- `Planner._control` carries `tuple(validated.queries)` into
  `ReasoningDecision.search_queries` (replacing the old `search_query: str
  | None` field). `engine/nodes.py::think` writes `tool_arguments =
  {"queries": [...]}` at both call sites: the model's own choice, and the
  `document_passage` redirect (ADR 0021/0025), which always sends exactly
  one query -- the contract's own `document_query`, still a single string
  (see below).
- `GraphNodes._queries(state)` (renamed from `_query`) reads
  `tool_arguments["queries"]`, falls back to the legacy singular `"query"`
  key wrapped as a one-entry tuple (a state built or replayed before this
  ADR stays runnable), and falls back to `(state.request,)` last -- the
  same three-level fallback `_query` already had, widened by one level.
- `retrieve_and_answer` runs `self.retriever.search(query, limit=
  self.evidence_limit)` once per query -- `EVIDENCE_LIMIT` stays a
  per-query budget, not a total, so three queries may return up to twelve
  raw hits before dedup -- then unions them in query order and dedupes by
  `EvidenceSnippet.tag` (`_dedupe_by_tag`, `engine/nodes.py`): the exact
  string the model is told to cite, so two snippets that would render the
  same citation are one passage for this purpose regardless of which
  query surfaced the copy. First occurrence wins, so a passage two
  queries both match keeps the rank its earlier query gave it. The
  `evidence_retrieved` event reports per-query raw counts and every query
  string (`"4+4+4 passage(s) for 3 queries: 'a', 'b', 'c'"`), so a trace
  reader sees both how much each query found and what was actually
  asked.
- **`ReplyContract.document_query` stays a single string.** It is the
  redirect fallback for a model that did not search at all (ADR 0021), and
  one query is the honest size of that fallback -- widening it to a list
  would grow `AgentState`'s schema (`STATE_VERSION` bump) for a path that
  exists only when the model skipped searching, not when it searched with
  too few queries.

**Rejected: a `retrieve -> think -> retrieve` loop**, letting the model
search again after seeing the first pass's results. Costs one planner call
per pass and, measured against this exact request, needs eleven node
executions against an eight-step budget -- the loop would blow the budget
on the case it was built for. It also reopens the question
`engine/transitions.py` deliberately left closed: "a second retrieval pass
is a real design question (what changed about the query? when does it
stop?)". A bounded list answers the stop question in the schema (at most
three) rather than in a runtime guard, keeps retrieval and composition one
obligation (the reason `retrieve_and_answer` is one node --
"an answer to a document question exists only if passages back it"), and
adds no new transition-table edge. If a future case needs a search that
depends on a *previous* search's result -- not just a second document, but
a follow-up question about what the first one said -- that is the day to
open the loop; nothing here forecloses it.

**Rejected: raise `EVIDENCE_LIMIT` instead of allowing multiple queries.**
A single query ranks against one embedding; raising the limit to 12 would
mean paying for three documents' worth of passages while still ranking
them all against one blended intent, and the live measurement above shows
the top-4 for the blended query is *already* four risk-register rows --
raising the limit to 12 for that query returns more risk-register rows,
not the sprint report. The problem is what is being ranked against, not
how many results come back.

## Consequences

Live check, same request, after all three steps
(`multi-document-turn-plan.md`): the model declares one query per
document ("severity of risks R-1 and R-2", "contingency allocated for
high-severity risks", "risks causing schedule slip in latest sprint
report"), all three run in one `retrieve_project_documents` step
(`4+4+4 passage(s) for 3 queries`), and the reply cites
`risk-register#row R-1`, `#row R-2`, `sprint-13-report#§1.2`, `#§2` --
correctly identifying that R-2 (not R-1) is the risk already causing
schedule slip, sourced from the sprint report. Six model calls, five node
executions after the contract declaration, well under the eight-step
budget.

**Not fully closed by this ADR:** the reply's sentence about contingency
("not explicitly mentioned in the Q3 budget summary") is imprecise -- priya
was never shown the Q3 budget summary at all (RC4: `project.docs.finance.
read` is wei's scope, not hers), so the honest claim is "I cannot access
the Q3 budget summary" rather than "the summary doesn't mention it." The
catalogue (ADR 0026) tells the model this when it *names* a document it
cannot reach; it does not yet stop the composer from phrasing an
answer-shaped sentence about a document it was never shown snippets of.
That is a composer-prompt gap, not a retrieval one, and is deliberately
left for a follow-up rather than folded into this change.

`eval/routing_cases.py`'s A1/A9 cases are unaffected (neither exercises
`search_project_documents`). `PLANNER_CONTRACT` changed again in this
commit (rule 1 gains the multi-query sentence), on top of ADR 0026's
change to the same rule -- the ADR 0020 routing comparison has now
accumulated two contract-text changes since it was last recorded and must
be re-run before its numbers mean anything, per `multi-document-turn-plan.
md` step 4.

## Tests

`tests/llm/test_tools.py`: the schema declares `minItems`/`maxItems`
correctly; one, three, empty, four, and blank-among-real query lists;
the legacy singular `query` argument is rejected outright (ADR 0027
replaced the shape, not extended it).

`tests/reasoning/test_planner.py`: a single query and several queries both
carry through to `search_queries` in order.

`tests/engine/test_nodes.py`: the legacy singular key still works
(backward compatibility); every planner query is actually searched, in
order; a passage two queries both match is shown to the composer once,
first occurrence's rank kept; the `evidence_retrieved` event names every
query and its raw count. `tests/engine/test_completeness.py`'s redirect
tests updated for the new `{"queries": [...]}` shape (one entry, from the
contract's single `document_query`).

Live: `scripts/run_turn.py --actor priya`, the exact request from
`run-e4feb394274f42c288be90ed37ad8c8e`, 2026-09-15 -- see Consequences.
