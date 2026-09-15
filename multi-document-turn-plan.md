# Multi-document turn: diagnosis and fix plan

Trace under review: `run-e4feb394274f42c288be90ed37ad8c8e` (priya, 2026-09-14 23:02 UTC,
session `sess-573ad6b70829`, first turn of the session).

Request:

> Atlas project: pull the current budget and open risks from the ERP, then cross-check
> each open risk against the risk register CSV for its severity, check the Q3 budget
> summary PDF for whether contingency has been allocated for high-severity risks, and
> check the latest sprint report to see if any of those risks are already causing
> schedule slip. Summarize the full picture with sources for each claim.

Reply delivered:

> I cannot cross-check the open risks against the risk register CSV or check the Q3
> budget summary PDF and the latest sprint report, as these documents are not accessible
> through the available tools. I can provide the current budget and open risks from the ERP.

---

## 1. What the trace says happened

| seq | node | event | detail |
|---|---|---|---|
| 0 | memory | memory_recalled | 4 memories (two of them contradict: VND vs USD) |
| 1 | contract | contract_declared | `needs=document_passage,erp_field`, query = "severity of open risks, contingency allocation for high-severity risks, schedule slip due to risks" |
| 3 | think | route_selected | `call_tool: get_budget_summary` |
| 6 | execute_tool | tool_called | `get_budget_summary -> ok` ($292,800 of $480,000) |
| 9 | think | route_selected | `call_tool: list_risks` |
| 12 | execute_tool | tool_called | `list_risks -> ok` (R-1 high, R-2 medium) |
| 15 | think | route_selected | **`refuse: called refuse`** |
| 17 | memory | memory_rejected | not_established (turn ended in refuse) |

Final state: `route=refuse`, `failure=none`, `evidence=[]`, `redirected_needs=[]`,
`step_count=5` of 8. Four model calls, $0.029, no `contract_enforced` event, no
`evidence_retrieved` event. **The retriever was never called.**

The prompt the model saw before the fourth call was rebuilt from the saved state
(`build_planner_messages` over the stored `AgentState`): `search_project_documents` **was
offered**, alongside the five ERP tools, `ask_clarification` and `refuse`. The model chose
`refuse` anyway.

## 2. Root causes (four, layered)

### RC1 -- the model refused a search it was offered, because nothing told it the named documents exist

The user named three artefacts by *file type*: "the risk register CSV", "the Q3 budget
summary PDF", "the latest sprint report". The only thing the model knows about the corpus
is the search tool's description (`llm/tools.py:380`):

> Search the project documents -- status reports, meeting notes, contracts -- ...

No register, no CSV, no budget summary, no PDF, no sprint report. Rule 4 of
`PLANNER_CONTRACT` ("If ... nothing available could support an answer, call refuse") leaves
the availability judgement to the model, and the model has no catalogue to judge from. It
read "cross-check the CSV / check the PDF" as file access it does not have and refused --
stating a falsehood about two of the three documents (the risk register and the sprint
report are `project.docs.read`, which priya holds).

This is a context-construction gap, not a model bug: the manifest
(`data/documents/manifest.json`) knows exactly which documents exist and who may read them,
and none of that reaches the prompt.

### RC2 -- `refuse` is an exit that skips the reply contract (ADR 0021 loophole)

`engine/nodes.py:276`:

```python
if decision.route in ("answer", "retrieve_project_documents"):
    gap = assess(state.contract, state)
    ...
```

The contract had declared `document_passage`; no search had run; `state.evidence` was
empty. Had the model chosen `answer`, `think` would have withheld the answer and redirected
to retrieval with the contract's query. Because it chose `refuse`, the gate was never
consulted. A refusal is the model *claiming* "nothing available could support an answer" --
exactly the unverified claim ADR 0021 exists to test -- and the turn ended with
`failure=none`, indistinguishable in the trace from a refusal that was correct.

### RC3 -- one retrieval per turn cannot serve a three-document request

Even with RC1 and RC2 fixed, the graph runs **one** search with **one** query and
`EVIDENCE_LIMIT = 4`, then composes (`retrieve_and_answer`, `nodes.py:505`). The transition
table forbids `retrieve -> retrieve` by design (`transitions.py:35-42`: "a second retrieval
pass is a real design question ... deferred"), and `retrieve_and_answer` never returns
`think`.

Measured against the live index, as priya:

| query | top-4 passages |
|---|---|
| the contract's compound query | `risk-register` rows R-3, R-6, R-8, R-1 -- **no R-2, no sprint report, no budget passage** |
| "risk register severity R-1 R-2" | R-1, R-8, R-2, R-6 |
| "contingency allocated for high-severity risks Q3 budget" | `status-report-2026-09#§4`, `steering-minutes-2026-08#§4.2`, R-8, R-6 |
| "sprint 13 schedule slip caused by risks" | `sprint-13-report#preamble`, `#§3.2`, `#§1.2`, `#§2` |

One query cannot cover the three lookups; three targeted queries cover them cleanly. The
request needs multi-query retrieval, which the runtime does not have.

### RC4 -- priya is not entitled to the Q3 budget summary (this part of the request is correctly unanswerable)

`budget-summary-q3` is `required_scope: project.docs.finance.read`, `classification:
confidential`. priya holds `project.docs.read` only; wei holds the finance scope. The
Qdrant pre-filter (`rag/access.py`) removes the PDF's chunks before ranking, so for priya
the "contingency" query returns the status report §4 and steering minutes §4.2 instead --
which is the correct behaviour. Confirmed: the same query as wei returns
`budget-summary-q3#p.2 part 1` ("3.3 Contingency ...").

So even a perfect turn for priya answers the contingency sub-question from the status
report and minutes, and must say it cannot read the Q3 budget summary. Today the model
cannot distinguish "no passage found" from "not entitled", because it is told neither what
exists nor what it may read.

### Verdict on "or if I'm wrong"

You are right that the turn is defective: RC1 + RC2 produced a false refusal, and RC3 means
the request is beyond what the graph can currently do. You are partly wrong about the
expected output: as priya, the Q3 budget summary PDF **should** be off-limits, and a correct
reply says so and answers contingency from the documents priya may read. Wei is the actor
who can see the PDF.

### Unrelated, but visible in this trace

Memory recalled two live, contradictory preferences: `budget_reporting_currency` ("thousands
of VND") and `budget_reporting_format` ("thousands of USD"), neither superseding the other.
That is the key-drift problem `fix-memory-key-drift-plan.md` already covers; it did not
cause this refusal but it would have corrupted the answer.

---

## 3. Fix plan

Ordered smallest-first; each step is one commit with its own tests, and each is
independently valuable. Steps 1-2 make *this* turn stop refusing. Step 3 makes it
*complete*. Step 4 is the evidence.

### Step 1 -- `engine:` a refusal is held to the reply contract (closes RC2)

**Change** `engine/nodes.py::GraphNodes.think`:

- Extend the gate at line 276 to `("answer", "retrieve_project_documents", "refuse")`.
- `erp_field` missing + un-redirected: the existing branch already handles it (a call is
  required with search withheld); it now also applies when the model chose `refuse`.
- `document_passage` missing + un-redirected + route in `("answer", "refuse")`: redirect to
  `retrieve_project_documents` with `state.contract.document_query`. For `answer` keep
  `draft=decision.message` (unchanged). For `refuse` set **no draft**, so
  `retrieve_and_answer`'s no-passages branch (`nodes.py:541`) falls through to the ordinary
  `refuse` with `insufficient_evidence` -- a refusal that was *tested* rather than asserted.
  The model's refusal reason goes into the `contract_enforced` event detail so the audit
  shows what was overridden.
- Do not touch `clarify` or `fail`: a clarification is a question back, not a claim about
  availability; a fail is the runtime's own verdict.

**Why this and not "forbid refuse while a need is unmet":** withholding the `refuse` tool
would make a genuinely out-of-scope request that the declarer mis-declared impossible to
refuse. One redirect per need, then the model's route stands -- the same bound ADR 0021
already sets.

**Tests** (`tests/engine/test_nodes.py`, fake planner):
- contract needs `document_passage`, evidence empty, planner returns `refuse` -> route is
  `retrieve_project_documents`, `redirected_needs == {"document_passage"}`, a
  `contract_enforced` event names the refusal.
- same, but retrieval returns nothing -> terminal `refuse`, `failure=insufficient_evidence`
  (not `incomplete_reply`, because no draft).
- `document_passage` already in `redirected_needs`, planner returns `refuse` -> refuse
  stands (bound holds).
- contract `None` -> refuse stands, no event (unchecked turns remain unchecked).

**ADR 0025** -- "a refusal is held to the reply contract too", amending 0021. **Done**
(commit `aeb87f1`): `engine/nodes.py::think`'s gate widened, no draft carried for a
refusal, five new tests in `tests/engine/test_completeness.py`.

### Step 2 -- `context:`/`llm:` the model is told which documents exist (closes RC1, gives RC4 a script)

**Change:** a typed `DocumentCatalogue` (new `context/catalogue.py`), built once per turn
in `composition/turn.py` from `resources.manifest` filtered through the actor's
`RetrievalContext` -- title, `document_type`, `effective_date`, `document_id` for every
document the actor **may read**. Rendered by `llm/prompts.py::render_catalogue` and
appended to the system role after the principal block (same pattern as
`render_principal`, so `SYSTEM_POLICY` stays a constant for `eval/routing.py`). About
150 tokens for the fixture corpus.

Rendered shape:

```
Documents you can search (any format -- CSV, PDF, HTML, Markdown -- is searched the same way):
- risk-register: Atlas Risk Register (risk_register, 2026-08-31)
- sprint-13-report: Sprint 13 Review and Retrospective - Atlas (sprint_report, 2026-09-06)
- ...
Documents outside your entitlements are not listed and are never returned by search. If the
user names one, say you cannot access it; never say it does not exist.
```

Deliberate trade-off: documents the actor cannot read are **not listed** (listing a
confidential title to an unentitled actor discloses its existence), but the model gets the
right script for when the user names one. Recorded in the ADR.

Wording changes, same commit:
- `SEARCH_PROJECT_DOCUMENTS_TOOL.description`: drop the "status reports, meeting notes,
  contracts" list; point at the catalogue: "Search the documents listed under 'Documents you
  can search' ...".
- `REFUSE_TOOL.description` and `PLANNER_CONTRACT` rule 4: "'nothing available' means no
  tool and no listed document; a document the user names by format (CSV, PDF,
  spreadsheet) is searched, never refused for its format."

**Tests:** `tests/llm/test_prompts.py` (catalogue rendered, only readable documents
appear, wei sees the budget summary and priya does not); `tests/context/` for the
catalogue builder; `tests/eval` -- `PLANNER_CONTRACT` changed, so the ADR 0020
comparison is re-run in step 4.

**ADR 0026** -- "the model is told which documents it may search". **Done**
(commit pending): `context/catalogue.py::build_catalogue` (reuses
`rag/access.py::is_authorized`, no second scope check), `llm/prompts.py::
render_catalogue`, `LLMGateway.catalogue` (bound at construction, read only by
`decide()`), `composition/turn.py` builds one `RetrievalContext` shared by the
retriever and the catalogue. Tool/contract wording updated. Live-verified against
the real corpus and a real model call: the same request that produced
`run-e4feb394274f42c288be90ed37ad8c8e` now routes to
`retrieve_project_documents` instead of `refuse` -- but with only one query
("risk severity in risk register"), the sprint report is still never fetched,
confirming RC3 is exactly step 3's problem, not this one's.

### Step 3 -- `rag:`/`engine:` multi-query retrieval in one node (closes RC3)

**Change:** `search_project_documents` takes `queries: list[str]` (1-3, each non-empty)
instead of `query: str`. `retrieve_and_answer` runs each query through the same bound
retriever with `EVIDENCE_LIMIT` per query, unions the hits in query order, dedupes by chunk
id, and composes once from the union. `_query(state)` becomes `_queries(state)` with the
same fallback (the request itself) for a replay or hand-built state. `tool_arguments`
becomes `{"queries": [...]}`; the `evidence_retrieved` event reports per-query counts
(`"4+4+3 passage(s) for 3 quer(y|ies)"`).

`ReplyContract.document_query` stays a single string: the redirect is the fallback for a
model that did not search at all, and one query is the honest size of that fallback.
(Declaring a list would change `AgentState`'s schema -> `STATE_VERSION = 3`; not worth it
for the fallback path. Revisit if the routing comparison shows redirects are common.)

**Why this and not a `retrieve -> think -> retrieve` loop:** a loop costs one planner call
per pass, blows the 8-step budget on exactly this request (11 node executions), and
reopens the question the transition table deliberately closed (when does it stop?). A
bounded list answers the stop question in the schema (max 3), keeps retrieval and
composition one obligation (the reason they are one node), and adds no edge. The loop
stays a rejected alternative in the ADR; if a future case needs a search that *depends on*
a previous search's result, that is the day to add it.

`PLANNER_CONTRACT` rule 1 gains one sentence: "When the question needs more than one
document, put one query per document in `queries` -- a single blended query returns
passages from whichever document matches best and starves the rest."

**Tests:** `tests/reasoning/test_planner.py` (queries validated, empty list rejected);
`tests/engine/test_nodes.py` (three queries -> union, deduped, per-query counts in the
event; a replayed state with legacy `{"query": ...}` still runs); `tests/rag/` unchanged
(the retriever itself does not change).

**ADR 0027** -- "retrieval takes a bounded list of queries".

### Step 4 -- `eval:`/`docs:` the evidence

- `eval/routing_cases.py`: add **A12** (this request, actor priya, accepted first routes
  `call_tool:get_budget_summary` / `call_tool:list_risks` / `retrieve_project_documents`,
  `expected_needs={document_passage, erp_field}`) and **A13** (same request as wei -- the
  PDF is in scope, so the reply must cite `budget-summary-q3`).
- Re-run `scripts/run_routing_comparison.py`; the contract text changed in steps 2 and 3,
  so ADR 0020's comparison must be re-recorded, not assumed.
- `docs/manual-test.md`: one row for the priya path (reply cites `risk-register#row R-1`,
  `#row R-2`, `sprint-13-report#§2`, and says the Q3 budget summary is not accessible;
  contingency answered from `status-report-2026-09#§4`) and one for wei.
- Browser check per CLAUDE.md: send the request as priya, watch it stream, click the
  citations, console clean.

### Expected trace after the fix (priya)

```
contract_declared   needs=document_passage,erp_field
think               call_tool: get_budget_summary
think               call_tool: list_risks
think               retrieve_project_documents: called search_project_documents (3 queries)
retrieve            evidence_retrieved: 4+4+4 passage(s) for 3 quer(y|ies)
answer              cites risk-register#row R-1, #row R-2, sprint-13-report#§2,
                    status-report-2026-09#§4; states the Q3 budget summary is not accessible
```

Seven node executions, under the budget of eight. If step 1 fires instead (the model still
refuses), the trace shows `contract_enforced: document_passage: refuse withheld, searching
...` and either a cited answer or a refusal that was actually tested.

---

## 4. Order and estimate

| step | area | size | unblocks |
|---|---|---|---|
| 1 | engine | small (one branch + 4 tests + ADR) | the false refusal can no longer end a turn silently |
| 2 | context, llm, composition | medium (new module, prompt block, 3 description edits, tests) | the model knows the CSV/PDF are searchable and what to say about the one it cannot read |
| 3 | rag boundary, engine, llm | medium (argument shape, node, tests, ADR) | the request can actually be completed |
| 4 | eval, docs | small-medium (2 cases, re-run, 2 manual rows, browser check) | the defence |

Steps 1 and 2 are independent of each other; 3 depends on nothing but is only *visible*
once 1-2 stop the refusal. Do 1 -> 2 -> 3 -> 4, one commit each, on
`fpt-bao/memory-refactor` or a new branch off `dev`.
