# Gap plan: what the manual walkthrough found, and how each one closes

**Audience:** the implementing model and the reviewer. Companion to
`docs/e2e-code-plan.md` (the build) and `docs/manual-test.md` (the walkthrough,
results in its §6, commit `638adbc`). Same rules as the code plan: every step
names the files it touches, the tests that prove it, the live verification, and
the commit message; the repo is green after every step.

**Read first:** ADR 0001 (one tokenizer authority), ADR 0005 (the graph is a
cycle bounded by a step budget), ADR 0006 (function calling is the only decision
channel), ADR 0016 (preflight before the pause), `docs/e2e-code-plan.md` §5
(known gaps 1–10 — this plan adds 11–14 and closes three of them).

---

## 0. What the walkthrough found (each verified in code before this plan was written)

Sixty-five scenarios were walked against the real stack (gpt-4o, Qdrant,
Postgres, a browser). Everything access-control, approval, citation, and
persistence related passed. Four things did not, or were found on the way:

| # | Scenario | Symptom | Root cause (file) | Kind |
|---|---|---|---|---|
| 11 | A11 | An approved `create_risk` for `orion.lead` wrote R-6 correctly, then the planner called `list_risks(project_id=orion)` three more times and the turn ended `max_steps_exceeded` with no reply. | `llm/prompts.py::PLANNER_CONTRACT` tells the model "never say or imply that anything has been recorded" — correct *before* approval, contradicted by the `create_risk -> ok: Recorded R-6 …` observation *after* it. The model cannot reconcile the two and keeps re-reading the register. The only guard against a repeated identical call is a sentence in the same prompt; `engine/nodes.py::think` has no structural one. | **Real defect.** The write is safe (approval, audit, project binding all correct); the user never sees a reply. |
| 12 | E4 | `estimated_input_tokens - input_tokens` ≈ −1,300 on every `routed` call out of ~2,600 actual (≈50 % under). | `llm/gateway.py` estimates with `counter.count_message_tokens(messages)` — the port-role messages only. The request the adapter actually sends (`llm/adapters/openai_chat.py::_build_payload`, `call_with_tools`) also carries (a) the `tools` array — nine function definitions with descriptions and JSON schemas, (b) `response_format` with `GroundedAnswer.model_json_schema()` on the answering call, and (c) the relabel preambles the wire fold prepends to `evidence`/`observation`/`history`/`memory`. None of the three are counted. | **Real defect** in the budget check's number. Harmless at 128k today; wrong by construction. |
| 13 | R1 | "Why is milestone M2 late and by how much?" took the retrieve-and-cite route once in four tries. The other three called `get_project_status` only, answered "two days" and dropped the "why" — grounded, but incomplete. | `PLANNER_CONTRACT` rule 1 (needs a quoted explanation → documents) and rule 2 (asks for an ERP field → tool) both match; nothing says what to do when both do, and nothing measures how often the model picks each. | **Routing inconsistency**, not a citation defect (constraint 3 held every time). Needs a measurement before a fix, and a regression gate after. |
| 14 | — | After the walkthrough's finishing check (`uv run pytest -q`) the evidence store held one row, `run-schema-test`. All sixty-five traces — the graded evidence — were gone. | Five `tests/persistence/test_postgres_*.py` modules build their `database` fixture with `connect()` and no URL, i.e. `POSTGRES_URL` or the compose default — the **same** database the server writes to — and `TRUNCATE` it per test. Running the test suite with the dev container up destroys every trace, pause, memory and session recorded so far. | **Real defect** in the test harness. Constraint 5 says traces are the audit evidence; the finishing check the project itself mandates deletes them. |

Two things were **not** verified and are carried, not fixed:

- Fifteen cells of the §4.7 access matrix were not individually re-run
  (`T1`/`T2`/`T3`/`R1`/`R2`/`R8` for actors whose code path was already proven
  by a neighbouring cell). Phase M re-runs them once the evidence store
  is isolated, so the log stops saying "reused".
- `TRUNCATE` of the dev store from the assistant's own shell was blocked by the
  sandbox; it did not matter because the store was nearly empty, and after
  Phase I it never needs to happen again.

---

## 1. Decisions taken by this plan (do not re-open them while implementing)

- **D1 — Tests get their own database; the dev store is never a test fixture.**
  `POSTGRES_TEST_URL` (default `postgresql://agentic_erp:agentic_erp@localhost:5432/agentic_erp_test`)
  is the only URL a test may connect to, and the fixture refuses any database
  whose name does not end in `_test`. The alternative — a `--keep-evidence`
  flag, or tests that clean up after themselves row by row — still leaves one
  wrong invocation able to erase the record. A name check is a fact, not a
  discipline.
- **D2 — A completed write ends the planning loop structurally, not by prompt.**
  After a mutating tool returns `ok`, the planner is called once more with **no
  tools offered** (`tool_choice: "none"` on the wire), so the only thing it can
  do is answer. A prompt sentence ("after a write, answer") would be the fourth
  sentence in `PLANNER_CONTRACT` asking the model to stop doing something, and
  the walkthrough is the evidence that the existing three are advisory. The
  route table already forbids a second unapproved write in the turn
  (`transitions.py`); this closes the read-loop the same way — by taking the
  option away rather than asking.
- **D3 — A repeated identical call is a guard in `think`, not a prompt rule.**
  If the decision names a tool and arguments whose `arguments_summary` already
  appears in `state.observations` with status `ok`, the call is not executed:
  the planner is re-asked once with no tools (D2's mechanism), and if that
  still does not answer, the turn ends `failed` with a new failure mode
  `planner_loop` — a typed reason, not a `max_steps_exceeded` that says nothing
  about *why* the budget went. The prompt sentence stays (it costs nothing);
  the guard is what the trace can prove.
- **D4 — The estimate counts the wire request, and the adapter says what the
  wire request is.** ADR 0001's rule — one tokenizer authority — is kept: the
  counter still does all counting. What changes is *what it is handed*: the
  port grows `estimate_payload(messages, tools=…, structured=…) -> str`, the
  adapter's own serialisation of the request it is about to send (folded
  roles, preambles, `tools`, `response_format`), and the gateway counts that
  string plus the documented per-message framing. A provider-side count
  endpoint is rejected: it costs a round trip per call to save a local
  computation, and the whole point of the budget check is that it runs before
  anything is sent.
- **D5 — The planner contract is chosen by a recorded comparison, not edited
  in place.** Three contracts — the production one as the baseline and two
  candidates — are run against the *same* six fixed cases through the *same*
  client, `N` repeats each, in one run, so the only variable is the prompt and
  the model's day-to-day variance is shared. The result is typed rows (chosen
  route, accepted routes, match flag, hallucinated-tool flag, cost), not a
  paragraph; an ADR applies a selection rule written *before* the run and
  promotes the winner into `PLANNER_CONTRACT`. Two of the six cases are writes,
  so a candidate that fixes R1 by pushing everything to documents is caught on
  the approval path. Rejected: editing the contract and comparing two separate
  runs (prompt change and run-to-run noise are confounded — the R1 finding *is*
  a variance finding, 1 of 4); one sample per prompt-by-case pair (would have
  shown R1 as pass or fail by luck); a "policy-first" candidate that asks the
  planner to apply authorization before routing (that is the gateway's job,
  ADR 0016 — a planner refusing on the actor's behalf would be measured on a
  job it must not have). The shape follows the reference project's router
  comparison (`run_comparison(prompts, cases, client)` → `RouterResult` rows);
  what is adapted is stated in Phase L.
- **D6 — Nothing in this plan touches the approval flow, the gateway order, or
  the access rules.** They passed; their scope is closed.

---

## 2. Target layout after this plan

```
src/agentic_erp_assistant/
  engine/nodes.py            think(): repeated-call guard (D3), post-write answer (D2)
  engine/transitions.py      unchanged edges; a comment naming the new failure mode
  reasoning/decision.py      FailureMode + "planner_loop"
  reasoning/planner.py       plan(state, *, offer_tools: bool = True)
  llm/ports.py               ToolCallingClient.call_with_tools(..., allow_tools=True)
                             + estimate_payload(...)
  llm/adapters/openai_chat.py  tool_choice "none" when tools are withheld; estimate_payload
  llm/gateway.py             estimates the payload, not the messages
  llm/tokenizer.py           count_request_tokens(payload_text, message_count)
  llm/prompts.py             build_planner_messages(..., contract=PLANNER_CONTRACT);
                             PLANNER_CONTRACT <- the winner ADR 0020 names
  eval/routing_prompts.py    V1_DIRECT (= PLANNER_CONTRACT, imported), V2_COMPOUND, V3_EVIDENCE_FIRST
  eval/routing_cases.py      six fixed cases, two of them writes
  eval/routing.py            run_comparison(prompts, cases, gateway_factory, repeats) -> RouterResult rows
  persistence/connection.py  test_url_from_environment() + the _test name check
tests/
  persistence/conftest.py    ONE database fixture, shared by the five modules
scripts/
  run_routing_comparison.py  -> evidence/routing/routing-comparison-<date>.json
  init_postgres.py           --test flag: creates agentic_erp_test and applies the schema
docs/adr/0018-…              tests own a database, the dev store is the record
docs/adr/0019-…              a completed write ends the loop by withholding tools
docs/adr/0020-…              the planner contract is chosen by a recorded comparison
evidence/routing/            the comparison report(s), committed
```

---

## 3. Step-by-step

### Phase I — Isolate the evidence store from the test suite (gap 14; do this first)

Everything after this phase re-runs live scenarios and reads the evidence
back. Until the suite stops truncating the dev store, every finishing check
erases the proof the next phase is about to cite.

**I1. A test URL, and a refusal to truncate anything else.**

- `persistence/connection.py`: `TEST_DATABASE_SUFFIX = "_test"`,
  `DEFAULT_POSTGRES_TEST_URL`, `test_url_from_environment()` reading
  `POSTGRES_TEST_URL`, and `assert_test_database(url)` raising
  `StoreConfigurationError` unless the database name ends in the suffix.
- `tests/persistence/conftest.py` (new): one `database` fixture, module-scoped
  the way the five copies are today, that calls `assert_test_database` *before*
  connecting, applies the schema, and yields. Delete the five per-module copies;
  each module's `store_connection`/truncating fixture stays but takes the shared
  one.
- `scripts/init_postgres.py --test`: `CREATE DATABASE agentic_erp_test` if
  absent (connecting to the dev database to issue it — the compose user owns
  both), then applies the schema there. `docker-compose.yml` comment and
  `CLAUDE.md` Commands block updated: `uv run python scripts/init_postgres.py
  --test` before `uv run pytest -m postgres`.
- `.env.example`: `POSTGRES_TEST_URL=` with the comment that it is *only* read
  by tests and must name a `_test` database.
- Tests: `tests/persistence/test_connection.py` — `assert_test_database`
  accepts `…/agentic_erp_test`, refuses `…/agentic_erp`, refuses a URL with no
  database; the fixture skips (not fails) when the test database is absent,
  with a message naming `init_postgres.py --test`.
- Verify: with the dev store holding at least one real run, `uv run pytest -q`
  and `uv run pytest -m postgres` both green, then
  `SELECT count(*) FROM runs` on the dev store unchanged. That query, before
  and after, goes in the commit message.
- Commit: `persistence: tests get a database of their own, and refuse any other`.

**I2. ADR 0018** — *The test suite owns a database; the dev store is the
record.* Context: constraint 5 and what `638adbc`'s finishing check did to it.
Alternatives: transactional tests with rollback (rejected: the adapters commit,
and `TRUNCATE … CASCADE` is what makes the tests observe a fresh store); a
guard flag (rejected: a discipline, not a fact). Commit:
`docs: ADR 0018 -- the evidence store is never a test fixture`.

### Phase J — A completed write ends the loop (gap 11)

**J1. The port can withhold tools.**

- `llm/ports.py::ToolCallingClient.call_with_tools(..., allow_tools: bool = True)`.
  With `allow_tools=False` the tools are still passed (the model may need to
  read the definitions to understand the observations) but the call is made
  with `tool_choice: "none"`, so the result is always content.
- `llm/adapters/openai_chat.py`: the payload's `tool_choice` follows the flag.
- `llm/gateway.py::decide(..., allow_tools=True)` threads it through; the
  telemetry row's `detail` records `tool_choice=none` so the trace shows the
  call was forced.
- Tests: `tests/llm/adapters/test_openai_chat.py` — payload carries
  `"tool_choice": "none"` when withheld, `"auto"` otherwise; the fake client
  in `tests/llm/conftest.py` records the flag.
- Commit: `llm: the planner call can withhold tools and force an answer`.

**J2. The planner and the engine use it.**

- `reasoning/planner.py::plan(state, *, offer_tools: bool = True)`; with
  `offer_tools=False` a tool call in the reply is `_unreadable` ("the model
  called a tool after tools were withheld") → `fail`, never executed.
- `reasoning/decision.py`: `"planner_loop"` added to `FailureMode` with the
  docstring "the planner repeated a call that had already succeeded and did
  not answer when tools were withheld".
- `engine/nodes.py::think`:
  1. `_last_write_succeeded(state)`: the most recent observation is from a
     mutating tool with status `ok` → call `self.planner.plan(state,
     offer_tools=False)`; event `route_selected answer: tools withheld after
     <tool> succeeded`.
  2. `_repeats_a_success(state, decision)`: the decision's tool and
     `arguments_summary` (built by `tools/gateway.py::_summarize` — export it
     and reuse it, do not re-implement) match an `ok` observation → event
     `planner_loop <tool> repeated with the same arguments; tools withheld`, one
     forced plan; if that still names a tool → `advance(state, "fail",
     failure="planner_loop", …)`.
  3. Everything else unchanged. `request_approval` is *not* affected by (1):
     a second write in the same turn is already the transition table's
     business, and preflight's `approval="not_required"` deviation stands.
- `llm/prompts.py::PLANNER_CONTRACT`: the "never say or imply that anything
  has been recorded" sentence becomes "…until an observation says
  `create_risk -> ok`; then say exactly what it recorded." The prompt stops
  contradicting the observation; the guard is what enforces the loop's end.
- Tests: `tests/engine/test_think_after_write.py` — (a) scripted planner that
  would call `list_risks` after a `create_risk -> ok` is invoked with
  `offer_tools=False` and its answer becomes the reply, 1 extra step; (b) a
  scripted planner that repeats `list_risks(project_id=orion)` after it
  succeeded is forced once, then the run fails `planner_loop`, `step_count`
  ≤ 4, never `max_steps_exceeded`; (c) T5's two-different-reads path is
  untouched; (d) `test_a_resumed_turn_can_pause_again_and_waits_anew` still
  passes (a second, *different* write is still a pause).
- Verify live: the A11 request as `orion.lead`, approve as `sponsor`
  (`scripts/run_turn.py --approve`) → the reply names the new risk id and
  project, `step_count` ≤ 5, one `routed` call with `tool_choice=none` in
  `model_calls.detail`. Reset `data/erp/project.json` after.
- Commit: `engine: a completed write ends the loop by withholding tools, and a
  repeated call is a typed failure`.

**J3. ADR 0019** — *A completed write ends the planning loop by withholding
tools.* Names D2 and D3, the prompt-contradiction root cause, and the
rejected alternatives: a smaller step budget after a write (rejected: it
turns the symptom into a different symptom), a "you already recorded it" line
in the observation (rejected: the observation already says so — that was the
contradiction), a `post_write_answer` node (rejected: one more node for a
decision the existing `think` already makes, with the tools it is given being
the only difference). Commit: `docs: ADR 0019 -- a write ends the loop
structurally`.

### Phase K — The estimate counts what is sent (gap 12)

**K1. The adapter exposes the wire form; the counter counts it.**

- `llm/ports.py`: `estimate_payload(messages, *, tools: Sequence[ToolSpec] |
  None, structured: bool) -> tuple[str, int]` on both client protocols — the
  text the counter should tokenise (folded messages with their preambles, the
  `tools` JSON when given, the `response_format` schema when `structured`), and
  the number of wire messages (for the per-message framing). Pure; sends
  nothing.
- `llm/adapters/openai_chat.py`: implements it by building the same payload
  `_build_payload`/`call_with_tools` build and serialising the relevant parts
  with `json.dumps(..., separators=(",", ":"))` — the compact form the SDKs
  send. The two code paths share one `_payload_for(...)` so the estimate and
  the request cannot drift.
- `llm/tokenizer.py`: `count_request_tokens(text, *, message_count, model)`
  = tokens(text) + `_TOKENS_PER_MESSAGE * message_count` + `_TOKENS_FOR_REPLY`.
  `count_message_tokens` stays for callers that only have messages
  (`context/builder.py`'s budget planning), with its docstring saying it is a
  lower bound.
- `llm/gateway.py`: `answer` and `decide`/`call_tools` estimate via
  `estimate_payload`. The fake clients in `tests/llm/conftest.py` implement it
  by concatenating content, so engine tests are unaffected.
- Tests: `tests/llm/test_tokenizer.py` — the request count exceeds the
  message count by exactly the tools' and schema's token count for a fixed
  registry; `tests/llm/adapters/test_openai_chat.py` — `estimate_payload` and
  the real payload agree on every field that carries text.
- Commit: `llm: the budget estimate counts the request the adapter sends,
  tools and schema included`.

**K2. Measure the drift and gate it.**

- `scripts/run_turn.py` prints, per model call, `estimated`, `actual`, and the
  delta; the `model_calls` row already carries both.
- `tests/llm/test_estimate_drift.py`, marked `live` (new marker beside
  `postgres`, skipped without `OPENAI_API_KEY`): one planner call and one
  answering call against gpt-4o; assert `abs(estimated - actual) / actual <
  0.05`. Five percent is the documented tiktoken-vs-invoice tolerance for
  tool-bearing requests; the number goes in the test's docstring with the
  measured value on the day it was written.
- Verify live: re-run R1 and T1, then §1.2's `model_calls` query — E4 passes
  its own criterion. Record both deltas in the commit message.
- Commit: `llm: measure estimate drift live and gate it at five percent`.

### Phase L — The planner contract is chosen by a recorded comparison (gap 13)

The shape is the reference project's router comparison: three prompt versions,
six cases held fixed across all three, every prompt-by-case pair actually sent
through the client, one typed result row per pair. Four things are adapted to
this repo and each is a decision, not a convenience: the contracts are Python
constants and the baseline *is* the production one (prompts are diffable code
here, and the winner has to become `PLANNER_CONTRACT` — two copies would
drift); the third variant is not "policy-first" (D5); each pair is run `N`
times, not once; and the write cases accept two first routes, because the
contract itself asks for `list_risks` before `create_risk`.

**L1. The contract is a parameter, defaulting to the production one.**

- `llm/prompts.py::build_planner_messages(..., contract: str = PLANNER_CONTRACT)`;
  the developer block carries `contract`.
- `llm/gateway.py::LLMGateway.planner_contract: str = PLANNER_CONTRACT`, a
  constructor field `decide()` passes through. Nothing in `composition/` sets
  it — production always runs the constant; only the harness builds gateways
  with a candidate.
- Tests: `tests/llm/test_prompts.py` — the developer block is the contract
  given; `tests/llm/test_gateway.py` — a gateway built with another contract
  sends it, and the default sends `PLANNER_CONTRACT` byte-for-byte.
- Commit: `llm: the planner contract is a gateway parameter, defaulting to the
  production one`.

**L2. Three contracts, six cases, eighteen-times-N recorded decisions.**

- `eval/routing_prompts.py`: `RouterPrompt(name, contract)` and three of them.
  `V1_DIRECT = RouterPrompt("v1-direct", PLANNER_CONTRACT)` — imported, never
  copied, so the baseline cannot drift from what production sends.
  `V2_COMPOUND` — v1 plus one sentence in rule 1: *"A question that asks both
  for a field and for the reason behind it ('why … and by how much') is a
  document question: the ERP holds the number, never the explanation, and an
  answer that gives only the number is incomplete."* `V3_EVIDENCE_FIRST` —
  rules 1–2 restructured: first name what kind of fact the reply needs (a
  quoted explanation, a live field, or both), then route — documents whenever
  an explanation is needed, the tool only when nothing has to be quoted. Rules
  3–5 and the three non-negotiables are identical across all three, so the
  diff of v2 and v3 against v1 is a few lines a reviewer can read in the ADR.
- `eval/routing_cases.py`: `RoutingCase(id, actor, request,
  accepted_first_routes: frozenset[tuple[DecisionRoute, str | None]], note)`.
  Six, all from `docs/manual-test.md`, fixed across the three prompts:

  | Case | Actor | Request | Accepted first route(s) | Why it is in the set |
  |---|---|---|---|---|
  | R1 | priya | Why is milestone M2 late and by how much? | `retrieve_project_documents` | the finding. Documents-first is unambiguous in *this* graph: `retrieve_project_documents` is terminal and the composer never sees observations, so tool-then-documents would drop nothing only by luck — and the status report holds both the delay and its cause |
  | T1 | priya | What is the status of milestone M2? | `call_tool:get_project_status` | the live-field control: a candidate that sends everything to documents loses here |
  | R10 | priya | What is the weather forecast in Hanoi next week? | `refuse`, or `retrieve_project_documents` | out of scope; the handbook accepts either the planner's refusal or the similarity gate's |
  | T9/1 | priya | How is the sprint going? | `clarify` | under-specified; no candidate may start guessing |
  | A1 | priya | Record a high severity risk on atlas: hypercare staffing is not confirmed for the M2 cutover. | `request_approval:create_risk`, or `call_tool:list_risks` | the approval path; `list_risks` first is what the contract asks for |
  | A9 | tomas | Record a medium risk on atlas: warehouse depot hardware refresh is unfunded. | same as A1 | tomas cannot write, and the planner must not be the one to say so — preflight refuses it (ADR 0016). A candidate that refuses on the actor's behalf scores a miss, which is the point |

  Two of six are writes, the reference project's proportion, and the reason
  a fix for R1 cannot quietly cost the approval route.
- `eval/routing.py`: `RouterResult(prompt, case_id, repeat, chosen_route,
  chosen_tool, accepted, matched, hallucinated_tool, input_tokens,
  output_tokens, cost_usd, error)` and `run_comparison(prompts, cases,
  gateway_factory, *, repeats) -> ComparisonReport`. Per prompt: one
  `LLMGateway` from `gateway_factory(prompt.contract)`, one `Planner` over it
  with the same offered tools production uses; per (case, repeat): a fresh
  `AgentState`, `Planner.plan` *only* — one model call, nothing executed, the
  question is what the planner chooses first. `matched` is whether
  `(route, tool)` is in the accepted set. `hallucinated_tool` is
  `decision.route == "fail"` with the planner's own "was not offered"
  rationale — `Planner._unreadable` already detects a call to a tool that
  does not exist; the harness records it rather than detecting it a second
  time. A provider error is a row with `error` set, not an aborted report
  (the same rule `eval/retrieval.py` follows). `ComparisonReport.summary()`:
  per prompt, match rate, hallucination count, cost; per (prompt, case), the
  route distribution. Three prompts × six cases × `N` repeats rows.
- `scripts/run_routing_comparison.py --repeats 5 [--prompts v1-direct,v2-compound]`
  → `evidence/routing/routing-comparison-<date>.json` plus a per-prompt table
  on stdout. The call count and the estimated cost print first (3 × 6 × 5 =
  90 planner calls ≈ 2.7k input tokens each).
- Tests: `tests/eval/test_routing.py`, with the scripted tool-calling client
  from `tests/llm/conftest.py` — 3 prompts × 6 cases × 1 repeat produce
  exactly 18 rows, each carrying chosen, accepted, `matched` and
  `hallucinated_tool`; the contract the client received differs per prompt
  (the parameter is wired, not ignored); `call_tool:list_risks` matches the
  write cases and `retrieve_project_documents` does not; a client that raises
  on one pair yields one row with `error` and seventeen without; a call to a
  tool that was not offered sets `hallucinated_tool` and not `matched`; the
  report round-trips through JSON.
- Verify: `uv run pytest tests/eval/test_routing.py -q`; then the script
  against gpt-4o, and the report committed. This is the number the
  walkthrough's 1-of-4 becomes.
- Commit: `eval: three planner contracts, six fixed cases, a recorded
  comparison`.

**L3. ADR 0020 applies the rule and promotes the winner.**

- The selection rule, written *before* the run so the ADR cannot be fitted
  to the numbers: highest overall match rate wins; a tie goes to the fewer
  hallucinated tools, then to the shorter contract; and v1 keeps its place
  unless a candidate beats it on R1 **and** loses on no other case. The rule
  goes into `eval/routing_prompts.py`'s module docstring in L2, i.e. it is in
  the commit *before* the report.
- `docs/adr/0020-the-planner-contract-is-chosen-by-a-recorded-comparison.md`:
  the three contracts by name with v2/v3's diff against v1 quoted; the table
  (match rate, hallucinations, cost per prompt; the route distribution for
  R1, A1 and A9); the rule; the outcome; and what was rejected (D5's list,
  plus "policy-first", by name, with the ADR 0016 reason).
- `llm/prompts.py::PLANNER_CONTRACT` ← the winner's text, if it is not v1.
  `V1_DIRECT` then resolves to the winner automatically, the losers stay in
  `eval/routing_prompts.py` as the record, and re-running the script later is
  the regression gate. If no candidate beats v1, nothing is promoted, the ADR
  records that, and gap 13 stays open with its measured rate — a prompt
  change with no measured effect is not kept (D5).
- Verify live: R1 three times through `scripts/run_turn.py`; the route each
  time goes into `docs/manual-test.md` §6.
- Commit: `llm: promote <winner> as the planner contract (R1 <v1 rate> ->
  <winner rate>, writes unchanged)` — or `docs: ADR 0020 -- no candidate beat
  the baseline`.

### Phase M — Re-walk what changed, and close the log

- Re-run A11 (J2's live check), E4 (K2's), R1 ×3 (L3's), and the fifteen
  §4.7 cells recorded as "reused" — with Phase I in place their traces now
  survive the finishing check. Append rows to `docs/manual-test.md` §6 with
  the new commit hash; the original `638adbc` rows are not edited (the log is
  a record).
- `docs/e2e-code-plan.md` §5: add gaps 11–14 with a one-line "closed by
  gap-plan Phase …" for 11, 12 and 14; 13 is closed or stays listed according
  to ADR 0020's outcome, with the measured rate either way.
- `CLAUDE.md`: Commands block gets `init_postgres.py --test`,
  `run_routing_comparison.py`; Open decisions gains one line under "Eval
  report format": routing joins retrieval as a JSON report under `evidence/`,
  and the planner contract is whichever ADR 0020 names.
- Finishing checks (`compileall`, `pytest -q`, frontend typecheck/test only if
  `ui/` was touched — it should not be), then commit:
  `docs: re-walk the four findings, record the results, list what stays open`.

---

## 4. Definition of done

- [ ] `uv run pytest -q` with the dev Postgres up leaves `SELECT count(*) FROM
      runs` unchanged (Phase I), and `tests/persistence/*` connect only to a
      `_test` database.
- [ ] A11 as `orion.lead`, approved by `sponsor`, replies with the recorded
      risk in ≤ 5 steps; a scripted planner that repeats a successful call ends
      `planner_loop`, never `max_steps_exceeded` (Phase J).
- [ ] `estimated_input_tokens` is within 5 % of `input_tokens` on a live planner
      call and a live answering call, and the live test that asserts it is
      marked and skips without a key (Phase K).
- [ ] `evidence/routing/` holds a comparison report with 3 × 6 × N rows; ADR
      0020 applies the rule written before the run; if a contract was promoted,
      `V1_DIRECT` and `PLANNER_CONTRACT` are the same object and the losers are
      still in `eval/routing_prompts.py` (Phase L).
- [ ] ADR 0018, 0019 and 0020 are in the index; `docs/e2e-code-plan.md` §5 lists gaps
      11–14 with their status; `docs/manual-test.md` §6 has the re-walk rows
      (Phase M).
- [ ] No change to `tools/gateway.py`'s order, `engine/transitions.py`'s
      edges, or `rag/access.py` (D6) — `git diff --stat` on those files is empty
      at the end.

## 5. Order and cost

I → J → K → L → M. I first because every later phase's live proof depends on
the evidence surviving the finishing check. J before K because J's live check
(one approved write) is the cheapest way to confirm the new `model_calls.detail`
field K reads. L last because it spends the most model calls (one comparison ≈ 90
planner calls, a second after promotion if the gate is wanted at once) and its
prompt change is the only change in this plan whose effect is measured rather
than proven — it should land on a repo that is otherwise done.

Estimated model spend: Phase J ≈ 3 turns, K ≈ 4 calls, L ≈ 90–180 planner calls,
M ≈ 25 turns — all gpt-4o at the walkthrough's observed ~2.7k input tokens per
call; well under the walkthrough's own cost.

## 6. Still open after this plan (add to `docs/e2e-code-plan.md` §5)

11. ~~Planner loop after a completed write~~ — closed by Phase J.
12. ~~Budget estimate omits tools and schema~~ — closed by Phase K.
13. **Compound-question routing is model-dependent.** Measured by
    `evidence/routing/`, and ADR 0020 says whether a contract moved the rate;
    a prompt moves a rate, not to certainty. A structural answer — a
    `documents_then_tool` route the planner can name, or a completeness check
    on the reply against the question's clauses — is the follow-up, and it is
    the same residual risk ADR 0014 already records for `think -> answer`.
14. ~~The test suite truncates the evidence store~~ — closed by Phase I.
