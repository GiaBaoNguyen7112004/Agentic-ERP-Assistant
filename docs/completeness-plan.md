# Completeness plan: the planner declares what a reply needs, and the graph holds it to that

**Audience:** the implementing model and the reviewer. Companion to
`docs/gap-plan.md` (Phases I–M, which measured this gap and closed the other
three) and `docs/manual-test.md` §6 (the R1 rows). Same rules: every step names
the files it touches, the tests that prove it, the live verification, and the
commit message; the repo is green after every step.

**Read first:** ADR 0006 (function calling is the only decision channel), ADR
0014 (the residual risk on `think -> answer`), ADR 0019 (a guard in `think`,
not a prompt rule; `tool_choice: "none"` on the wire), ADR 0020 (why no prompt
wording is going to fix this), `engine/transitions.py`'s module docstring (what
the table deliberately leaves out, and why).

---

## 0. What is being fixed, stated precisely

Gap 13 (`docs/e2e-code-plan.md` §5). "Why is milestone M2 late and by how
much?" is two questions: *by how much* is a field the ERP holds
(`get_project_status` → `days_late: 2`), *why* is a sentence only a document
holds (`status-report-2026-09` §2.2: the reconciliation exceptions). ADR 0020
measured gpt-4o choosing the tool for this question 15 times in 15 under every
contract wording tried, then answering "two days late" from the observation
and stopping. The reply is grounded (constraint 3 held every time) and
incomplete, and nothing in the graph can tell.

Three facts about the current graph decide the shape of the fix:

1. **Nothing typed says what a complete reply to *this* question must rest
   on.** `PLANNER_CONTRACT` rules 1 and 2 describe the two kinds of fact in
   prose; the model applies one of them per call; no field records that it
   needed both. So there is nothing for a check to check.
2. **`retrieve_and_answer` composes without the turn's observations.**
   `AnswerComposerPort.answer(question, evidence, memories, history)` -- the
   composer never sees `get_project_status`'s result. The one walkthrough run
   that answered R1 fully (`run-c6c900e1…`, tool then documents) did so because
   the status report happens to state the two-day slip itself. A tool-then-
   documents path is correct only by luck of the corpus, which
   `eval/routing_cases.py`'s R1 note already says.
3. **A planner that elects `answer` is believed.** `think` delivers
   `decision.message` with a `Sources:` trailer of observed ids and no check
   that the text rests on anything -- ADR 0014's named residual risk, and
   `docs/e2e-code-plan.md` §5 gap 1. Gap 13 is one instance of gap 1: the
   model answered from what it had, and what it had was half.

The fix, in one sentence: **before the graph runs, the planner declares -- in a
typed object, through the same function-calling channel it decides with -- what
kinds of fact a complete reply must rest on; a pure function checks the state
against that declaration every time the planner tries to end the turn with an
answer; and an answer that does not meet it is not delivered until the missing
kind has been fetched, at most once per kind, after which it is delivered
marked incomplete.** The model says what "complete" means for this question;
the runtime holds it to its own word; no prose is parsed anywhere.

One more thing, found while planning, fixed first:

15. **`session_turns`' `failure` CHECK was never re-applied.** Phase J added
    `planner_loop` to `FailureMode`; `persistence/schema.py` derives the CHECK
    from the Literal, but only `trace_events_kind_check` is named and
    dropped-and-re-added on every init run (ADR 0014's mechanism). The dev
    database, checked live on 2026-09-12:
    `session_turns_failure_check` = `{context_budget_exceeded, provider_failure,
    insufficient_evidence, tool_failure, max_steps_exceeded, none}` -- no
    `planner_loop`. A `planner_loop` turn inside a browser session loses its
    `session_turns` row silently (`RunOrchestrator` logs and eats a record
    failure, by design: "one line of context lost, not the answer"). The same
    would happen to the failure mode this plan adds. Phase N fixes it before
    anything else lands.

---

## 1. Decisions taken by this plan (do not re-open them while implementing)

- **D1 -- The reply contract is typed state, declared by the model through a
  single offered function.** `ReplyContract(needs: frozenset[ReplyNeed],
  document_query: str | None)` with `ReplyNeed = Literal["document_passage",
  "erp_field"]`, declared by calling `declare_reply_contract` -- one function,
  offered alone, `tool_choice: "required"` -- exactly the shape
  `propose_memories` and `propose_session_summary` already use for a typed,
  non-answer model output. Not to be confused with `PLANNER_CONTRACT` (the
  developer block, ADR 0020): that is what the planner is *told*; this is what
  the planner *declares*, per turn.
  Rejected: a keyword rule over the question ("why" → document) -- not the
  model's declaration, brittle across phrasings, and it would make the check
  wrong exactly when the question is unusual; it stays named in §6 as the
  zero-cost fallback if the declaration proves unreliable live. A `needs`
  field added to every tool's argument schema -- it would land in
  `arguments_summary`, the audit row, and the approver's card, none of which
  are about what the reply needs. Reading a JSON block from the content beside
  a tool call -- a prose channel, which ADR 0006 exists to refuse. A
  `documents_then_tool` route -- a new route, node and edge row for one
  question shape, and it would still need fact 2 fixed to work; fact 2 fixed
  is most of the answer on its own.
- **D2 -- Declared once per turn, by the orchestrator, before the graph runs,
  like recall.** `AgentState.contract` is filled where `memories` and
  `history` are, for the reason their docstrings give: a re-plan in the middle
  of a reason-act cycle must not quietly change what the turn is being held
  to, and the declaration should not be paid for out of the step budget. A
  resumed turn carries the contract it was paused with. A declaration that
  fails or is unreadable costs the turn its check, never its answer -- the
  same never-fail rule recall follows -- and `contract=None` means "unchecked",
  which is what every hand-built state in an engine test and every replay is.
  Rejected: declaring in the first `think` (every scripted planner in nine test
  files would need a `declare`, and a replayed state would force a model call
  nobody scripted); declaring on every plan (a moving target, at a call each).
- **D3 -- The check is a pure function over typed state and reads no prose.**
  `reasoning/completeness.py::assess(contract, state) -> Completeness`:
  `document_passage` is met when `state.evidence` is non-empty; `erp_field`
  when any observation has `status == "ok"`. That is the whole rule. It cannot
  read the reply, the rationale, or the passages' text -- the same discipline
  `decision.py` applies to `rationale` and `events.py` to `detail`. Rejected: a
  second model call grading the first ("is this reply complete?") -- an
  unmeasured judge of an unmeasured answer, and behavior smuggled into a prompt
  where typed state should be (constraint 6); a regex over the reply for each
  clause of the question.
- **D4 -- The check redirects, at most once per need, and never delivers less
  than the planner would have.** When the planner elects `answer` and a need is
  missing: a missing `document_passage` routes `retrieve_project_documents`
  with the declared `document_query`, and the planner's withheld text is kept
  on the state as `draft`; a missing `erp_field` re-asks the planner once with
  `tool_choice: "required"` and `search_project_documents` withheld, so the
  model must pick an ERP tool, a clarification or a refusal. Each need is
  redirected once (`AgentState.redirected_needs`); a need still missing after
  its redirect is delivered as it stands with `failure="incomplete_reply"`, a
  typed mode, and an `error_detail` naming the need. If the redirected search
  finds nothing, the `draft` is delivered, marked -- the user gets exactly the
  "two days late" they get today, plus a trace that says what is missing and
  why. Rejected: failing the turn on an unmet contract (strictly worse for the
  user than today, for a check meant to add to answers); delivering silently
  (the trace would not say, and a reviewer counting incomplete replies would be
  grepping prose).
- **D5 -- Fields first, passages last, because retrieval is where composition
  happens and composition must see the field.** `retrieve_and_answer` keeps
  composing at the end of the turn; it now hands the composer the observations
  too (`AnswerComposerPort.answer(..., observations=())`, a new `observation`
  block in `build_messages`), and the `Sources:` trailer merges the document
  citations with the observed ERP ids -- the two things each existing path
  already appends, now both. So when the planner picks retrieval while an
  `erp_field` need is unmet, `think` applies D4's `erp_field` redirect first,
  and retrieval follows on the next plan. Rejected: teaching
  `retrieve_and_answer` to hand back to `think` for the tool -- the
  `retrieve_project_documents -> think` edge exists, but the turn would then
  end on `think -> answer`, which delivers planner text, not a composed and
  grounding-checked reply; that path would need a second composer call site and
  a retrieval repeat-guard, which `engine/transitions.py` names as an open
  design question this plan does not reopen.
- **D6 -- `tool_choice` is the port's vocabulary: `auto`, `none`, `required`;
  a withheld tool is simply not offered.** ADR 0019's `allow_tools: bool`
  becomes `tool_choice: ToolChoice`, the wire's own three values, and
  `Planner.plan` gains `withhold: frozenset[str]`. Rejected: forcing a *named*
  function (`{"type": "function", "function": {"name": …}}`) -- it would let
  the engine choose which ERP tool answers a question, which is the planner's
  decision to make; the engine only forces *that* one is chosen.
- **D7 -- Nothing in this plan touches the approval flow, the gateway order,
  the access rules, or what `eval/routing.py` measures.** They passed; their
  scope is closed. The comparison harness keeps calling `Planner.plan` alone
  on a fresh state (`contract=None`), so ADR 0020's number is still a routing
  number; the declaration gets its own, separate record (Phase R).

---

## 2. Target layout after this plan

```
src/agentic_erp_assistant/
  state/reply_contract.py    ReplyNeed, ReplyContract (frozen, needs + document_query)
  state/agent_state.py       contract: ReplyContract | None, redirected_needs, draft
                             (all defaulted -- no STATE_VERSION bump, per its docstring)
  state/events.py            EventKind + "contract_declared", "contract_enforced"
  reasoning/decision.py      FailureMode + "incomplete_reply"
  reasoning/completeness.py  Completeness, assess(contract, state), REDIRECT_ORDER
  reasoning/planner.py       plan(state, *, tool_choice="auto", withhold=frozenset())
                             declare(state) -> ReplyContract; DecisionModel.declare
  llm/ports.py               ToolChoice; call_with_tools(..., tool_choice="auto")
  llm/adapters/openai_chat.py  tool_choice passed through as the wire string
  llm/gateway.py             call_tools/decide(tool_choice=); declare(question, history)
  llm/tools.py               DeclareReplyContractArguments, DECLARE_REPLY_CONTRACT_TOOL
                             (offered alone, never in PLANNING_TOOLS)
  llm/prompts.py             DECLARATION_CONTRACT, build_declaration_messages;
                             build_messages(..., observations=()) + observation block;
                             SYSTEM_POLICY rule 8 (an observed value is a live fact)
  engine/ports.py            AnswerComposerPort.answer(..., observations=())
                             ContractDeclarerPort (one method: declare)
  engine/nodes.py            think(): the two redirects; retrieve_and_answer():
                             composer gets observations, sources merged, draft path
  engine/orchestrator.py     declarer: ContractDeclarerPort | None; declared before run
  composition/turn.py        declarer=planner
  persistence/schema.py      session_turns_failure_check / _route_check named, re-applied
  eval/routing_cases.py      RoutingCase.expected_needs
  eval/routing.py            one declaration per (case, repeat), recorded beside the rows
tests/
  reasoning/test_completeness.py   the pure function, every rule
  engine/test_completeness.py      the redirects, through think and the runtime
  live/test_reply_contract.py      R1 declares both; T1 declares the field only
docs/adr/0021-…                    the planner declares what a reply needs
```

---

## 3. Step-by-step

### Phase N -- Prerequisite: `session_turns`' CHECKs survive a vocabulary change (gap 15)

- `persistence/schema.py`: name the two constraints
  (`session_turns_failure_check`, `session_turns_route_check`) and add the same
  `ALTER TABLE … DROP CONSTRAINT IF EXISTS … ; ADD CONSTRAINT …` pair the file
  already has for `trace_events_kind_check`, with the same comment about why
  `CREATE TABLE IF NOT EXISTS` is not enough. Postgres auto-named the existing
  one `session_turns_failure_check`, so the drop matches what is deployed.
- Tests: `tests/persistence/test_schema.py` -- after `init`, a `session_turns`
  row with `failure='planner_loop'` inserts; and against a connection whose
  constraint was first created without it (create the table with the old
  list, then run the statements), it inserts too.
- Verify live: the `pg_constraint` query from §0 before and after
  `uv run python scripts/init_postgres.py`; both databases (`--test` as well).
- Commit: `persistence: re-apply session_turns' CHECKs on every init, as
  trace_events' kind already is`.

### Phase O -- The port can require a tool, and the planner can withhold one

**O1. `tool_choice` replaces `allow_tools`.**

- `llm/ports.py`: `ToolChoice = Literal["auto", "none", "required"]`;
  `ToolCallingClient.call_with_tools(..., tool_choice: ToolChoice = "auto")`.
  The docstring keeps ADR 0019's sentence about `none` and adds `required`'s:
  the model must call *some* offered function; content alone cannot come back.
- `llm/adapters/openai_chat.py`: the payload's `"tool_choice"` is the string
  as given (all three are wire values).
- `llm/gateway.py`: `_allow_tools_kwarg` → `_tool_choice_kwarg` (omitted when
  `"auto"`, for the same fake-compatibility reason); `call_tools`/`decide`
  take `tool_choice`; the telemetry detail suffix becomes
  `(tool_choice=none)` / `(tool_choice=required)`.
- `reasoning/planner.py`: `plan(state, *, tool_choice: ToolChoice = "auto",
  withhold: frozenset[str] = frozenset())`. Offered = `self.tools` minus
  `withhold`; `_by_name` stays over all of them so a call to a withheld tool is
  `_unreadable("… which was withheld for this call")`, not "not offered". A tool
  call back from `tool_choice="none"` is unreadable as today; content back from
  `tool_choice="required"` is unreadable too ("answered in prose after a call
  was required") -- the real provider cannot produce either.
- `engine/nodes.py`: ADR 0019's forced call is `plan(state, tool_choice="none")`.
- Every fake that implements `decide`/`call_with_tools` with `allow_tools`
  (`tests/engine/test_observer.py`, `test_orchestrator.py`,
  `test_runtime_routes.py`, `test_think_after_write.py`,
  `tests/eval/test_routing.py`, `tests/llm/test_gateway.py`,
  `tests/llm/adapters/test_openai_chat.py`,
  `tests/persistence/test_postgres_adapters.py`,
  `tests/reasoning/test_planner.py`) takes `tool_choice` instead. No shim.
- Tests: `tests/llm/adapters/test_openai_chat.py` -- the payload carries each
  of the three; `tests/reasoning/test_planner.py` -- a withheld tool is not in
  the offered list the model saw, a call to it fails as withheld, prose after
  `required` fails as unreadable; `tests/llm/test_gateway.py` -- the detail
  suffix for `required`.
- Commit: `llm: tool_choice is auto, none or required, and the planner can
  withhold a tool`.

### Phase P -- The planner declares a reply contract (fact 1)

**P1. The contract, the function, the prompt.**

- `state/reply_contract.py`: `ReplyNeed`, `ReplyContract` (frozen,
  `extra="forbid"`; `needs: frozenset[ReplyNeed]`; `document_query: str |
  None`, required non-blank iff `"document_passage" in needs`, forbidden
  otherwise -- the same "a field on the route that reads it, rejected
  elsewhere" rule `ReasoningDecision` uses). `EMPTY_CONTRACT =
  ReplyContract(needs=frozenset())`: a declaration of nothing, which is what an
  unreadable one becomes.
- `llm/tools.py`: `DeclareReplyContractArguments(StrictArguments)` --
  `needs: list[ReplyNeed]` and `document_query: str | None`
  (required-and-nullable, strict mode's shape for an optional), each with a
  description that is the whole steering; `DECLARE_REPLY_CONTRACT_TOOL`
  (`mutating=False`). **Not** in `PLANNING_TOOLS` or `DEFAULT_TOOLS` -- offered
  alone, like `PROPOSE_MEMORIES_TOOL`.
- `llm/prompts.py`: `DECLARATION_CONTRACT` -- the developer block: *before
  anything is looked up, say what a complete reply must rest on: a passage from
  a project document (an explanation, a decision, a commitment -- anything that
  must be quoted), a live ERP field (a status, a burn-down, a budget, the open
  risks), both, or neither (a request to record something, a question outside
  the project, one too vague to act on). A question that asks why, or what was
  decided, needs a passage; one that asks for a current number needs a field;
  one that asks both needs both, and a reply carrying one of them is
  incomplete. `document_query` is the words a document would have to contain,
  in the user's own terms; resolve "that milestone" from the history role.*
  `build_declaration_messages(question, history=())` → four blocks: system,
  developer, user, history. No evidence, observation or memory block: nothing
  has run, and memory is not a reason to need less.
- `llm/gateway.py::declare(question, history=()) -> ToolCallResult` =
  `call_tools(build_declaration_messages(...), tools=(DECLARE_REPLY_CONTRACT_TOOL,),
  tool_choice="required")`. The telemetry row reads `chose
  declare_reply_contract (tool_choice=required)` -- distinguishable in
  `model_calls` without a new outcome value.
- `reasoning/planner.py`: `DecisionModel.declare(question, history=())`;
  `Planner.declare(state) -> ReplyContract` validates through
  `DECLARE_REPLY_CONTRACT_TOOL.validate_arguments` and builds the contract.
  Every unreadable shape -- prose, a different function, invalid arguments, a
  `document_query` with no `document_passage` need -- returns `EMPTY_CONTRACT`
  with a `logger.warning`, the proposer's severity: a declaration that cannot
  be read is a turn that is not checked, not a turn that fails. A provider
  exception propagates; the orchestrator handles it (P2).
- Tests: `tests/state/test_reply_contract.py` (the validator, both
  directions); `tests/llm/test_tools.py` (strict-compatible; not offered to
  the planner); `tests/llm/test_prompts.py` (four blocks, the history block
  rendered); `tests/reasoning/test_planner.py` (a scripted `declare` yields the
  contract; each unreadable shape yields `EMPTY_CONTRACT` and logs).

**P2. The orchestrator declares before the graph runs.**

- `state/agent_state.py`: `contract: ReplyContract | None = None`. Docstring
  says what `memories`' does -- filled by the orchestrator before the graph,
  never by a node -- and that `None` is "unchecked": a replay, an evaluation
  case, a state paused before this field existed.
- `state/events.py`: `"contract_declared"`; detail `needs=document_passage,
  erp_field query='why is milestone M2 late'`, or `needs=(none)`, or
  `unreadable: <why>`; `failed: <ExceptionName>` when the call raised.
  `persistence/schema.py`'s `EVENT_KINDS` follows the Literal and the named
  CHECK re-applies itself (Phase N's precedent, already in place for this
  table).
- `engine/ports.py`: `ContractDeclarerPort` (`declare(state) -> ReplyContract`),
  satisfied by `Planner` structurally, declared in `engine/` because the
  orchestrator is the caller.
- `engine/orchestrator.py`: `declarer: ContractDeclarerPort | None = None`,
  the `memory`/`conversation` pattern -- `None` is a complete configuration.
  `_recalled` gains a fourth step after history and memory recall: declare,
  set `contract`, append the event. A raising declarer is caught, logged at
  warning, and the turn runs with `contract=None` and the event's detail
  saying so -- "a worse answer, not no answer", the rule the two recalls
  already follow. `resume` does not declare: the paused state has what it had.
- `composition/turn.py`: `declarer=planner` -- the same `Planner` object the
  runtime plans with; "the planner exposes the contract" is literally true.
- Tests: `tests/engine/test_orchestrator.py` -- the contract lands on the
  state the runtime receives; the event is present; a raising declarer leaves
  `contract=None` and the run still completes; `resume` never calls it.
- Verify live: `uv run python scripts/run_turn.py --actor priya "Why is
  milestone M2 late and by how much?"` -- the trace's first event after the
  recalls is `contract_declared needs=document_passage,erp_field query=…`;
  `model_calls` has one more `routed` row than before, detail `chose
  declare_reply_contract (tool_choice=required)`. Then T1 (`What is the status
  of milestone M2?`) → `needs=erp_field`. Nothing else changes yet.
- `tests/live/test_reply_contract.py` (marked `live`, next to
  `test_estimate_drift.py`, outside `tests/llm/`'s environment-neutralising
  conftest for the reason that file's docstring gives): R1 declares both needs
  and a non-blank query; T1 declares `erp_field` alone. Three repeats each,
  `temperature=0.0`; the docstring records the measured declaration on the day
  it was written.
- Commit: `reasoning: the planner declares what a complete reply must rest on,
  before the graph runs`.

### Phase Q -- The check holds the turn to the contract (facts 2 and 3)

**Q1. The pure function.**

- `reasoning/completeness.py`: `Completeness(satisfied, missing)` (frozen,
  both `frozenset[ReplyNeed]`; `complete` property); `assess(contract:
  ReplyContract | None, state: AgentState) -> Completeness` per D3 --
  `None` → nothing missing. `REDIRECT_ORDER = ("erp_field",
  "document_passage")`: the field first (D5). `next_redirect(completeness,
  redirected: frozenset[ReplyNeed]) -> ReplyNeed | None`: the first need in
  that order that is missing and not yet redirected.
- Tests: `tests/reasoning/test_completeness.py` -- every combination of
  `{needs} × {evidence present/absent} × {ok observation present/absent/failed
  only}`; a mutating `ok` observation satisfies `erp_field` too (a write's
  receipt is a live fact); `None` is complete; the redirect order and the
  once-only rule.

**Q2. `think` redirects.**

- `state/agent_state.py`: `redirected_needs: frozenset[ReplyNeed] =
  frozenset()`; `draft: str | None = None` -- "the reply the planner offered
  and the check withheld; delivered, marked, if the redirect finds nothing".
- `reasoning/decision.py`: `"incomplete_reply"` in `FailureMode`, with the
  docstring saying who assigns it (`think` and `retrieve_and_answer`, never
  `classify_failure`) and why it is not `insufficient_evidence`: evidence
  was not lacking for what was said; something the model itself declared
  necessary was never fetched.
- `state/events.py`: `"contract_enforced"` -- emitted once per redirect, detail
  `erp_field: a call was required, search withheld` or `document_passage:
  answer withheld, searching '<query>'`, and once at delivery when a need
  stays unmet: `unmet after redirect: document_passage`.
- `engine/nodes.py::think`, after ADR 0019's guard and its belt-and-suspenders
  check, before the route branches:

  ```
  gap = assess(state.contract, state)
  need = next_redirect(gap, state.redirected_needs) if decision.route in ("answer", "retrieve_project_documents") else None
  if need == "erp_field":
      state = state.evolve(redirected_needs=state.redirected_needs | {need}, events=+contract_enforced)
      decision = self.planner.plan(state, tool_choice="required", withhold={RETRIEVAL_TOOL})
      -> falls through to the ordinary branches with the new decision
         (call_tool / request_approval / clarify / refuse / fail as the model chose;
          a retrieval cannot come back, it was withheld)
  elif need == "document_passage":            # only reachable with route == "answer"
      return advance(state, "retrieve_project_documents",
                     tool_name=RETRIEVAL_TOOL,
                     tool_arguments={"query": state.contract.document_query},
                     tool_mutating=False,
                     draft=decision.message,
                     redirected_needs=state.redirected_needs | {need},
                     events=+route_selected("retrieve_project_documents: contract needs a document passage; answer withheld")+contract_enforced)
  if decision.route == "answer" and gap.missing:   # every missing need already redirected once
      deliver as today, plus failure="incomplete_reply",
      error_detail=f"contract not met after redirect: {', '.join(sorted(gap.missing))}",
      events=+contract_enforced("unmet after redirect: …")
  ```

  Order matters and is stated in the docstring: ADR 0019's guard sees the
  planner's decision first (a repeat is refused before anything else);
  the `required` redirect is a second planner call inside the same node
  execution, exactly the shape ADR 0019 already established; a planner
  exception on that call is the existing `provider_failure` path. The
  `erp_field` redirect runs for a `retrieve_project_documents` decision too
  (D5): the model may then choose the tool, and retrieval follows on its next
  plan -- or it may choose retrieval again on the next plan without the tool
  ever running, in which case `retrieve_and_answer` marks the composed reply
  incomplete (Q3), since the need was redirected once and stays missing.
- `retrieve_and_answer` sees `draft` only on the path Q3 describes.
- Tests: `tests/engine/test_completeness.py`, in the style of
  `test_think_after_write.py` (a `FakePlanner` with a scripted `plan` that
  records `tool_choice` and `withhold`; a `FakeComposer` that records what it
  was handed; a runtime built from four fakes):
  (a) **R1's real shape**: contract `{both, query}`; plan → `call_tool
  get_project_status`; ok observation; plan → `answer "two days late"` → the
  runtime routes retrieval with the declared query, the composer is handed the
  observation, the reply carries the document citations *and* `milestone-m2`,
  `failure == "none"`, `step_count == 4`, `draft == "two days late"`, events
  contain `contract_declared` and one `contract_enforced`;
  (b) **T1's shape**: `{erp_field}`; tool then answer → delivered, no redirect,
  no `contract_enforced`, `step_count == 3`;
  (c) **answering from history** (ADR 0014's residual): `{erp_field}`; plan →
  `answer` at once → the fake saw a second plan with `tool_choice="required"`
  and `search_project_documents` absent from its offer → it calls the tool →
  ok → answer → delivered complete; `redirected_needs == {"erp_field"}`;
  (d) the forced call chooses `ask_clarification` → the turn ends `clarify`;
  the model's choice is honored, nothing is forced twice;
  (e) **documents-first on a compound question**: `{both}`; plan → retrieval
  → redirected to `required` without search → tool → ok → plan → retrieval
  (the model's own, this time) → composed with both → complete;
  (f) **contract `None`**: every step identical to today -- run one of
  `test_runtime_routes.py`'s existing scenarios through the new code
  unchanged (the existing suite proves this too, since no existing test sets
  a contract);
  (g) a need is redirected once: `{erp_field}`, forced call runs a tool that
  fails → `execute_tool` already ends the turn `tool_failure`, no second
  redirect (assert the fake's call count);
  (h) ADR 0019's forced `none` call returning `answer` is still subject to the
  check: A11's shape with a contract `{erp_field}` delivers (the write's
  receipt is an ok observation), and with `{both}` redirects to retrieval
  after the write -- the two guards compose.

**Q3. Composition sees the field; the draft has a path.**

- `llm/prompts.py`: `build_messages(question, evidence, memories=(),
  history=(), observations=())` → seven blocks, `observation` after
  `evidence` (`_render_observations`, already written; `NO_OBSERVATIONS`
  when empty, so the block shape stays constant). `SYSTEM_POLICY` rule 8: *a
  value in the observation role is what this turn's own ERP call returned:
  state it as the current ERP value, without a tag -- nothing in it is a
  passage, the runtime lists the record it came from beside your citations.
  It is never a substitute for a passage when the question asks for a reason,
  a decision, or anything that must be quoted.* Rule 1 is unchanged: the
  claim that needs a citation still needs one, and `_ungrounded` still
  refuses an answer that cites nothing.
- `engine/ports.py::AnswerComposerPort.answer(..., observations: Sequence[ToolOutcome] = ())`;
  `llm/gateway.py::answer` threads it; the adapter's wire fold already
  relabels `observation` (it does for the planner call).
- `engine/nodes.py::retrieve_and_answer`:
  - the composer gets `state.observations`;
  - the reply's trailer is `citations + _observed_sources(state.observations)`
    (deduped, citations first) -- `parse_citations` in `web/service.py`
    already renders a document tag as a document chip and an ERP id as an
    `erp` chip, so the UI needs nothing;
  - after composing, `assess`: if `erp_field` is missing (declared, redirected
    once in `think`, never observed) the composed reply is delivered with
    `failure="incomplete_reply"` and the `contract_enforced` unmet event;
  - **no passages**: if `"document_passage" in state.redirected_needs` and
    `state.draft is not None` -- this retrieval was the check's own -- deliver
    `_with_sources(state.draft, _observed_sources(...))` on the `answer`
    route (`retrieve_project_documents -> answer` is a declared edge) with
    `failure="incomplete_reply"`, `error_detail="contract needs a document
    passage; the redirected search for '<query>' found none"`, and the
    `evidence_retrieved 0 passage(s)` event as today. A planner-chosen search
    that finds nothing refuses exactly as it does today.
- The estimate: `estimate_extra_tokens` already counts the schema; the new
  block is counted by `count_message_tokens` like the others. Re-run
  `tests/live/test_estimate_drift.py`; the answering-call number moves by the
  block's ~12 tokens and stays inside its 10 %.
- Tests: `tests/llm/test_prompts.py` (seven blocks; the observation block
  renders the outcome line; `NO_OBSERVATIONS` when empty; rule 8 present in
  the policy); `tests/llm/test_gateway.py` (`answer` passes observations
  through to the messages); `tests/engine/test_completeness.py` cases (a)
  and (f) above, plus the draft path: redirected retrieval → 0 passages →
  the draft is delivered with `milestone-m2` and `incomplete_reply`; and a
  planner-chosen retrieval → 0 passages → `refuse`/`insufficient_evidence`
  unchanged; `tests/engine/test_nodes.py`: sources merged and deduped.
- Verify live, each through `scripts/run_turn.py` and then once in the
  browser (`uv run agentic-erp-assistant serve`, as `priya`, watching the
  trace panel):
  - R1 ×3: `contract_declared needs=document_passage,erp_field`,
    `route_selected call_tool: called get_project_status`, `tool_called`,
    `contract_enforced document_passage: answer withheld, searching …`,
    `evidence_retrieved 4 passage(s)`, a reply that names the two days *and*
    the reconciliation exceptions, `Sources:` listing
    `[status-report-2026-09#§2.2]` (at least) and `milestone-m2`,
    `failure=none`, `step_count=4`, `model_calls`: declare, routed ×2,
    answered. Record all three trace ids.
  - T1: `needs=erp_field`, no redirect, `step_count=3`, reply unchanged.
  - T9/1 ("How is the sprint going?"): the planner clarifies as before; the
    contract is declared and never enforced (no `answer` was attempted).
  - R10 (weather): refused as before.
  - A1 as `priya`, approve as `sponsor`: pause, approval card, `create_risk ->
    ok`, the reply names the recorded risk; `step_count` unchanged from
    Phase J's row. A9 as `tomas`: preflight refuses, unchanged. **The
    approval flow is not touched (D7)** -- these two rows are the proof.
  - A11 as `orion.lead`, approve as `sponsor`: Phase J's ≤ 5 steps still
    hold; the repeat guard and the check compose, as test (h) says.
  - Reset `data/erp/project.json` after the writes (`git checkout -- data/erp/project.json`).
- Commit: `engine: an answer is delivered only once the contract the planner
  declared is met, or marked when it cannot be`.

### Phase R -- The record: ADR 0021, the declaration measured, the log closed

**R1. The declaration gets a number of its own.**

- `eval/routing_cases.py`: `RoutingCase.expected_needs: frozenset[ReplyNeed]
  | None` -- `None` means "not scored" (R10, A1, A9: any declaration is fine
  for a refusal or a write). R1 → `{document_passage, erp_field}`; T1 →
  `{erp_field}`; T9/1 → `{erp_field}`.
- `eval/routing.py`: `run_comparison` declares once per `(case, repeat)`
  through `gateway_factory(V1_DIRECT.contract)`'s `declare` -- the declaration
  prompt does not vary by planner contract, so it is not repeated per prompt --
  and records `DeclarationResult(case_id, repeat, declared_needs,
  document_query, expected_needs, matched, cost_usd, error)` rows in a
  `declarations` section of the report; `ComparisonReport.summary()` gains
  `declaration_match_rate`. `matched` is set equality, not superset: T1
  declaring both would send a field question through a search and a compose
  it does not need, and the control has to show that.
- `scripts/run_routing_comparison.py` prints the declaration table after the
  prompt table; the estimated call count printed first includes the extra
  `cases × repeats`.
- Tests: `tests/eval/test_routing.py` -- 6 cases × 1 repeat produce 6
  declaration rows beside the 18 routing rows; `expected_needs=None` rows are
  neither matched nor unmatched (excluded from the rate); the report
  round-trips through JSON with the new section.
- Verify: `uv run python scripts/run_routing_comparison.py --repeats 5` →
  `evidence/routing/routing-comparison-<date>.json`, committed. The routing
  numbers should reproduce ADR 0020's (the planner call is unchanged, and so
  is its prompt); the declaration rate is the new number. If R1's declaration
  rate is below 5/5, the fix is the `DECLARE_REPLY_CONTRACT_TOOL` field
  descriptions or `DECLARATION_CONTRACT`, measured again the same way -- not a
  keyword fallback added quietly.
- Commit: `eval: the comparison records what each case declares it needs`.

**R2. ADR 0021** -- *The planner declares what a reply needs, and the graph
holds it to that.* Context: gap 13 as ADR 0020 left it, and facts 1–3 from §0.
Decision: D1–D6, the pure rule in full, the redirect order, the once-only
bound, the draft path. Consequences: `docs/e2e-code-plan.md` §5 gap 1 is
closed structurally for every turn that declared `erp_field`
(ADR 0014's residual risk); gap 13 is closed with the measured declaration
rate and the three R1 trace ids; one more model call per turn (~1.1k input
tokens; the measured cost from `model_calls` goes in the ADR); the two new
event kinds and the new failure mode. Alternatives: everything D1–D6 rejected,
by name, with the reason. Status of ADR 0014's residual-risk paragraph: not
edited (an ADR is never edited into a different decision); ADR 0021 says what
supersedes that paragraph and ADR 0014's status line gains "residual risk
closed by 0021".

- `docs/adr/README.md`: row 0021.
- `docs/e2e-code-plan.md` §5: gap 1 and gap 13 struck through with "closed by
  ADR 0021, `docs/completeness-plan.md`"; gap 15 added and struck through
  (Phase N).
- `docs/manual-test.md` §6: rows for R1 ×3, T1, T9/1, R10, A1, A9, A11 with
  the new commit hash and trace ids; the earlier rows are not edited. §4.2
  gains R1's new expected trace shape in its "expected" column, dated.
- `CLAUDE.md`: Commands block unchanged; Open decisions -- the memory-topology
  bullet's "the `think -> answer` route has no structural grounding check"
  sentence becomes "the `think -> answer` route is held to the reply contract
  the planner declared (ADR 0021); a turn with no contract is unchecked, and
  says so in its trace"; the eval bullet notes the declaration rate beside the
  routing rate.
- Finishing checks: `uv run python -m compileall -q src`, `uv run python -c
  "import agentic_erp_assistant"`, `uv run pytest -q` (with Postgres and the
  key present, so `postgres` and `live` run rather than skip). `ui/` is not
  touched -- `AnswerEvent.failure` and `TraceRow.kind` are strings on the
  TypeScript side, `FailureBlock` renders any non-`none` mode, and
  `parse_citations` already handles both chip kinds -- so no rebuild; the
  browser check in Q3 is the UI verification.
- Commit: `docs: ADR 0021 -- the planner declares what a reply needs; gap 13
  closed, gap 15 fixed` (ADR + README + e2e-code-plan + CLAUDE.md), then
  `docs: re-walk R1, T1, A1, A9, A11 against the completeness check`
  (manual-test.md).

---

## 4. Definition of done

- [x] `session_turns` accepts every current `FailureMode` and `DecisionRoute`
      on both databases after `init_postgres.py`, and would after the next
      vocabulary change too (Phase N, the `pg_constraint` query in the commit
      `5a0369c` -- caught live twice more afterward, when P2 and Q2 each added
      a new `EventKind` member and `trace_events_kind_check` needed
      re-applying again before the first live run would file; both times
      confirmed and fixed the same way).
- [x] `uv run python scripts/run_turn.py --actor priya "Why is milestone M2
      late and by how much?"` three times: every run declares
      `{document_passage, erp_field}`, calls the tool, is redirected to
      retrieval, and replies with both halves, citing at least one
      `status-report-2026-09` locator and `milestone-m2`; `failure=none`,
      `step_count=4` (Phase Q; the three trace ids -- run-dd74c3b4…,
      run-d57783be…, run-a5fb23ce… -- in ADR 0021, `step_count`/`failure`
      confirmed by direct SQL query against `runs.state`).
- [x] T1, T9/1, R10 unchanged in route and reply; A1, A9, A11 unchanged in
      route, pause shape and step count (D7; Phase Q's rows in
      `docs/manual-test.md` §6).
- [x] A scripted planner that answers a `{erp_field}` question at once is
      re-asked with `tool_choice="required"` and no search, exactly once; one
      that answers a `{document_passage}` question is redirected to retrieval
      exactly once; a redirected search that finds nothing delivers the draft
      marked `incomplete_reply`, never `max_steps_exceeded` and never a bare
      refusal of a question the ERP half-answered (`tests/engine/
      test_completeness.py`).
- [x] Every existing engine test passes unchanged in behavior -- a state with
      no contract is the same graph as before (`test_no_contract_redirects_nothing`,
      and the full suite: 1765 passing, none of the pre-existing tests'
      assertions altered beyond the mechanical signature threading Phase O/Q3
      needed -- `tool_choice=`/`observations=` parameters with defaults).
- [x] `evidence/routing/` holds a report with a `declarations` section, R1's
      declaration rate is 5/5 and T1's is 5/5 (Phase R1,
      `evidence/routing/routing-comparison-2026-09-12-adr0021.json` -- filed
      under its own name, not the date-stamped default, because today's date
      collided with ADR 0020's already-committed file of the same name;
      confirmed byte-identical before and after this run). ADR 0021 is in the
      index; `docs/e2e-code-plan.md` §5 lists 1, 13 and 15 as closed with the
      reference; `docs/manual-test.md` §6 has the re-walk rows.
- [x] `engine/transitions.py`'s edges, `tools/gateway.py`, and `rag/access.py`
      are untouched by this plan (`git diff --stat 4ed77a6 HEAD` -- the commit
      immediately before Phase N -- is empty for all three, checked directly,
      and stated in ADR 0021).

## 5. Order and cost

N → O → P → Q → R. N first because Q adds a failure mode and P adds event
kinds, and the dev database has already shown it keeps the old lists. O before
P because P's declaration needs `required`. P before Q because Q's check reads
what P declares, and P alone is observable live (the event and the
`model_calls` row) without changing any reply -- the safest point to confirm
the declaration is what the model actually says, before anything acts on it.
R last for ADR 0020's reason: the numbers land on a repo that is otherwise
done.

Model spend: one extra call per turn from P onwards (~1.1k input tokens, four
role blocks); R1-shaped turns cost one retrieval and one composition more than
today, which is the price of the missing half. Phase R1's comparison is 90
planner calls (as ADR 0020) plus 30 declarations, paced 4 s apart as before.

## 6. Still open after this plan (add to `docs/e2e-code-plan.md` §5)

1. ~~`think -> answer` has no structural grounding check~~ -- closed by ADR
   0021 for every turn whose contract names a need. A turn that declares
   `needs=()` (a write, a refusal, a clarification) is still delivered on the
   planner's word, which is correct for those shapes and stays model-dependent
   for a question the declaration misjudged as needing nothing.
13. ~~Compound-question routing is model-dependent~~ -- closed by ADR 0021:
    the route is still the model's, the completeness is the graph's.
15. ~~`session_turns`' CHECKs were never re-applied~~ -- closed by Phase N.
16. **The declaration is itself a model output.** Measured (Phase R1), not
    guaranteed. If a live run ever shows a question declared as needing
    nothing when it plainly did, the recorded fallback is a keyword rule
    *widening* the declaration (never narrowing it) -- "why", "reason",
    "decided" add `document_passage` -- applied in `Planner.declare` after the
    model's own, and stated in the `contract_declared` event's detail so the
    trace shows which half came from where. Not added now: it would be
    designed against a failure nobody has seen.
17. **A redirected search that finds nothing delivers the draft, marked.** The
    hand-back-to-`think` alternative (a second search with a better query,
    the planner having seen the empty result) is the retrieval repeat-guard
    question `engine/transitions.py` already names, unchanged by this plan.
