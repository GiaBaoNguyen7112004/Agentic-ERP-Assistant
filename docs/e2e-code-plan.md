# End-to-end code plan: wire the core together and put a streaming React chat on it

**Audience:** the implementing model (Sonnet 5) and the reviewer. Every step below
names the files it touches, the tests that prove it, the verification command, and
the commit message. Steps are ordered so the repo is green after every one of them.

**Read first:** `CLAUDE.md` (hard constraints), `docs/adr/README.md` (the index),
ADR 0004/0005/0006 (gateway order, the graph, the decision channel), ADR 0008
(one access rule), ADR 0011/0014 (memory on both sides of the run). Everything in
this plan is built on those decisions, not around them.

---

## 0. Where the repo stands today (what the investigation found)

**Every layer exists and is tested (1381 tests pass, 51 Postgres-marked skip
without the container), but nothing composes them against a real model.**

| Layer | State | Composed in a real run? |
|---|---|---|
| `engine/` — `WorkflowRuntime`, `GraphNodes`, transition table, step budget, `RunOrchestrator` | complete | only with fakes |
| `reasoning/` — `Planner` (one function call → one typed `ReasoningDecision`) | complete | only with scripted `ToolCallResult`s |
| `llm/` — `LLMGateway` (build → budget → retry → validate → cost), `OpenAIChatClient` over `httpx`, prompts with 7 roles, `GroundedAnswer` schema, tokenizer, pricing | complete, **request/response only — no streaming** | ingest + eval scripts only (embeddings) |
| `tools/` — registry (6 tools, one mutating), gateway (validate → scope → rate-limit → approval → audit → execute → trace), handlers over `MockErp` | complete | demo scripts, scripted |
| `rag/` — manifest, loaders (md/html/pdf/csv), chunking with structural locators, Qdrant dense + from-scratch BM25, RRF, access rule in one function, `RetrievalService.for_context` | complete; evidence run 9/9 | `scripts/run_retrieval_evaluation.py` only |
| `memory/` — pure `policy.decide`, `MemoryService`/`SessionMemory`, `LLMMemoryProposer`, `ConversationMemory` window + promotion | complete | demo script, scripted proposer, hashed embeddings |
| `persistence/` — nine Postgres tables, adapters for trace/pause/audit/memory/conversation | complete | demos |
| `trace/` — `RunRecord`, in-memory stores, `RunTelemetry` (per-run cost sink) | complete | **`RunTelemetry` is never constructed by any script — `model_calls` has never been written in a real run** |
| `web/`, `guardrails/`, any UI | **do not exist** | — |
| `agentic_erp_assistant:main` | prints "Hello" | — |

Facts that shape the plan (each verified in the code):

1. **The engine is synchronous and blocking.** `WorkflowRuntime.run` loops over
   nodes; each node makes blocking HTTP calls. The web layer must run a turn in a
   worker thread and bridge to an async response.
2. **Events are only visible after the run returns.** `AgentState.events` is
   appended by nodes; nothing observes a node finishing. A live trace needs a hook.
3. **`ToolGateway.on_event` exists and is wired nowhere.** Gateway-internal
   events (`retry_scheduled` per attempt, `approval_recorded` beside the audit
   row) never reach a store or a screen.
4. **The answering call is structured output.** `GroundedAnswer` arrives as one
   JSON object, so "stream the answer" means streaming the value of its `answer`
   field out of a JSON document as it arrives — not the raw JSON.
5. **The planner's `think -> answer` route is plain content** from a tool-calling
   call (`ToolCallResult.content`). That is the other text worth streaming.
6. **Ports are per-turn by design.** `RetrievalService.for_context`,
   `MemoryService.for_scope`, `RunTelemetry(trace_id, store)` all bind one actor /
   one run. The composition root therefore builds the port graph **per request**
   over process-wide shared resources (HTTP client, Qdrant client, lexical index,
   ERP store, rate limiter). This is cheap (frozen dataclasses holding references)
   and is exactly what the ADRs ask for.
7. **`GraphNodes.retrieve_and_answer` does not catch retriever exceptions**
   (`engine/nodes.py`, the `self.retriever.search(...)` call). An embeddings 429
   raises through the engine and the run is never filed.
8. **The approver's identity is not recorded.** `pauses` has `decision` and
   `decided_at` but no `decided_by`; the `approval_recorded` event says
   "create_risk approved" and not by whom.
9. **A write pauses for a human before any gateway check runs.** The planner
   routes a mutating tool straight to `request_approval` (ADR 0006), and the
   transition guard forbids `call_tool` for an unapproved write — so the
   gateway's steps 1–4 (tool exists, arguments valid, scope held, budget left)
   run only *after* approval. An actor without `project.risk.write` gets a human
   asked about a call that will be refused whatever the human says, which is the
   exact thing ADR 0004 says must not happen. **Fixed in this plan (Phase B).**
10. **Tools are scope-gated but not project-gated.** `rag/access.py` binds every
    document to the actor's `project_code`; the tool gateway checks only
    `required_scope`, and neither `AgentState` nor `ToolRequest` carries a
    project. The mock ERP holds one project, so it is invisible today and
    becomes a leak the moment a second project's actor exists. **Fixed in this
    plan (Phase B).**
11. **`MockErp` has no lock** and its docstring says to add one when a second
    request can arrive mid-write. The web layer is that moment.
12. **psycopg connections are not for sharing across request threads.** One
    connection per request (opened in the worker thread, closed in `finally`).
13. **`context/history_injection._CITATION_TAG` does not match the locators the
    loaders produce** (`row R-2`, `§3.2 part 1` contain spaces), so an inline tag
    with such a locator survives into the `history` role. Small fix, step A5.
14. **`AgentState` has no `project_code`.** Adding it as a required field means
    a stored v1 state refuses to load — which is what `STATE_VERSION` is for.
    The bump is deliberate and documented (step B3).

---

## 1. Decisions taken by this plan (do not re-open them while implementing)

| # | Decision | Why |
|---|---|---|
| D1 | **Backend: FastAPI + uvicorn**, one worker. SSE hand-written (`text/event-stream`, `event:`/`data:` lines) — no `sse-starlette`. | Matches the Python-only backend, gives typed request models from pydantic (already a dependency) and `/docs`. SSE is one-directional and works with `fetch`, which is all a chat needs. One worker because the rate limiter, the flaky counter and the ERP lock are in-process (ADR 0004 records this limit). |
| D2 | **Frontend: React 18 + TypeScript + Vite** in `ui/` at the repo root; Vitest + Testing Library for tests; no UI kit, no router, no state library (`useReducer`). The built bundle is written to `src/agentic_erp_assistant/web/static/` and served by FastAPI; in development Vite's dev server proxies `/api` to uvicorn. | The brief asks for React. A pure reducer over the SSE event stream is the testable core; everything else is presentation. Vite is the standard React toolchain with the least configuration. The build output is git-ignored; `npm run build` is the frontend "does it build" check CLAUDE.md asks for. |
| D3 | **The engine stays synchronous.** A turn runs in a thread from the default executor; a `TurnStream` bridges thread → `asyncio.Queue` → SSE. | Rewriting the engine async would touch every node and every test for a property the web layer can supply at the edge. |
| D4 | **Streamed tokens are a preview; the final `answer` event is authoritative.** The UI replaces the streamed buffer with `answer.text` when it arrives. | The grounding check (`_ungrounded`) and the `Sources:` trailer run after the full JSON arrives. A citation to a source that was never retrieved turns the streamed text into a refusal, and the UI must show the refusal, not the preview. |
| D5 | **Streaming is bound at the gateway, not passed through ports.** `LLMGateway.stream: AnswerStreamSink | None` (constructor field). `PlannerPort`, `AnswerComposerPort`, `DecisionModel` are unchanged. Memory work gets a second gateway with no sink. | Keeps the engine's ports minimal and the planner/composer signatures untouched; a memory proposal must never stream into the chat. |
| D6 | **Per-request composition** in a new package `composition/` (process resources vs. per-turn assembly). `web/` imports `composition/`; nothing imports `web/`. | Fact 6 above. Also the place the CLI and scripts use, so the wiring is written once. |
| D7 | **The approval gate is shown only calls that would otherwise run.** `ToolGatewayPort` gains `preflight(request) -> ToolOutcome`: the gateway's steps 1–4 (find, validate, scope + project, budget) with nothing counted and nothing executed; `think` calls it before routing to `request_approval`, and a refusal ends the turn there. | Fact 9. ADR 0004's rule — never ask a human about a call policy will refuse — becomes true for writes, not only for rate limits. The transition guard is untouched: a write still cannot enter `call_tool` unapproved. |
| D8 | **Tools are bound to the actor's project, enforced twice like documents are (ADR 0008):** at the store — `MockErp.for_project(code)` is the only view a handler reads through, so records of another project do not exist for it — and at the gateway — a tool whose arguments name a project (`project_argument` declared in the registry) is refused `denied` when that argument is not the request's `project_code`. `project_code` becomes required on `ToolRequest` and on `AgentState`; the ERP fixture gains a second project so the rule has something to refuse. | Fact 10. The same principle as the document filter: decided before the handler sees data, deny by default, one rule per path, a drift test proving both paths agree. |
| D9 | **Engine changes are small and enumerated:** `observer` on `WorkflowRuntime`; `try/except` around the retriever; `decided_by` through `resume_approval`; `preflight` in `think`; `project_code` on the state and the request. Nothing else in `engine/` changes. | Everything else the web needs is composition or reading. |
| D10 | **Users are data** in `data/users.json`: actor, display name, role, `project_code`, `scopes`. No login; the UI switches actor with a `<select>`. `approvals.decide` is a web-layer scope checked by the approval endpoint. | Dev/testing requirement from the brief; scopes stay in one reviewable file beside the manifest. |
| D11 | **Cancellation = the client stops listening.** The turn runs to completion, the trace is filed. A "Stop" button aborts the fetch; nothing in the engine is interrupted. | CLAUDE.md: "the trace still complete when a stream is cancelled". Cooperative cancellation inside the engine is a named follow-up, not v1. |
| D12 | **Read model for the UI is plain SQL** in `persistence/postgres_queries.py`, not new methods on the evidence ports. | The ports are write-side contracts the orchestrator depends on; a listing query is a screen's need and must not widen them. |
| D13 | **Dev toggles live only in `composition/settings.py`**, read from `DEV_*` env vars, logged loudly at startup, and never referenced by engine/tools/rag code. | Rate limits, the flaky tool, the step budget and the window size are otherwise untestable from a browser. |

---

## 2. Target layout after this plan

```
src/agentic_erp_assistant/
  state/agent_state.py        + project_code (required); STATE_VERSION = 2
  state/tool_request.py       + project_code (required)
  engine/ports.py             + ToolGatewayPort.preflight()
  engine/nodes.py             + preflight before request_approval; retriever failure -> routed fail
  engine/workflow.py          + observer field; _observe() after every node; decided_by on resume
  engine/orchestrator.py      + resume(..., decided_by=)
  context/history_injection.py + citation regex accepts locators with spaces
  erp/mock.py                 + threading.Lock; for_project() -> ProjectErp view
  tools/models.py             + ExecutionContext
  tools/handlers.py           + handlers take (arguments, context); read through erp.for_project()
  tools/registry.py           + project_argument per tool; read_rate_limit parameter
  tools/gateway.py            + step 3b project check; preflight(); _checks() shared with execute()
  llm/streaming.py            NEW: AnswerStreamSink protocol, JsonStringFieldExtractor
  llm/ports.py                + on_delta on complete() and call_with_tools()
  llm/adapters/openai_chat.py + _send_stream(); SSE parsing; stream_options usage
  llm/gateway.py              + stream field; on_delta wiring; reset on retry
  persistence/schema.py       + pauses.decided_by; runs.project_code
  persistence/postgres_pause.py  + claim(..., decided_by=)
  persistence/postgres_trace.py  + writes runs.project_code
  persistence/postgres_queries.py NEW: EvidenceQueries (read model)
  trace/memory.py             + InMemoryPauseStore.claim(decided_by=)
  composition/__init__.py     NEW
  composition/settings.py     NEW: Settings.from_env() incl. DEV_* toggles
  composition/users.py        NEW: User, UserDirectory
  composition/resources.py    NEW: AppResources (process-wide)
  composition/turn.py         NEW: build_turn() (per request)
  web/__init__.py             NEW
  web/protocol.py             NEW: SSE event models + encode_sse
  web/stream.py               NEW: TurnStream (thread -> asyncio bridge)
  web/service.py              NEW: ChatService (run_turn, decide_approval)
  web/app.py                  NEW: create_app(), routes, static mount
  web/static/                 build output of ui/ (git-ignored)
  __init__.py                 main() -> `serve`
ui/                           NEW: React + TypeScript + Vite (see §3 Phase G)
data/users.json               NEW
data/erp/project.json         + the Orion project (project, milestone O2, budget)
scripts/run_turn.py           NEW: one turn from the CLI, real model, prints the trace
tests/…                       one test module per new module; drift tests named below
docs/adr/0015-*.md, 0016-*.md, 0017-*.md   NEW
```

---

## 3. Step-by-step

Conventions for every step: run `uv run python -m compileall -q src`,
`uv run python -c "import agentic_erp_assistant"`, `uv run pytest -q` before the
commit; from Phase G also `npm --prefix ui run typecheck && npm --prefix ui test
&& npm --prefix ui run build`; commit on `dev` (or a feature branch) with the
message given; never `--no-verify`.

### Phase A — Engine observability and small safety fixes

#### A1. `WorkflowRuntime.observer` — a live view of each step

**Files:** `engine/workflow.py`, `tests/engine/test_observer.py` (new).

Add to `WorkflowRuntime` (frozen dataclass), after `nodes`:

```python
observer: Callable[[AgentState], object] | None = None
"""Called with the state after every node execution, after the engine's own
bookkeeping, and after the loop guard or an approval changes it. Read-only by
contract: the state is frozen, and the return value is ignored. Never allowed
to fail a run -- an observer that raises is logged and dropped for the rest of
the run, because a screen going away must not end a turn.
"""
```

Implement `_observe(self, state)`: wraps the call in `try/except Exception` →
`logger.warning(...)`. Call it:

- in `run()`, after the `node_exited` event is appended (once per node);
- on the state returned by `_out_of_budget` (before returning it);
- in `resume_approval`, on `decided` (so the `approval_recorded` event is seen
  live) and on the `refuse` state when denied. The approved branch calls `run()`,
  which observes as usual.

**Tests:** an observer receives one state per node execution, in order, and the
last one is the state `run()` returns; a raising observer does not change the
outcome; `resume_approval(denied)` calls the observer with a terminal `refuse`
state; `tests/engine/test_ports.py` (import rule) and `test_transitions.py` (no
`route=` outside `advance`) still pass.

**Commit:** `engine: let a caller observe each step as it happens`

#### A2. A retriever that raises becomes a routed `fail`

**Files:** `engine/nodes.py`, `tests/engine/test_nodes.py`.

In `retrieve_and_answer`, wrap `self.retriever.search(...)` exactly the way the
composer call is wrapped: `except Exception` → `advance(state, "fail",
failure="provider_failure", error_detail=_clip(f"{type(error).__name__}: {error}"),
events=... + _event("retrieve", "failed", f"retriever raised {type(error).__name__}"))`.

**Test:** a fake retriever whose `search` raises `RuntimeError("embeddings down")`
produces a terminal state with `failure == "provider_failure"`, a `failed` event
naming the exception type, and no exception escaping `run()`.

**Commit:** `engine: a retriever that raises ends the turn as a traced failure`

#### A3. Record who decided an approval

**Files:** `engine/workflow.py` (`resume_approval(state, approved, *, decided_by=None)`),
`engine/orchestrator.py` (`resume(trace_id, *, approved, decided_by=None)`),
`trace/ports.py` (`PauseStore.claim(trace_id, *, approved, decided_by=None)`),
`trace/memory.py`, `persistence/postgres_pause.py`, `persistence/schema.py`, tests
in `tests/engine/`, `tests/trace/`, `tests/persistence/`.

- Event detail becomes `f"{tool} {'approved' if approved else 'denied'}"` plus
  `f" by {decided_by}"` when given.
- Schema: append `ALTER TABLE pauses ADD COLUMN IF NOT EXISTS decided_by text;`
  to `_SCHEMA_TEMPLATE` (idempotent, like the constraint re-apply). `claim` sets
  `decided_by = %s` in the same `UPDATE`.
- In-memory pause store keeps `decided_by` beside the decision.

**Commit:** `trace: the pause records who decided it`

#### A4. `MockErp` write lock

**Files:** `erp/mock.py`, `tests/erp/test_mock.py`.

`self._lock = threading.Lock()` in `__init__`; `create_risk` holds it around
append + flush + rollback. Test: two threads calling `create_risk` concurrently on
one store produce distinct ids and the file has both rows.

**Commit:** `erp: serialize writes now that a second request can arrive mid-write`

#### A5. History stripping must match the locators the loaders produce

**Files:** `context/history_injection.py`, `tests/context/test_history_injection.py`.

`_CITATION_TAG = re.compile(r"\[[^\[\]\s]+#[^\[\]\s]+\]")` refuses whitespace,
but real locators contain spaces: `row R-2` (CSV), `§3.2 part 1` (a split
section), see ADR 0009 and `evidence/rag/retrieval-report.json`. An inline
`[risk-register#row R-2]` in a prior reply survives into the `history` role,
which ADR 0014 says must never carry anything citation-shaped. `EvidenceSnippet`
forbids `[`, `]`, `#` and line breaks — not spaces — so the docstring's "the same
shape" claim is false today.

Change the pattern to `r"\[[^\[\]\n#]+#[^\[\]\n]+\]"`; tests with
`[risk-register#row R-2]` and `[status-report-2026-09#§3.2 part 1]` inline.

**Commit:** `context: strip citation tags whose locator has a space, as CSV rows and split sections do`

#### A6. Registry accepts a read rate limit

**Files:** `tools/registry.py`, `tests/tools/test_registry.py`.

`build_default_registry(erp=None, *, read_rate_limit: RateLimitPolicy = DEFAULT_RATE_LIMIT)`;
the `read()` helper passes `rate_limit=read_rate_limit`. Writes keep
`WRITE_RATE_LIMIT`. Test: the parameter reaches every read definition and not the
write.

**Commit:** `tools: let the composition root tighten the read budget`

### Phase B — The two gate fixes (fact 9 and fact 10)

Do B1–B5 in order; the suite is red between B1 and B2 only inside one commit
(B1+B2 land together if you prefer one commit per invariant).

#### B1. `project_code` on the state and the request

**Files:** `state/agent_state.py`, `state/tool_request.py`, every
`AgentState(`/`ToolRequest(` construction (35 + 7 sites; `grep -rn "AgentState(\|ToolRequest("`
— most are in fixtures such as `tests/engine/test_nodes.py::state()` and
`tests/memory/builders.py`), `scripts/demo_*.py`.

```python
# AgentState, after `actor`
project_code: str = Field(min_length=1)
"""The project this turn works on, snapshotted with the actor and the scopes.

Required for the reason `actor` is: every authorization decision -- a document,
a memory, an ERP record -- is "this project and this entitlement", and a turn
that could exist without a project is a turn whose tool calls cannot be bound
to one. The composition root reads it off the user record, next to the scopes.
"""
```

`STATE_VERSION = 2`, with the docstring gaining one line: *v2 added the required
`project_code`; a v1 state refuses to load rather than being read as a turn on
no project.* `ToolRequest.project_code: str = Field(min_length=1)` with the same
argument `scopes` already makes ("required rather than defaulted"). The node
passes `project_code=state.project_code`.

**Tests:** `test_agent_state.py` — blank project rejected, v1 document refuses
to load with the version message; `test_tool_request.py` — required.

**Commit:** `state: a turn and a tool call both name the project they are bound to`

#### B2. The ERP knows its projects; handlers read through a project view

**Files:** `erp/mock.py`, `data/erp/project.json`, `tools/models.py`,
`tools/handlers.py`, `tests/erp/test_mock.py`, `tests/tools/test_handlers.py`
(new if absent).

`erp/mock.py`:

```python
@dataclass(frozen=True)
class ProjectErp:
    """One project's slice of the store -- the only thing a handler may read.

    Records of another project do not exist through this view: `milestone`,
    `sprint`, `budget`, `project` return None for them and `risks_for` returns
    (). The rule is decided here, before a handler sees data, for the reason
    rag/access.py filters before ranking: a check a handler has to remember to
    make is a check one handler forgets.
    """
    store: MockErp
    project_code: str
    def milestone(self, milestone_id) -> Milestone | None   # None unless m.project_id == project_code
    def sprint(...), budget(...), project(...), risks_for(...)  # same rule
    def create_risk(self, *, project_id, title, severity) -> Risk:
        if project_id != self.project_code: raise ErpAccessError(...)   # defence in depth; the gateway refused first
        return self.store.create_risk(project_id=..., title=..., severity=...)

MockErp.for_project(self, project_code: str) -> ProjectErp
```

`data/erp/project.json` gains the Orion programme so the rule has something to
refuse (mirrors `orion-status-report-2026-09.md`): project `orion` / "Orion CRM
migration" (`project-orion`); milestone `O2` "Pipeline migration", due
`2026-10-02`, `on_track`, `days_late 0` (`milestone-o2`); budget `orion`
approved 260000 / spent 118400 / forecast 249000 as of `2026-08-31`
(`budget-orion-q3`). **No Orion risk and no Orion sprint**, so `create_risk` on
Atlas still produces `R-3` next and `list_risks(orion)` returns "No open risks
recorded for orion" (the project record cited).

`tools/models.py`:

```python
@dataclass(frozen=True)
class ExecutionContext:
    """What a handler is told about the call it is running, beyond the arguments."""
    trace_id: str
    actor: str
    project_code: str
```

`tools/handlers.py`: `Handler = Callable[[BaseModel, ExecutionContext], HandlerResult]`;
every handler begins `store = erp.for_project(context.project_code)` and reads
through it; `create_risk` calls `store.create_risk(...)`; `ErpAccessError` is
re-raised as `ToolError` like `ErpNotPersistedError` is. The flaky counter stays
a per-process closure.

**Tests:** `for_project("orion").milestone("M2") is None`;
`for_project("atlas").risks_for("orion") == ()`; a handler asked for `M2` under
`project_code="orion"` raises `ToolError("no milestone 'M2' exists")` (existence
hidden, like a filtered document); `create_risk` through the wrong view raises.

**Commit:** `erp: a project view is the only thing a handler reads through`

#### B3. Gateway: the project check, and `preflight`

**Files:** `tools/registry.py` (`ToolDefinition.project_argument: str | None`,
keyword-only, default `None`; set `"project_id"` on `get_budget_summary`,
`list_risks`, `create_risk`), `tools/gateway.py`, `engine/ports.py`,
`tests/tools/test_gateway.py`, `tests/tools/test_registry.py`, `tests/engine/test_ports.py`.

Refactor `execute` so steps 1–5 live in one private `_checks(request) -> tuple[ToolDefinition, BaseModel] | ToolOutcome`
used by both public methods:

1. find; 2. validate; **3. permission = scope, then project:** if
   `definition.project_argument` is set and
   `getattr(arguments, definition.project_argument) != request.project_code` →
   `denied` with error `f"call names project {other!r}; actor {actor!r} is bound to {request.project_code!r}"`;
   4. budget check; 5. approval → returns the `approval_required` outcome (via
   `_refused`, which writes the gated audit row and emits the hook event) when
   `definition.approval_required and request.approval != "approved"`.

```python
def preflight(self, request: ToolRequest) -> ToolOutcome:
    """Every check that precedes execution, and nothing that is execution.

    Steps 1-5 of `execute`, with the handler never reached and the budget
    never counted (ADR 0004: counted at execution, not at the check). The
    answer a caller wants is the status: `approval_required` means the call
    may be put to a human; anything else is the refusal that human would
    otherwise have been asked to rule on.
    """
```

`preflight` is defined for gated tools only. On a gated tool it returns exactly
what steps 1–5 produce: a refusal (`failed`, `invalid_arguments`, `denied`,
`rate_limited`) or `approval_required` when every check passes. On a tool that
needs no approval there is no honest outcome to return — an `ok` would claim a
call ran, and `approval_required` would claim a gate that does not exist — so it
raises `ValueError("preflight is for calls that will stop for a human; run
ungated tools with execute()")`. The `think` node only calls it on the
`request_approval` route, where the tool is mutating and therefore gated by the
registry invariant (`ToolDefinition.__post_init__`).

`engine/ports.py` — `ToolGatewayPort.preflight(self, request: ToolRequest) -> ToolOutcome`
with the contract above; docstring names the reason (D7). Fakes in
`tests/engine/` gain a `preflight` that returns
`ToolOutcome(tool_name=..., status="approval_required", error="needs a human")`
by default and is scriptable.

**Tests (gateway):** `project_id="orion"` from an `atlas`-bound request →
`denied`, no handler call, audit row `approval=not_required, status=denied`;
`preflight` on a write with a missing scope → `denied`; on a spent budget →
`rate_limited` with `retry_after_seconds`; on a valid write → `approval_required`,
one audit row, hook event `approval_requested`, **limiter not counted** (assert
`limiter.check` afterwards still passes); `preflight` never calls a handler (a
handler that raises `AssertionError` if called); drift test: **for every tool in
the default registry**, a call from project `orion` either is `denied` at the
gateway (tools with `project_argument`) or reads as nonexistent through the view
(tools without) — no tool returns another project's data.

**Commit:** `tools: bind a call to the actor's project, and let the engine ask before it pauses`

#### B4. `think` asks before it pauses

**Files:** `engine/nodes.py`, `tests/engine/test_nodes.py`, `tests/engine/test_runtime_routes.py`.

In `think`, on `decision.route == "request_approval"`:

```python
outcome = self.tools.preflight(ToolRequest(
    trace_id=state.trace_id, tool_name=decision.required_tool,
    arguments=decision.tool_arguments or {}, actor=state.actor,
    project_code=state.project_code, scopes=state.scopes, approval=state.approval))
if outcome.status != "approval_required":
    return advance(state, "fail",
        tool_name=decision.required_tool, tool_arguments=decision.tool_arguments,
        tool_mutating=decision.mutating,
        observations=state.observations + (outcome,),
        failure="tool_failure",
        error_detail=_clip(f"{outcome.tool_name} -> {outcome.status}: {outcome.error or ''}", ERROR_DETAIL_MAX_CHARS),
        events=events + (_event("think", "failed", f"{outcome.tool_name} refused before approval: {outcome.status}"),))
# else: the existing request_approval advance, with observations=state.observations + (outcome,)
```

The `call_tool` route is untouched (`execute` still runs every check at
execution time, so an entitlement revoked between pause and approval is still
refused). `rate_limited` at preflight fails the turn with the wait in
`error_detail` rather than sleeping: a write budget of 5/min firing on a call
that has not been approved yet is a runaway, not a queue.

**Tests:** a scripted `preflight` returning `denied` → terminal `fail`,
`failure == "tool_failure"`, **no pause**, observation recorded, the fake
gateway's `execute` never called; `approval_required` → paused exactly as
before, with the preflight outcome in `observations`; the transition/guard tests
still pass (no new edge).

**Commit:** `engine: a write is put to a human only after the gateway's checks pass`

#### B5. Schema: `runs.project_code`; ADRs 0016 and 0017

**Files:** `persistence/schema.py` (`ALTER TABLE runs ADD COLUMN IF NOT EXISTS project_code text;`),
`persistence/postgres_trace.py` (`save_run` writes it; upsert updates it),
`tests/persistence/test_postgres_adapters.py`, `docs/adr/0016-…`, `docs/adr/0017-…`,
`docs/adr/README.md`.

- **ADR 0016 — The approval gate is shown only calls that would otherwise run.**
  Context: fact 9. Decision: `preflight` on the port, called by `think`.
  Alternatives: route writes through `call_tool` and let the gateway refuse
  (rejected — the transition guard exists to keep an unapproved write out of
  `call_tool`, and weakening it to gain a check is backwards); check scope in
  the planner (rejected — the decision layer does not import the registry, by
  design); check in the web layer before showing the card (rejected — the gate
  would hold only for one caller). Consequences: one extra audit row per write
  (`approval_required` at pause time, then the execution row) — the same shape
  the escalated-read path already produces.
- **ADR 0017 — Tools are bound to the actor's project, at the store and at the
  gateway.** Context: fact 10. Decision: D8. Alternatives: per-handler checks
  (rejected — five copies of one rule); a per-request registry over a
  project-filtered store (rejected — resets the flaky counter and the handler
  bindings per request for no gain); trusting the model's `project_id`
  argument (rejected — the model is not the authority on entitlement).
  Consequences: `STATE_VERSION` 2; a v1 `runs.state`/`pauses.state` refuses
  to load — in development, `TRUNCATE` (see `docs/manual-test.md` §1.3); the
  ERP fixture now has two projects and the handbook's cross-project scenarios.

**Commit:** `docs: ADR 0016 (preflight before the pause) and ADR 0017 (tools bound to the project)`

### Phase C — Streaming in the LLM layer (llm/)

#### C1. `llm/streaming.py` — the sink and the field extractor

**Files:** `llm/streaming.py` (new), `tests/llm/test_streaming.py` (new).

```python
@runtime_checkable
class AnswerStreamSink(Protocol):
    def delta(self, text: str) -> None: ...   # more of the reply, in order
    def reset(self) -> None: ...              # forget everything streamed so far
```

`reset()` exists because the gateway retries a transient failure: a stream that
died at token 40 is replayed from token 0 on the next attempt, and a sink that
was not told would show the text twice.

```python
class JsonStringFieldExtractor:
    """Feed a JSON document a fragment at a time; get back the decoded text of
    one top-level string field as it becomes available.

    Only the string value of `field` at object depth 1 is captured. A key with
    the same name nested inside `citations[...]` is ignored (depth 2+). Handles
    fragments that split anywhere: inside an escape, inside a key, between the
    key and the colon. Escapes are decoded (\n \" \\ \/ \b \f \r \t \uXXXX,
    surrogate pairs included); an incomplete escape is held back until it
    completes.
    """
    def __init__(self, field: str = "answer") -> None: ...
    def feed(self, fragment: str) -> str: ...   # returns newly decoded field text
    @property
    def complete(self) -> bool: ...
```

Implementation guidance (a character state machine): track `depth` on `{`/`[`
and `}`/`]` outside strings; `in_string`, `escape`, `unicode_buffer`; at depth 1
remember the last completed string; when the next non-whitespace char after it
is `:` it was a key; if it equals `field`, capture the next string value,
appending decoded characters to the return buffer until the unescaped closing
quote; afterwards `feed` returns `""` forever.

**Tests:** for a fixed `GroundedAnswer` JSON `S` with escapes and a nested
`"answer"` key inside a citation object, **for every split index k**,
`feed(S[:k]) + feed(S[k:]) == expected`; three-way splits for a shorter sample;
`answer` after `citations`; no `answer` field → `""`; an emoji split between its
two `\u` escapes decodes to one character.

**Commit:** `llm: stream the answer field out of a structured reply as it arrives`

#### C2. Port: optional `on_delta` on both client methods

**Files:** `llm/ports.py`, `tests/test_llm_ports.py`.

```python
def complete(self, messages, *, temperature: float,
             on_delta: Callable[[str], None] | None = None) -> CompletionResponse: ...
def call_with_tools(self, messages, *, tools, temperature: float,
                    on_delta: Callable[[str], None] | None = None) -> ToolCallResult: ...
```

Contract: when `on_delta` is given the adapter **may** stream; it calls
`on_delta` with each fragment of *reply content* (never tool-call arguments) in
order, and still returns the complete, normalized result. An adapter that cannot
stream may ignore `on_delta` — every existing fake stays a valid client. The
gateway passes `on_delta` only when it has a sink.

**Commit:** `llm: the port may be asked to stream, and may decline`

#### C3. Adapter: one streaming path shared by both request shapes

**Files:** `llm/adapters/openai_chat.py`, `tests/llm/adapters/test_openai_chat_stream.py` (new).

Add `_send_stream(payload, on_content) -> tuple[dict, Usage]` that:
1. sends `{**payload, "stream": True, "stream_options": {"include_usage": True}}`
   with `self._http.stream("POST", "/chat/completions", ...)`;
2. on a status ≥ 400: `response.read()` then the existing `_raise_for_status`;
3. reads `iter_lines()`; ignores lines not starting with `data:`; stops at
   `[DONE]`; `json.loads` each chunk (unreadable → `TransientProviderError`);
4. accumulates `content` fragments (calling `on_content` per fragment),
   `tool_calls` by `index` (concatenating `function.arguments`, taking `id` and
   `function.name` when present), `finish_reason`, `model`, and `usage` from the
   final chunk;
5. `httpx.TimeoutException` / `httpx.TransportError` mid-stream →
   `TransientProviderError`;
6. **synthesizes the non-streaming body shape** and returns it through the
   existing readers (`_first_choice`, `_read_text`, `_read_decision`,
   `_read_usage` unchanged):
   ```python
   body = {"model": model, "usage": usage,
           "choices": [{"finish_reason": finish,
                        "message": {"content": "".join(content) or None,
                                    "tool_calls": [...] or None}}]}
   ```

`complete(..., on_delta=None)` / `call_with_tools(..., on_delta=None)` choose
`_send_stream` when `on_delta` is not `None`, else `_send`.

**Tests** (`httpx.MockTransport` returning a `text/event-stream` body): content in
three fragments → `on_delta` ×3 in order, `text` concatenated, `usage` from the
last chunk, `last_usage` set; a tool call streamed as `name` then three
`arguments` fragments → one `ToolCallResult`, `on_delta` never called; no
`usage` chunk → `TransientProviderError`; 429 → `TransientProviderError`, 401 →
`ProviderAuthError`; body carries `stream: true` and `stream_options`.

**Commit:** `llm: stream Chat Completions, and read the stream into the same shapes`

#### C4. Gateway: the sink, the extractor, the reset

**Files:** `llm/gateway.py`, `tests/llm/test_gateway.py`.

```python
stream: AnswerStreamSink | None = None
"""Where reply text goes while it is still arriving. Bound here rather than
passed through the ports: the planner and the composer stay ignorant of
streaming, and a gateway built for memory work is built without one."""
```

`answer()`: with a sink, each attempt gets a fresh
`JsonStringFieldExtractor("answer")`; attempts after the first call
`self.stream.reset()` first; `on_delta` feeds the extractor and forwards
non-empty output to `sink.delta`. `call_tools()`: `on_delta = self.stream.delta`
with the same reset rule. Pass `on_delta=` **only** when a sink is bound.

**Tests:** a recording sink receives the `answer` field text and nothing else; a
client failing on attempt 1 and succeeding on 2 produces exactly one `reset()`
between them; `decide()` streams content for a no-tool reply and nothing for a
tool call; a gateway without a sink never passes `on_delta`.

**Commit:** `llm: the gateway streams to a sink it was built with`

### Phase D — Composition root (`composition/`)

#### D1. Settings and users

**Files:** `composition/__init__.py`, `composition/settings.py`,
`composition/users.py`, `data/users.json`, `.env.example`,
`tests/composition/test_settings.py`, `tests/composition/test_users.py`.

`Settings.from_env()` (frozen dataclass; `load_dotenv(override=False)` once):

| Field | Env | Rule |
|---|---|---|
| `model` | `OPENAI_MODEL` | required (adapter already refuses blank) |
| `context_window` | `OPENAI_CONTEXT_WINDOW` | **required, no default** — same argument as `LLMGateway.context_window`; `.env.example` documents `128000` for gpt-4o |
| `output_reserve` | `OPENAI_OUTPUT_RESERVE` | default 1024 |
| `postgres_url` | `POSTGRES_URL` | via `url_from_environment()` |
| `memory_proposer` | `MEMORY_PROPOSER` | `"on"` (default) / `"off"` |
| `memory_required_scope` | — | `"project.docs.read"` constant, documented |
| `dev_tool_rate_limit` | `DEV_TOOL_RATE_LIMIT` | `"N/S"` → `RateLimitPolicy(max_calls=N, per_seconds=S)` or `None` |
| `dev_flaky_status` | `DEV_FLAKY_STATUS` | `"1"` → offer `get_project_status_flaky` *instead of* `get_project_status` |
| `dev_max_steps` | `DEV_MAX_STEPS` | int or `None` |
| `dev_history_turn_limit` | `DEV_HISTORY_TURN_LIMIT` | int or `None` |
| `users_path` | `USERS_PATH` | default `data/users.json` |

`Settings.log_dev_toggles(logger)` warns one line per active toggle.

`data/users.json`:

```json
{
  "_readme": "Who can use the dev chat, and as what. Synthetic people from the corpus. Scopes are the same vocabulary tools/registry.py and data/documents/manifest.json use; approvals.decide is read only by the web layer.",
  "users": [
    {"actor": "priya",      "display_name": "Priya Raman",         "role": "Delivery lead",            "project_code": "atlas",
     "scopes": ["project.docs.read","project.status.read","project.sprint.read","project.budget.read","project.risk.read","project.risk.write","approvals.decide"]},
    {"actor": "wei",        "display_name": "Wei Chen",            "role": "Finance business partner", "project_code": "atlas",
     "scopes": ["project.docs.read","project.docs.finance.read","project.status.read","project.budget.read","project.risk.read"]},
    {"actor": "tomas",      "display_name": "Tomas Lindqvist",     "role": "Warehouse engineer",       "project_code": "atlas",
     "scopes": ["project.docs.read","project.status.read","project.sprint.read"]},
    {"actor": "sponsor",    "display_name": "Programme sponsor",   "role": "Approver",                 "project_code": "atlas",
     "scopes": ["project.docs.read","project.status.read","approvals.decide"]},
    {"actor": "orion.lead", "display_name": "Orion delivery lead", "role": "Delivery lead (Orion)",    "project_code": "orion",
     "scopes": ["project.docs.read","project.status.read","project.budget.read","project.risk.read","project.risk.write"]},
    {"actor": "guest",      "display_name": "Guest",               "role": "No entitlements",          "project_code": "atlas",
     "scopes": []}
  ]
}
```

`User` is a frozen pydantic model (`extra="forbid"`), `scopes: frozenset[str]`;
`UserDirectory.load(path)`, `.get(actor) -> User` (raises `UnknownUser`),
`.all()`. `User.can_approve` = `"approvals.decide" in scopes`.

**Commit:** `composition: settings from the environment, and the users the dev chat may act as`

#### D2. Process resources

**File:** `composition/resources.py`, `tests/composition/test_resources.py` (fakes only).

```python
@dataclass
class AppResources:
    settings: Settings
    users: UserDirectory
    chat_client: ...          # OpenAIChatClient (satisfies both client protocols)
    embeddings: EmbeddingsPort
    retrieval: RetrievalService       # lexical index built once from Qdrant
    memory_index: MemoryVectorStorePort
    erp: MockErp                      # MockErp.load(), the repo file
    registry: ToolRegistry            # build_default_registry(erp, read_rate_limit=...)
    limiter: RateLimiter              # InMemoryRateLimiter(), shared
    manifest: Mapping[str, ManifestEntry]   # for /api/documents access checks
    connect: Callable[[], psycopg.Connection]
    @classmethod
    def from_env(cls) -> "AppResources": ...
    def close(self) -> None: ...
```

`from_env` fails loudly if Qdrant is down or the collection is empty (log: run
`scripts/ingest_documents.py`).

**Commit:** `composition: the process-wide resources, built once`

#### D3. Per-turn assembly

**File:** `composition/turn.py`, `tests/composition/test_turn.py` (fakes for
every resource; assert the wiring: the memory gateway has no sink; the telemetry
sink carries the trace id; the retriever's context and the state's
`project_code` both come from the same `User`; the ToolGateway got the shared
limiter and the per-turn `on_event`).

```python
def build_turn(resources, *, user: User, session_id: str, trace_id: str,
               connection, stream: TurnStreamLike | None) -> TurnPorts:
    s = resources.settings
    traces = PostgresTraceStore(connection)
    telemetry = RunTelemetry(trace_id=trace_id, store=traces)
    answering = LLMGateway(resources.chat_client, context_window=s.context_window,
                           output_reserve=s.output_reserve, telemetry=telemetry, stream=stream)
    memory_model = LLMGateway(resources.chat_client, context_window=s.context_window,
                              output_reserve=s.output_reserve, telemetry=telemetry)
    planner = Planner(answering, tools=offered_tools(s))
    retriever = resources.retrieval.for_context(
        RetrievalContext.for_actor(user.actor, project_code=user.project_code, scopes=user.scopes))
    gateway = ToolGateway(registry=resources.registry, audit=PostgresAuditLog(connection),
                          limiter=resources.limiter, on_event=stream.trace_event if stream else None)
    runtime = WorkflowRuntime(retriever=retriever, tools=gateway, planner=planner, composer=answering,
                              max_steps=s.dev_max_steps or MAX_STEPS,
                              observer=stream.step if stream else None)
    service = MemoryService(store=PostgresMemoryStore(connection), index=resources.memory_index,
                            embeddings=resources.embeddings, model=s.model,
                            required_scope=s.memory_required_scope,
                            proposer=LLMMemoryProposer(model=memory_model) if s.memory_proposer else None,
                            summary_proposer=LLMSessionSummaryProposer(model=memory_model),
                            audit=PostgresMemoryAudit(connection))
    memory = service.for_scope(MemoryScope.for_actor(user.actor, project_code=user.project_code,
                                                     session_id=session_id, scopes=user.scopes))
    conversation = ConversationMemory(store=PostgresConversationStore(connection), model=s.model,
                                      turn_limit=s.dev_history_turn_limit or HISTORY_TURN_LIMIT)
    orchestrator = RunOrchestrator(runtime, traces, PostgresPauseStore(connection),
                                   memory=memory, conversation=conversation)
    return TurnPorts(orchestrator, connection, EvidenceQueries(connection))

def initial_state(user: User, *, session_id, trace_id, message) -> AgentState:
    return AgentState(request=message, actor=user.actor, project_code=user.project_code,
                      scopes=user.scopes, trace_id=trace_id, session_id=session_id)
```

`TurnStreamLike` is a Protocol declared here (`step`, `trace_event`, `delta`,
`reset`) so composition does not import `web/`.

**Commit:** `composition: assemble the ports for one turn over the shared resources`

#### D4. `scripts/run_turn.py` — prove the wiring before the web exists

`uv run python scripts/run_turn.py --actor priya --session s1 "Why is milestone M2 late?"`
builds `AppResources.from_env()`, opens a connection, `build_turn(...)` with a
console stream (tokens to stdout as they arrive, then the events, the reply,
the pause if any, and the `model_calls` rows). `--approve <trace_id>` /
`--deny <trace_id>` resume a pause with `decided_by=--actor`. Exit code 2 on
`StoreConnectionError`. **Run it for real** and paste the output in the commit.

**Commit:** `scripts: one real turn from the terminal, with the trace beside it`

### Phase E — Read model (`persistence/postgres_queries.py`)

**Files:** `persistence/postgres_queries.py`, `persistence/__init__.py`,
`tests/persistence/test_postgres_queries.py` (marked `postgres`).

`EvidenceQueries(connection)` — read-only, plain SQL:

| Method | Query |
|---|---|
| `pending_approvals(actor=None)` | `pauses` where `status='pending'` (+ `actor`), joined to `session_turns` on `trace_id` for `session_id` |
| `run(trace_id)` | `runs` row + `state` validated through `AgentState` |
| `events(trace_id)` | `trace_events` ordered by `seq` |
| `audit_rows(trace_id)` | by run |
| `model_calls(trace_id)` | rows + totals (calls, tokens, cost, unpriced count) |
| `memory_audit(trace_id)` | by run |
| `memories(actor, project_code, session_id)` | live memories in bounds |
| `sessions(actor)` | distinct `session_id`, first request, last `started_at` |
| `session_turns(session_id, actor)` | ordered by `started_at` |

**Commit:** `persistence: the read model a screen needs, as plain SQL`

### Phase F — Web API (`web/`)

Add dependencies: `uv add fastapi uvicorn`. `httpx` is present for the `TestClient`.

#### F1. Event protocol

**File:** `web/protocol.py`, `tests/web/test_protocol.py`.

Pydantic models (frozen, `extra="forbid"`), all with `type: Literal[...]`:

| `type` | Payload | Emitted when |
|---|---|---|
| `turn_started` | `trace_id, session_id, actor, resumed: bool` | before the engine runs |
| `trace` | `seq: int \| null, node, kind, detail, source: "engine" \| "tool_gateway"` | each new `state.events` entry (seq) or a gateway hook event (seq null) |
| `step` | `route, tool_name, tool_arguments, tool_mutating, approval, step_count, terminal` | after every node (from the observer) |
| `token` | `text` | each streamed fragment (D4: preview) |
| `reset` | — | the gateway retried a streamed call |
| `approval_required` | `trace_id, tool_name, arguments, summary, actor` | the run paused |
| `answer` | `text, route, failure, error_detail, citations: [{source_id, locator, tag, kind: "document" \| "erp"}]` | the run ended (authoritative) |
| `turn_finished` | `outcome, route, failure, step_count, evidence, memories_recalled, history_shown, observations: [{tool, status, attempts}], model_calls: {count, input_tokens, output_tokens, cost_usd, unpriced}` | after the run was filed |
| `error` | `message` | any exception escaping the service (also logged) |

`encode_sse(event) -> bytes` = `f"event: {type}\ndata: {json}\n\n".encode()`.

`ServerEvent = Annotated[Union[...], Field(discriminator="type")]`; export
`EVENT_TYPES: tuple[str, ...]` from the union for the drift test in G6.

Citations for `answer`: split the `Sources: ` trailer that
`engine/nodes._with_sources` wrote on `", "`; an item shaped `[id#locator]` is a
document citation (kept only if `id` is in `state.evidence`), anything else is
an ERP record id (`kind: "erp"`). Locators may contain spaces; do not tokenize on
whitespace. Use `SOURCES_PREFIX` from the node.

#### F2. The bridge

**File:** `web/stream.py`, `tests/web/test_stream.py`.

```python
class TurnStream:
    def __init__(self, loop: asyncio.AbstractEventLoop, *, start_seq: int = 0): ...
    def _put(self, event): self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
    def step(self, state): ...          # trace for state.events[emitted:], then StepOut
    def trace_event(self, event): ...   # gateway hook, seq None
    def delta(self, text): ...
    def reset(self): ...
    def emit(self, event): ...          # service-side events
    def flush_events(self, state): ...  # after handle(): the orchestrator's memory/history events, trace only
    def close(self): self._put(None)
    async def events(self): ...         # until None
```

`start_seq` matters on resume: only events newer than the paused state's are streamed.

#### F3. The service

**File:** `web/service.py`, `tests/web/test_service.py` (fake resources, fake `build_turn`).

`run_turn(actor, session_id, message, stream)` in the worker thread: resolve the
user; `trace_id = f"run-{uuid4().hex}"`; `connection = resources.connect()`
(closed in `finally`, then `stream.close()`); `build_turn`; emit `turn_started`;
`initial_state(...)`; `handle`; `flush_events`; `approval_required` or `answer`;
`turn_finished` with `queries.model_calls(trace_id)`; `except Exception` → `error`
+ `logger.exception`.

`decide_approval(trace_id, approver, approved, stream)`: `approver.can_approve`
else `Forbidden`; `pending = PostgresPauseStore(connection).pending(trace_id)` →
`NotFound` if none; the turn resumes **as the requester** (`users.get(pending.actor)`),
`TurnStream(start_seq=len(pending.events))`; `resume(trace_id, approved=…,
decided_by=approver.actor)`; `ApprovalAlreadySettled` → `Conflict`; same tail.

Validation: strip; empty → 422; > 4000 chars → 422.

#### F4. The app

**File:** `web/app.py`, `tests/web/test_app.py` (TestClient + fake `ChatService`).

`create_app(resources=None, service=None) -> FastAPI`; lifespan builds
`AppResources.from_env()` when none injected.

| Method | Path | Body / query | Response |
|---|---|---|---|
| GET | `/api/health` | — | `{status, model, chunks_indexed, postgres: "ok"\|"down"}` |
| GET | `/api/users` | — | `[{actor, display_name, role, project_code, scopes, can_approve}]` |
| POST | `/api/sessions` | `{actor}` | `{session_id: "sess-<12 hex>"}` |
| GET | `/api/sessions` | `?actor=` | `queries.sessions(actor)` |
| GET | `/api/sessions/{id}/turns` | `?actor=` | `queries.session_turns(...)` |
| POST | `/api/chat` | `{actor, session_id, message}` | **SSE** |
| GET | `/api/approvals` | `?actor=` | `queries.pending_approvals(...)` |
| POST | `/api/approvals/{trace_id}` | `{actor, approved}` | **SSE**; 403 / 404 / 409 as JSON before streaming |
| GET | `/api/runs/{trace_id}` | — | `{run, events, audit_rows, model_calls, memory_audit}` |
| GET | `/api/documents/{document_id}` | `?actor=` | source text (PDF → its `sources/*.md` twin, else 415); **403 unless `is_authorized(entry, RetrievalContext(...))`** |
| GET | `/api/protocol` | — | `{"event_types": EVENT_TYPES}` |
| GET | `/`, `/assets/*` | — | the built React app, if `web/static/index.html` exists; else 503 `{"detail": "UI not built: npm --prefix ui install && npm --prefix ui run build"}` |

SSE headers: `text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`.
The endpoint is `async def`; it creates the `TurnStream`, starts the worker with
`loop.run_in_executor(None, ...)` (not awaited), and returns a `StreamingResponse`
over `stream.events()`. Validation errors are JSON 4xx before the thread starts.
Mount `StaticFiles(html=True)` **last**, so `/api/*` is matched first.

**Tests:** static served when built / 503 when not; `/api/users` lists the
fixture; `/api/chat` with a scripted fake service yields
`turn_started, token×3, answer, turn_finished` in order; blank → 422; unknown
actor → 404; approver without scope → 403; settled pause → 409.

**Commits:** `web: a typed event stream over the engine, thread-bridged to SSE`;
`web: the HTTP surface — chat, approvals, runs, documents`

#### F5. `serve` entry point

`agentic_erp_assistant/__init__.py: main()` → argparse `serve [--host] [--port] [--reload]`
→ `uvicorn.run("agentic_erp_assistant.web.app:create_app", factory=True, workers=1, ...)`.
Logs the dev toggles at startup.

**Commit:** `web: serve from the package entry point`

### Phase G — React UI (`ui/`)

#### G1. Scaffold

```bash
npm create vite@latest ui -- --template react-ts     # or write the files by hand
cd ui && npm install && npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```

Pin: React 18, TypeScript 5, Vite 5, Vitest 2. Node ≥ 20 (v23 is installed).

`ui/vite.config.ts`:

```ts
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/agentic_erp_assistant/web/static", emptyOutDir: true },
  server: { proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } } },
  test: { environment: "jsdom", setupFiles: "./src/setupTests.ts" },
});
```

`ui/package.json` scripts: `dev`, `build` (`tsc -b && vite build`), `preview`,
`test` (`vitest run`), `typecheck` (`tsc --noEmit`). `tsconfig` strict.

`.gitignore`: `ui/node_modules/`, `src/agentic_erp_assistant/web/static/`.
(The built bundle is not committed; a grader without Node can still run every
Python check and `scripts/run_turn.py`. If a no-Node demo is ever needed, commit
the build in a tagged release commit and say so in its message.)

**Commit:** `ui: React + TypeScript + Vite scaffold, building into the package's static dir`

#### G2. Protocol types, SSE reader, API client

**Files:** `ui/src/protocol.ts`, `ui/src/sse.ts`, `ui/src/api.ts`,
`ui/src/__tests__/sse.test.ts`.

`protocol.ts` mirrors `web/protocol.py` one-to-one: a discriminated union
`ServerEvent` on `type` with the nine members and their payloads; `Citation`,
`TraceRow`, `StepInfo`, `ModelCallTotals` etc.

`sse.ts`:

```ts
export async function* readSse(response: Response, signal: AbortSignal): AsyncGenerator<ServerEvent> {
  // reads response.body with a TextDecoder, splits on "\n\n", parses "event:" and "data:" lines,
  // JSON.parses data, yields; stops on signal.abort or stream end. Tolerates a chunk boundary
  // inside a frame (buffer carries over) and comment lines (":").
}
export function postSse(url: string, body: unknown, signal: AbortSignal): Promise<Response>
```

`api.ts`: typed `fetch` helpers for every JSON route (`getUsers`, `createSession`,
`listSessions`, `listTurns`, `listApprovals`, `getRun`, `documentUrl`).

**Tests:** `readSse` over a `ReadableStream` that splits one frame across two
chunks yields the right events in order; aborting mid-stream ends the generator
without throwing.

**Commit:** `ui: the protocol types and an SSE reader over fetch`

#### G3. The turn reducer (the testable core)

**Files:** `ui/src/turnReducer.ts`, `ui/src/__tests__/turnReducer.test.ts`.

```ts
export type TurnView = {
  traceId: string | null; status: "starting" | "running" | "paused" | "done" | "error";
  streamed: string;                       // preview text (D4)
  answer: AnswerEvent | null;             // authoritative once present
  approval: ApprovalRequiredEvent | null;
  events: TraceRow[];                     // trace, both engine and gateway rows
  decisions: StepInfo[];                  // step events where route changed
  summary: TurnFinishedEvent | null;
  error: string | null;
};
export const initialTurn: TurnView;
export function turnReducer(view: TurnView, event: ServerEvent): TurnView;
```

Rules: `token` appends to `streamed`; `reset` clears it; `answer` sets `answer`
and status `done` (the view shows `answer.text`, never `streamed`, once present);
`approval_required` → `paused`; `step` pushes a decision only when `route`
differs from the previous decision's; `turn_finished` sets `summary`;
`error` → status `error`.

**Tests:** the D4 rule (answer replaces the streamed preview); reset; decision
de-duplication; paused then resumed (a second `turn_started` with `resumed:true`
keeps the earlier events); error.

**Commit:** `ui: a pure reducer from server events to what a turn looks like`

#### G4. The stream hook and the app state

**Files:** `ui/src/useTurnStream.ts`, `ui/src/appState.ts`, `ui/src/App.tsx`, `ui/src/main.tsx`.

`useTurnStream()` returns `{ start(url, body), stop(), turn: TurnView, running }`;
holds an `AbortController`; on `start` posts, iterates `readSse`, dispatches
into `turnReducer`; `stop()` aborts (D11: the server keeps going).

`appState.ts`: `useReducer` for `{actor, users, sessionId, sessions, messages: Message[], approvals}`
where a `Message` is either `{role:"user", text}` or `{role:"assistant", turn: TurnView}`.
Persist `actor` and `sessionId` in `localStorage` (guarded by try/catch).

`App.tsx` lays out three columns (`Sidebar`, `ChatPanel`, `TracePanel`), stacked
below 1000px; every gutter ≥ 16px; the whole app usable with a keyboard.

**Commit:** `ui: the stream hook and the application state`

#### G5. Components

**Files under `ui/src/components/`:**

| Component | Renders | Notes |
|---|---|---|
| `ActorSwitcher` | `<select>` of users (display name — role), scope chips, `can_approve` badge | switching actor starts a new session |
| `SessionList` | sessions for the actor; **New chat** | selecting loads `listTurns` into messages as finished `TurnView`s (no trace rows) |
| `ApprovalQueue` | pending pauses (tool, summary, actor, age); Approve/Deny | buttons disabled with hint "switch to an approver" unless `can_approve`; polled after every turn |
| `ChatPanel` → `MessageList`, `AssistantMessage`, `Composer` | bubbles; `<ol aria-live="polite">` | `Composer`: textarea, Enter sends, Shift+Enter newline, Send/Stop |
| `AssistantMessage` | route badge (`documents` / `tool` / `clarify` / `refused` / `failed` / `waiting for approval`), `StreamedText` (preview) or `answer.text`, `CitationChips`, `FailureBlock`, `ApprovalCard` | the D4 replacement is a `useMemo` on `turn.answer ?? turn.streamed` |
| `CitationChips` | `[doc#locator]` chips → `documentUrl(id, actor)` in a new tab; ERP ids as plain chips | |
| `ApprovalCard` | tool, arguments `<dl>`, summary, Approve/Deny → starts a resumed stream into the **same** `TurnView` | |
| `TracePanel` → `TurnTrace`, `EventRow`, `DecisionList`, `TurnSummary` | per turn: heading = `trace_id` + link to `/api/runs/{id}`; event rows coloured by kind (decision blue, tool green, approval amber, memory purple, failure red; gateway rows italic, no seq); decisions with pretty-printed arguments; the `turn_finished` summary | |

**Tests (Testing Library):** `AssistantMessage` shows the streamed text while
running and the authoritative text after `answer`; the approval card's buttons
are disabled for a non-approver; `EventRow` renders a gateway row in italics
with no seq.

**Commit:** `ui: the chat, the approval card, the citation chips and the live trace panel`

#### G6. Drift guard and the frontend check

**Files:** `tests/web/test_protocol_drift.py`, `CLAUDE.md`.

The Python test reads `ui/src/protocol.ts` and asserts every member of
`EVENT_TYPES` appears as a `type: "<name>"` literal, and no other `type:` literal
exists — so a renamed event fails in `pytest`, not in a browser.

**Frontend check (CLAUDE.md, "Verify it builds — backend and frontend"):**

```bash
npm --prefix ui run typecheck && npm --prefix ui test && npm --prefix ui run build
uv run agentic-erp-assistant serve        # then open http://127.0.0.1:8000
```

Load `/`, confirm the console is clean, send "Why is milestone M2 late?" as
`priya`, watch tokens stream and the trace fill, click a citation chip, and
confirm `/api/runs/{trace_id}` matches. Development loop: `npm --prefix ui run dev`
(Vite on 5173, `/api` proxied to uvicorn).

**Commit:** `web: fail the suite when the UI's protocol drifts from the server's`

### Phase H — Docs, evidence, housekeeping

- **CLAUDE.md:** settle "Web layer" (FastAPI + React/Vite) and "Trace persistence"
  (Postgres) in *Open decisions*; record the real frontend commands; add
  `serve`, `scripts/run_turn.py`, the single-worker rule, the `DEV_*` toggles,
  `OPENAI_CONTEXT_WINDOW`, `STATE_VERSION = 2`.
- **`.env.example`:** `OPENAI_CONTEXT_WINDOW=`, `MEMORY_PROPOSER=`, `DEV_*` (commented).
- **ADR 0015 — The web layer is a thin async shell over a synchronous engine;
  streamed text is a preview and the filed trace is the record.** D1–D6, D11.
  Alternatives to name: async engine; WebSocket; sink through the ports;
  streaming the raw JSON; simulated streaming of a finished answer; server-side
  rendering / a Python template UI (rejected: the brief asks for React, and a
  reducer over typed events is testable in a way a template is not).
- **README:** quick start (compose up → init_postgres → ingest → `npm --prefix ui run build` → serve).
- **`docs/manual-test.md`** — walk it once; note results in the commit.

**Commit:** `docs: ADR 0015 for the web layer and streaming; CLAUDE.md settles the stack`

---

## 4. Definition of done for v1

- [ ] `uv run pytest -q` green; `uv run pytest -m postgres` green with the container.
- [ ] `npm --prefix ui run typecheck`, `npm --prefix ui test`, `npm --prefix ui run build` green.
- [ ] `scripts/run_turn.py` prints streamed text, a cited answer, the event list;
      `model_calls` has rows for that trace id.
- [ ] In the browser: a document question streams and ends with clickable
      citations; a tool question shows the decision, the observation and
      `Sources:`; `create_risk` as `priya` pauses with the card, Approve resumes
      the stream and `data/erp/project.json` gains `R-3`; Deny ends in the refusal
      text; a second decision returns 409; **`create_risk` as `tomas` ends as
      `failed`/`denied` with no pause**; **`orion.lead` asking about `M2` gets
      "no milestone 'M2' exists" and about `O2` gets its status**.
- [ ] Every request has a `runs` row (with `project_code`), `trace_events` rows,
      and (when a model was called) `model_calls` rows; every gated call has an
      `audit_rows` row.
- [ ] Switching actor changes what retrieval returns (`wei` sees the budget PDF,
      `priya` does not, `orion.lead` sees Orion and not Atlas, `guest` gets a
      refusal with 0 passages).
- [ ] Killing the tab mid-stream still leaves a complete `runs` row.
- [ ] CLAUDE.md, `.env.example`, ADRs 0015–0017 updated; `docs/manual-test.md`
      walked once with results noted in the commit.

---

## 5. Known gaps after v1 (list them in the defence; do not fix here)

1. **`think -> answer` has no structural grounding check** (ADR 0014 residual
   risk). The `Sources:` trailer of observed ids is evidence of what was read,
   not a check that the text follows from it.
2. **Cancellation is not cooperative**; a stopped stream still spends the
   remaining model calls (D11).
3. **Gateway-internal retry events reach the live stream but not the DB**; the
   outcome's `attempts` count is persisted, the per-attempt `retry_scheduled`
   from the gateway hook is not (node-level `retry_scheduled` for rate limits is).
4. **No `guardrails/` package.** Input checks are the web layer's length/blank
   validation and the planner's `refuse`; output checks are the schema and the
   grounding check. A dedicated layer (injection markers on input, PII on output)
   is a follow-up.
5. **Intent lifecycle** (`start_intent`/`advance_intent`/`close_intent`) is still
   driven by nobody; the open intent is recalled if one exists, none is created.
6. **Single process only** (limiter, flaky counter, ERP lock). The ports exist
   for shared implementations.
7. **`MIN_COSINE_SIMILARITY`** stays provisional at 0.30 (measured gap 0.17–0.52
   in `evidence/rag/retrieval-report.json`).
8. **Pricing table** has one row (`gpt-4o`); any other `OPENAI_MODEL` records
   `cost_usd = NULL`, visibly, never zero.
9. **The ERP's project binding is a view over one file**; a real ERP adapter
   would push the project into its own query, the way Qdrant takes the filter.
   The `ProjectErp` seam is where that adapter plugs in.
10. **A revoked entitlement between pause and approval** is caught by `execute`
    at resume time (every check runs again), but the approver is not warned in
    advance; the queue shows the call as it was preflighted.
