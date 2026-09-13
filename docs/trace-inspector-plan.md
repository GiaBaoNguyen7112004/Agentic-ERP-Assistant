# Trace Inspector Plan — the right panel as an execution inspector

**Status:** plan only; no code changed. Written against commit `3a76312` on
`fpt-bao/refactor-UI`. Every claim below was checked against the source it
cites; nothing here names an event kind, field, endpoint or state attribute
that does not exist today unless it is explicitly marked **NEW**.

The question the redesigned panel must answer:

> For this user query, exactly what did my agent do, what data did each node
> receive, what did it produce, and why did the execution move to the next step?

The one-paragraph answer to "can the current architecture answer it": *mostly
yes, but the answer is scattered*. Almost every payload a developer wants
already exists as typed state (`AgentState.evidence / memories / history /
contract / observations / response …`) and is filed in `runs.state`; the live
stream projects only a thin slice of it (`StepEvent`), and the trace event log
is — by a deliberate design decision — a 280-character line per fact with no
payload. The plan therefore **widens the projection, not the record**: the
stream sends the payloads the observer already holds, keyed to the node
execution that produced them, and the persisted evidence tables stay exactly as
they are. Three things are genuinely not captured anywhere (prompt/response
content of model calls, retrieval scores, per-node timing) and get the smallest
capture that closes each gap.

---

## 1. Current architecture findings

### 1.1 The trace is a typed event log on the state, with no payload by design

- `state/events.py` — `TraceEvent(node: str, kind: EventKind, detail: str ≤ 280)`.
  Frozen, `extra="forbid"`, **no timestamp** (the docstring: ordering is the log's
  position; wall-clock belongs to the run record), **no payload**
  (`EVENT_DETAIL_MAX_CHARS = 280`: "anything larger than a line belongs in the
  artifact the event points at, not in the log entry").
- `EventKind` is a closed set of 18: `node_entered, node_exited, route_selected,
  evidence_retrieved, tool_called, approval_requested, approval_recorded,
  retry_scheduled, rate_limited, memory_recalled, memory_written,
  memory_rejected, history_recalled, history_promoted, contract_declared,
  contract_enforced, failed, run_failed`. It is mirrored into a Postgres CHECK
  constraint (`persistence/schema.py`), so adding a kind is a schema change.
- `AgentState.events: tuple[TraceEvent, ...]` is appended to by:
  | Emitter | `node` value | Kinds |
  |---|---|---|
  | `engine/workflow.py::WorkflowRuntime.run` (lines 236–243) | `state.route or "start"` → `start`, `think`, `retrieve_project_documents`, `call_tool` | `node_entered`, `node_exited` — one pair per node execution, `step_count` incremented alongside `node_entered` |
  | `engine/workflow.py::_out_of_budget` | `engine` | `run_failed` |
  | `engine/workflow.py::resume_approval` | `approval` | `approval_recorded` (`<tool> approved/denied by <actor>`) |
  | `engine/nodes.py::think` | `think` | `route_selected` (`<route>: <rationale or forced note>`), `contract_enforced`, `approval_requested`, `failed` |
  | `engine/nodes.py::retrieve_and_answer` | `retrieve` | `evidence_retrieved` (`N passage(s) for '<query>'`), `contract_enforced`, `failed` |
  | `engine/nodes.py::execute_tool` / `_throttled` | `execute_tool` | `tool_called`/`failed` (`<tool> -> <status>`), `approval_requested`, `rate_limited`, `retry_scheduled` |
  | `engine/orchestrator.py` (before the run) | `history`, `memory`, `contract` | `history_recalled`, `memory_recalled`, `contract_declared` |
  | `engine/orchestrator.py::_consolidated` (after the run) | `history`, `memory` | `history_promoted`, `memory_written` (`<decision> (<reason>)`), `memory_rejected` (`<rejection>: <reason>`) |
  | `tools/gateway.py::ToolGateway._emit` | `tool_gateway` | `tool_called`, `failed`, `retry_scheduled`, `rate_limited`, `approval_requested`, `approval_recorded` — **live only**, via `on_event`; never on the state (ADR 0015 known gap 3) |

  Note the naming split: the engine's `node_entered/node_exited` use **route
  names** (`start`, `think`, `retrieve_project_documents`, `call_tool`) while the
  nodes' own rows use **function names** (`think`, `retrieve`, `execute_tool`).
  The inspector has to map both onto one node label (§4.3).

### 1.2 The payloads live on the state, not in the events

`state/agent_state.py::AgentState` (`STATE_VERSION = 2`) carries, typed and
frozen: `request, actor, project_code, scopes, trace_id, session_id, route,
evidence: tuple[EvidenceSnippet]` (`source_id, locator, text`), `memories:
tuple[MemoryRecord]` (13 fields incl. `kind, key, statement, confidence,
recorded_in_run, recorded_at, supersedes, links`), `history:
tuple[ConversationTurn]` (already clipped and citation-stripped copies —
`context/history_injection.py::_clipped_copy`), `contract: ReplyContract | None`
(`needs`, `document_query`), `redirected_needs`, `draft`, `observations:
tuple[ToolOutcome]` (`tool_name, arguments_summary, status, summary
(unbounded), source_ids, error, attempts, retry_after_seconds`), `tool_name,
tool_arguments, tool_mutating, approval, response, failure, error_detail,
terminal, step_count, retry_count, events`.

Nodes take a state and return a state (`Node = Callable[[AgentState],
AgentState]`); the transition table in `engine/transitions.py` is the only way
the route changes. So "what a node received" **is** the previous state and "what
it produced" **is** the diff between two adjacent states — the engine already
hands both to an observer (`WorkflowRuntime.observer`, called after every node,
after `_out_of_budget`, and inside `resume_approval` once for the recorded
decision and — on a denial — once more for the refusal; an approval then runs
the engine loop, which observes each node as usual).

Two things a developer would want are *not* on the state:
- the planner's `ReasoningDecision` object itself (route, `required_tool`,
  `tool_arguments`, `search_query`, `message`, `rationale`, `confidence`,
  `mutating`). Its effects are on the state (`route`, `tool_name`,
  `tool_arguments`, `tool_mutating`, `response`/`draft`) and its rationale is in
  the `route_selected` detail. `confidence` is always `UNSCORED_CONFIDENCE =
  0.5` (`reasoning/planner.py`) and `required_evidence` is never set — neither is
  worth showing.
- the composer's `GroundedAnswer` (`answer, citations, grounded, confidence,
  refusal_reason`). The answer text and citations become `response` + the
  `Sources:` trailer; `grounded=False`/invented citations become a `failed`
  detail and a refusal (`engine/nodes.py::_ungrounded`).

### 1.3 What is persisted (`persistence/schema.py`, nine tables)

| Table | Written by | Holds |
|---|---|---|
| `runs` | `PostgresTraceStore.save_run` once per segment (handle / resume), upsert | `trace_id, actor, request, outcome, started_at, finished_at, project_code, state jsonb` — **the whole final `AgentState`** |
| `trace_events` | same write, `(trace_id, seq, node, kind, detail)` | the event log, `ON CONFLICT DO NOTHING` |
| `model_calls` | `RunTelemetry → PostgresTraceStore.record_model_call`, immediately per call | `model, outcome, estimated_input_tokens, input_tokens, output_tokens, cost_usd, latency_seconds, attempts, occurred_at, detail` — **no node/step link, no prompt, no response** |
| `audit_rows` | `ToolGateway._write_audit_row` (gated and rate-limited calls only) | `actor, tool_name, arguments_summary, approval, status, source_ids, occurred_at` |
| `memory_audit` | `SessionMemory._audit` during consolidation | `memory_id, kind, decision, rejection, reason, statement_summary` + scope ids — **one row per proposal, rejections included** |
| `pauses`, `session_turns`, `memories`, `intents` | — | not trace data |

The connection is autocommit (`persistence/connection.py`), so `model_calls`
and `memory_audit` rows written mid-turn are readable by the same connection
before the stream's tail is emitted — `web/service.py::_emit_tail` already reads
`model_calls` this way.

### 1.4 What reaches the browser today (`web/protocol.py`, nine SSE events)

| Event | When | Carries |
|---|---|---|
| `turn_started` | before the run | `trace_id, session_id, actor, resumed` |
| `trace` (`TraceRow`) | at each `step()` for the rows added since the last one; immediately for gateway rows | `seq (null for gateway), node, kind, detail, source` |
| `step` (`StepEvent`) | after every observer call | `route, tool_name, tool_arguments, tool_mutating, approval, step_count, terminal` — **no evidence, observations, response, failure, or node name** |
| `token` / `reset` | during the composer call | text preview (D4) |
| `approval_required` | paused | `trace_id, tool_name, arguments, summary, actor` |
| `answer` | terminal | `text, route, failure, error_detail, citations[]` |
| `turn_finished` | after filing | `outcome, route, failure, step_count, evidence: doc ids (deduped), memories_recalled: int, history_shown: int, observations: [{tool, status, attempts}], model_calls: totals only` |
| `error` | any escape | `message` |

Ordering fact that shapes the client model: `web/stream.py::TurnStream.step`
streams `state.events[_emitted:]` **after** the node returns, so a node's own
`node_entered … node_exited` rows arrive *after* anything emitted live during
that node (gateway rows via `trace_event`, tokens, and — after this plan — model
calls and retrieval diagnostics). The tree builder has to attach "mid-node"
arrivals to the span that opens next, or the stream has to stamp them (§7.2).

`web/service.py::_emit_tail` runs after `flush_events(final)` (the
orchestrator's post-run rows), so `memory_*` / `history_promoted` rows precede
`answer` and `turn_finished` on the wire.

`GET /api/runs/{trace_id}` (`web/app.py:289`) returns an **untyped dict**:
`{run: RunRow(state=AgentState), events: TraceEvent[], audit_rows, model_calls:
{records, totals}, memory_audit}`. The UI only links to it
(`ui/src/api.ts::getRun(): Promise<unknown>` is never called).

### 1.5 What the UI does with it

- `ui/src/turnReducer.ts` — `TurnView { traceId, status, streamed, answer,
  approval, events: TraceRow[], decisions: StepEvent[] (steps de-duplicated by
  route), summary, error }`.
- `ui/src/components/TracePanel.tsx` picks the latest assistant message;
  `TurnTrace.tsx` renders four blocks: **Trace** header (badge, `MonoId` trace id
  with copy + link to `/api/runs/…`), **Decisions** (`DecisionList`: route badge →
  tool, arguments `JsonBlock` default-open), **Events** (dense `Table` of
  `EventRow`), and `TurnSummary` (`KeyValueList`). A history-loaded turn
  (`events.length === 0 && status !== 'starting'`) shows an `EmptyState` with the
  run link.
- Shared primitives that survive: `JsonBlock`, `KeyValueList`, `SectionHeading`,
  `MonoId`, `ToneBadge`, `RouteBadge`, `EmptyState`, `ui/collapsible`,
  `lib/traceTone.ts`, `lib/routeBadge.ts`, `lib/format.ts`.
- `docs/manual-test.md` §3 quotes the panel ("headed by the `trace_id`",
  "Decisions sub-list", "Italic, muted rows marked ↳ have no `seq`") and §1's
  SQL queries take `<trace_id>` from the panel. E2 checks "every non-italic row on
  screen has a `seq` and is in `trace_events`".

### 1.6 Diagnostics that are computed and then thrown away

These exist in code, are documented as "for the trace", and never reach it:

| Where | What it computes | Where it is dropped |
|---|---|---|
| `rag/retriever.py::HybridRetriever.search_detailed` | `RetrievalOutcome(hits: FusedHit[] (chunk, fused score, per-list ranks + scores, title, heading_path), best_similarity, dense_candidates, lexical_candidates, gated)` | `search()` projects to `EvidenceSnippet` (the port `engine/ports.py::DocumentRetrieverPort.search` deliberately returns no scores) |
| `memory/service.py::SessionMemory.recall_in_detail` | `MemorySelection(selected, skipped: (record, SkipReason), plan: ContextPlan (included/excluded/used_tokens/budget))` | `recall()` returns `.selected`; the orchestrator's `TurnMemoryPort.recall` is what is called |
| `context/history_injection.py::select_history` | `HistorySelection(selected, dropped, plan)` | `memory/conversation.py::ConversationMemory.recall` returns `.selected` |
| `llm/gateway.py` | the seven role blocks (`build_messages` / `build_planner_messages` / `build_declaration_messages` / memory prompts), the offered `tools`, `tool_choice`, the raw reply (`response["text"]` / `ToolCallResult`) | never recorded; `ModelCallRecord.detail` keeps one line (`chose X (tool_choice=none)`) |
| `llm/gateway.py` retries | per-attempt errors | `retry_with_backoff` is called without `on_retry`; only `attempts` survives (plus the `reset` SSE event for streamed answers) |

---

## 2. Current trace data model (one table)

| Fact | Recorded? | Where | Live to UI? | In `/api/runs`? | Bounded? |
|---|---|---|---|---|---|
| Node execution boundaries + order | yes | `node_entered/node_exited` events, `step_count` | yes (`trace`, `step`) | yes | — |
| Node timing | **no** | — | — | run-level `started_at/finished_at` only | — |
| Planner route + rationale | yes | `route_selected` detail | yes | yes | 280 chars |
| Chosen tool + arguments | yes | `state.tool_name/tool_arguments` | yes (`step`) | yes | — |
| Contract redirect | yes | `contract_enforced` detail, `state.redirected_needs`, `state.draft` | detail only | yes | 280 |
| Evidence snippets (id, locator, text) | yes | `state.evidence` | **no** (ids only at the end) | yes | chunk size |
| Retrieval query | yes | `state.tool_arguments.query`, `evidence_retrieved` detail | yes | yes | — |
| Retrieval scores / ranks / candidates / floor | **no** | computed in `search_detailed`, dropped | — | — | — |
| Tool outcome (status, summary, source_ids, error, attempts, retry_after) | yes | `state.observations` | **no** (tool/status/attempts at the end) | yes | `summary` unbounded |
| Gateway internals (retries, refusals, audit) | live only | `ToolGateway.on_event` | yes (`seq: null`) | `audit_rows` for gated calls | 280 |
| Approval request / decision / who | yes | events, `state.approval`, `pauses.decided_by` | yes | yes | — |
| History shown (clipped turns) | yes | `state.history` | **no** (count only) | yes | 80/160 tokens each |
| Memory recalled (records) | yes | `state.memories` | **no** (count only) | yes | statement ≤ 400 |
| Memory recall skips (out_of_scope / superseded / not_relevant) and budget exclusions | computed, **not recorded** | `recall_in_detail` | — | — | — |
| Memory relevance score | **no** — the vector hit scores are discarded in `SessionMemory._semantic`; only the hit *set* is passed to the selector | — | — | — | — |
| Memory decisions after the turn (write/update/reject, rule, reason, statement) | yes | `memory_audit` rows; `memory_written/rejected` details | detail only | yes (`memory_audit`) | reason ≤ 280, statement ≤ 200 |
| Evicted turns folded into the summary | partly | `history_promoted` count; the summary record's `links` | count only | via `memories` table, not the run report | — |
| Reply contract (needs, query) | yes | `state.contract`, `contract_declared` detail | detail only | yes | — |
| Model call cost/latency/attempts/outcome | yes | `model_calls` | totals only | yes (`records`) | — |
| Model call ↔ node | **no** | order only | — | order only | — |
| Prompt messages, tools offered, tool_choice | **no** | — | — | — | — |
| Raw model reply | **no** | — | — | — | — |
| Final answer, failure, error_detail, citations | yes | `state.response/failure/error_detail` + trailer | yes (`answer`) | yes (text; citations must be parsed) | 280 for error_detail |

---

## 3. Already available vs. missing — and the smallest capture for each gap

**Reuse as-is (UI-only or projection-only work):** node order and boundaries;
route/rationale; tool + arguments; evidence snippets; observations; history;
memories; contract; redirects; approval facts; gateway rows; memory audit rows;
model-call telemetry; answer/failure; `/api/runs` for old turns.

**Gaps, each with the smallest change that closes it:**

| Gap | Smallest change | Layer |
|---|---|---|
| G1. Payloads never leave the server live | Project them in `TurnStream.step` as a per-node **delta** of the state (the observer already holds the state); emit history/memory/contract once at the start | web (projection) |
| G2. Nothing says which node a `step` closed | `StepEvent.node` **NEW**, read by the stream from the `node_entered` row among the rows it just streamed; `null` = engine-level state change | web |
| G3. Mid-node live rows arrive before their node's rows | `TraceRow.step` **NEW** (`int | null`): the ordinal of the node executing when the row was produced = `last_observed.step_count + 1` | web |
| G4. Node duration | `StepEvent.elapsed_ms` **NEW**, measured in `TurnStream` between observer calls (monotonic clock, server-side) | web |
| G5. Model call ↔ node | `LLMGateway.inspector` **NEW** hook (the same shape as `stream`), called from `_record` with the `ModelCallRecord`; the stream folds pending records into the next `step` / the `context` event; the run total still comes from `model_calls` via `turn_finished.model_calls.records` **NEW** | llm + web + composition |
| G6. Prompt / tools / reply content | The same hook, with bounded `ModelRequestSnapshot` / `ModelResponseSnapshot` **NEW** built at the gateway's call sites, **only when `DEV_TRACE_MODEL_IO=1`**; never persisted | llm + composition/settings |
| G7. Retrieval scores | A composition-level wrapper around `HybridRetriever` that calls `search_detailed`, hands the `RetrievalOutcome` to the stream, and returns the snippets — the port is untouched | composition + web |
| G8. Memory decisions' full rows live | `turn_finished.memory_audit` **NEW** from `EvidenceQueries.memory_audit(trace_id)` (already written by then) | web |
| G9. Run start/finish on the wire | `turn_finished.started_at/finished_at` **NEW** from `EvidenceQueries.run(trace_id)` (already filed by then) | web |
| G10. Old turns show nothing | Hydrate a `TurnView` from `/api/runs/{id}` (typed) with a documented best-effort node attribution | ui (+ a typed response model) |
| G11. Recall-side skips/drops | Optional, later: `ConversationMemory.recall_in_detail` + composition wrappers | memory + composition |

Not closed, on purpose: a numeric memory relevance score (not recorded; do not
invent), per-attempt LLM retry errors (only `attempts` exists; an `on_retry`
hook is listed as optional in Phase 5), and evicted-turn contents (the
`session_summary` audit row and its `links` are enough).

---

## 4. Proposed trace UI information architecture

### 4.1 The panel, top to bottom

```
┌ Trace panel ─────────────────────────────────────────────────────────┐
│ [route badge] [status]   run-8cda6c1f…  ⧉                             │  header (no "Trace" section, no JSON link)
│ outcome · route · steps · model calls · tokens · cost · ≈ duration    │  overview strip (the old TurnSummary, moved up)
├──────────────────────────────────────────────────────────────────────┤
│ ▸ Context                                          3 history · 1 mem │  prelude phase
│    request · history (n) · memory (n) · contract · declaration call   │
│ ▾ 1  Planner (think)                              1 LLM call · 812ms │  node execution
│    Input      request, evidence 0, observations 0, contract needs…    │
│    LLM call   gpt-4o · routed · 1.2k/31 tok · $0.00x  [request ▸] [reply ▸]
│    Decision   route_selected  call_tool: called get_project_status    │
│               tool get_project_status  {"milestone_id": "M2"}         │
│    → call_tool                                                        │
│ ▾ 2  Execute tool (call_tool)                     ok · 1 attempt      │
│    Input      get_project_status(milestone_id=M2) · approval n/a      │
│    ↳ gateway  tool_called get_project_status -> ok in 1 attempt        │
│    Result     status ok · summary "M2 (…) is at risk and 2 days late…"│
│               source_ids [milestone-m2]                               │
│    → think    (tool returned ok; back to the planner)                 │
│ ▾ 3  Planner (think)                              1 LLM call · 640ms │
│    …                                                                  │
│    Output     response "…"  failure none                              │
│    → answer   (terminal)                                              │
│ ▸ Consolidation                                   1 written · 2 rejected
│    memory_written write (…)  · memory_rejected instruction_like: …    │
│    audit rows: kind · decision · rejection · reason · statement       │
│    proposal call(s)                                                   │
│ ▸ Answer                                          route answer · none │
│    text · citations · failure/error_detail                            │
│ ▸ Raw events (14)                                                     │  the existing dense table, collapsed
└──────────────────────────────────────────────────────────────────────┘
```

Phases, in order: **Context** (before the first node), **node executions**
(one card per `node_entered … node_exited` span, numbered by `step_count`),
**engine-level rows** rendered as slim cards between them (`approval_recorded`
on resume, `run_failed` from the loop guard), **Consolidation** (rows the
orchestrator appends after the run), **Answer**, **Raw events**.

### 4.2 What each node card shows (data-driven by node kind)

| Node (engine name → label) | Input (from the previous step / context) | Inside (rows + payloads) | Output (delta from `step`) | Transition |
|---|---|---|---|---|
| `start` / `think` → **Planner** | `request`; evidence tags so far; observations so far (summaries); `contract.needs` + `redirected_needs`; history/memory counts | model call(s) (1–3: the plan, a forced re-plan (ADR 0019), a redirect (ADR 0021)); `route_selected`; `contract_enforced`; `approval_requested` + the preflight outcome; `failed` | `route`, `tool_name`/`tool_arguments`/`tool_mutating`, `approval`, `draft`, `redirected_needs`, `response`/`failure`/`error_detail` on terminal | `→ <route>` with the `route_selected` detail as the reason |
| `retrieve_project_documents` → **Retrieve & compose** | the query (`tool_arguments.query`), `draft` if a redirect, `contract` | retrieval diagnostics (query, hits with ranks/scores, floor, candidates, gated); `evidence_retrieved`; the composer call; `failed` / `contract_enforced` | `evidence[]` (the snippets, once), `response`, `failure`, `error_detail` | terminal (`answer` / `refuse` / `fail`) |
| `call_tool` → **Execute tool** | `tool_name(arguments)`, `tool_mutating`, `approval`, `retry_count` | gateway rows (↳); `tool_called`/`failed`/`rate_limited`/`retry_scheduled`/`approval_requested` | the new `ToolOutcome` (status, summary, source_ids, error, attempts, retry_after) | `→ think` / `→ request_approval` / `→ fail`; a throttled retry stays on `call_tool` and appears as the next card with `retry_count + 1` |

Engine-level cards: **Approval** (`approval_recorded` — who decided, what;
then the denied refusal or the resumed `call_tool` card) and **Loop guard**
(`run_failed`).

### 4.3 Node label map (one data-driven map, `ui/src/lib/nodeLabel.ts`)

```
start, think                  → Planner          (think)
retrieve_project_documents    → Retrieve & compose
call_tool                     → Execute tool
approval                      → Approval
engine                        → Loop guard
history / memory / contract   → Context
tool_gateway                  → ↳ gateway   (row prefix, not a node)
```
Inner-row node names (`think`, `retrieve`, `execute_tool`) are never used for
grouping — grouping is by the `node_entered/node_exited` spans (Phase 1) and
then by `StepEvent.node` / `TraceRow.step` (Phase 2). Raw names stay visible in
mono next to the label.

---

## 5. Files likely to change

**UI (`ui/src/`)**
- `turnReducer.ts` — drop `decisions`; keep every `step` (`steps: StepEvent[]`);
  add `context`, `modelCallsPending`, `retrievalPending`; handle the new events.
- `lib/executionTree.ts` **NEW** — pure builder `buildExecutionTree(view):
  ExecutionTree` (phases, node spans, attachment rules, transitions).
- `lib/nodeLabel.ts` **NEW**, `lib/transitionReason.ts` **NEW** (pure).
- `hydrate.ts` **NEW** (Phase 4) — `hydrateTurn(report: RunReport): TurnView`.
- `protocol.ts` — mirror every wire change (drift-tested).
- `api.ts` — `getRun(): Promise<RunReport>` (Phase 4).
- `appState.ts` — optional `inspectedId` (Phase 4).
- `components/TurnTrace.tsx` — rewritten as the inspector; `TracePanel.tsx` unchanged
  except the selected-turn hook (Phase 4).
- `components/DecisionList.tsx` — **deleted**.
- `components/TurnSummary.tsx` — becomes the overview strip (moved, same items +
  duration).
- `components/EventRow.tsx` — unchanged (raw table + E2 in the handbook).
- `components/trace/` **NEW** — `PhaseCard`, `NodeCard`, `ContextBlock`,
  `HistoryBlock`, `MemoryBlock`, `ContractBlock`, `EvidenceBlock`,
  `RetrievalBlock`, `ToolOutcomeBlock`, `ModelCallBlock`, `MemoryAuditBlock`,
  `TransitionRow`, `RawEventsTable`.
- `components/shared/TextBlock.tsx` **NEW** — bounded long-text rendering
  (collapse > N lines, "showing 4,000 of 12,345 chars", copy).
- Tests under `__tests__/` (see §11).

**Web layer (`src/agentic_erp_assistant/web/`)**
- `protocol.py` — `ContextEvent` **NEW**; widened `StepEvent`, `TraceRow`,
  `TurnFinishedEvent`; new `*Out` sub-models; `TextOut` + `clip_text`.
- `stream.py` — `TurnStream`: `_last` state, `context()`, `model_call()`,
  `retrieval()`, delta computation, `elapsed_ms`, `step` stamping, `baseline`.
- `service.py` — `_emit_tail` adds `memory_audit`, `model_calls.records`,
  `started_at/finished_at`; `PendingDecision` carries the paused state as the
  stream baseline.
- `app.py` — Phase 4 only: `GET /api/runs/{id}` returns a typed `RunReportOut`
  (same keys, same content).

**Engine / composition / llm (small, additive)**
- `engine/orchestrator.py` — `RunOrchestrator.on_start` **NEW** (never-fail
  hook called with the state the graph starts from, in `handle` and `resume`).
- `composition/turn.py` — `TurnStreamLike` gains `context`, `model_call`,
  `retrieval`; `InspectedRetriever` **NEW** wrapper; wire `inspector` on both
  gateways; wire `on_start`.
- `composition/settings.py` — `DEV_TRACE_MODEL_IO` **NEW** (default off,
  logged by `log_dev_toggles`); `.env.example`.
- `llm/inspection.py` **NEW** — `ModelRequestSnapshot`, `ModelResponseSnapshot`,
  `ModelCallInspector` Protocol, `snapshot_request()` / `snapshot_response()`
  with bounds.
- `llm/gateway.py` — `inspector` + `inspect_io` fields; `_record` takes
  `request=`/`response=`; `_check_budget` takes `request=`.
- `docs/manual-test.md` §3 wording; `docs/adr/0022-…md` **NEW**; `CLAUDE.md`
  open-decisions bullet for the web layer (the new toggle and the event list).

**Unchanged:** `state/`, `engine/nodes.py`, `engine/workflow.py`,
`engine/transitions.py`, `engine/ports.py`, `trace/`, `persistence/`, `tools/`,
`rag/`, `memory/` (until Phase 5), `llm/telemetry.py`, `llm/ports.py`.

---

## 6. Backend / API changes (wire contract)

All additive. `tests/web/test_protocol_drift.py` only checks the `type` literals,
so the one new event type (`context`) must be added to both `EVENT_TYPES` and
`protocol.ts`; widened fields are checked by the new protocol tests (§11).

### 6.1 Shared sub-models (`web/protocol.py`)

```python
TEXT_MAX_CHARS = 4_000            # wire cap for any free-text payload block

class TextOut(_Event):
    text: str                     # ≤ TEXT_MAX_CHARS, whitespace preserved
    truncated: bool
    chars: int                    # the original length

def clip_text(text: str, limit: int = TEXT_MAX_CHARS) -> TextOut: ...

class HistoryTurnOut(_Event):     # ConversationTurn as recalled (already clipped)
    trace_id: str; request: str; response: str | None
    route: DecisionRoute | None; failure: FailureMode
    tool_name: str | None; approval: ApprovalDecision
    started_at: datetime; finished_at: datetime

class MemoryOut(_Event):          # MemoryRecord, minus nothing that exists
    memory_id: str; kind: MemoryKind; key: str; statement: str
    confidence: float; recorded_in_run: str; recorded_at: datetime
    supersedes: tuple[str, ...]; links: tuple[str, ...]
    required_scope: str; actor: str; project_code: str; session_id: str

class ContractOut(_Event):
    needs: tuple[ReplyNeed, ...]; document_query: str | None

class EvidenceOut(_Event):
    source_id: str; locator: str; tag: str; text: TextOut

class RetrievalHitOut(_Event):    # FusedHit
    chunk_id: str; document_id: str; locator: str; title: str
    score: float; ranks: dict[str, int]; scores: dict[str, float]

class RetrievalOut(_Event):       # RetrievalOutcome + the wrapper's inputs
    query: str; limit: int; hits: tuple[RetrievalHitOut, ...]
    best_similarity: float | None; minimum_similarity: float
    dense_candidates: int; lexical_candidates: int; gated: bool

class ToolOutcomeOut(_Event):     # ToolOutcome, summary bounded for the wire
    tool_name: str; arguments_summary: str; status: ToolStatus
    summary: TextOut; source_ids: tuple[str, ...]; error: str | None
    attempts: int; retry_after_seconds: float | None

class MessageOut(_Event):
    role: str; content: TextOut   # the port's Role, before adapter folding

class ModelRequestOut(_Event):
    kind: Literal["answer", "tools"]      # structured answer vs function calling
    messages: tuple[MessageOut, ...]
    tools: tuple[str, ...]                # names only; schemas are static code
    tool_choice: str | None               # "auto" | "none" | "required"; None for answer()
    temperature: float

class ModelResponseOut(_Event):
    content: TextOut | None               # the answer JSON text, or prose content
    tool_name: str | None; arguments: dict[str, object] | None
    stop_reason: str | None               # answer() only; None for tool calls

class ModelCallOut(_Event):               # ModelCallRecord (+ optional I/O)
    model: str; outcome: str; estimated_input_tokens: int
    input_tokens: int; output_tokens: int; cost_usd: float | None
    latency_seconds: float; attempts: int; occurred_at: datetime
    detail: str | None
    request: ModelRequestOut | None       # only with DEV_TRACE_MODEL_IO=1
    response: ModelResponseOut | None     # only with DEV_TRACE_MODEL_IO=1

class MemoryAuditOut(_Event):             # MemoryAuditRow
    occurred_at: datetime; memory_id: str; kind: MemoryKind
    decision: MemoryDecisionKind; rejection: RejectionReason | None
    reason: str; statement_summary: str
```

### 6.2 Events

```python
class ContextEvent(_Event):               # NEW — once per stream, before the first node
    type: Literal["context"] = "context"
    request: str
    history: tuple[HistoryTurnOut, ...]   # state.history
    memories: tuple[MemoryOut, ...]       # state.memories
    contract: ContractOut | None          # None = "unchecked" (ADR 0021)
    model_calls: tuple[ModelCallOut, ...] # the declaration call, when seen live
    resumed: bool

class TraceRow(_Event):                   # + one field
    step: int | None                      # NEW: node ordinal executing when a live
                                          # row was produced; None for rows streamed
                                          # from state.events (their span is explicit)

class StepEvent(_Event):                  # + the delta
    node: str | None                      # NEW: node_entered.node of the span this
                                          # step closes; None = engine-level change
    elapsed_ms: float                     # NEW: since the previous observer call
                                          # (or since context() for the first)
    evidence: tuple[EvidenceOut, ...] | None   # NEW: only when state.evidence changed
    observations: tuple[ToolOutcomeOut, ...]   # NEW: new since the previous state
    response: str | None                  # NEW: set on the step that produced it
    failure: FailureMode; error_detail: str | None   # NEW
    draft: str | None; redirected_needs: tuple[ReplyNeed, ...]; retry_count: int   # NEW
    retrieval: RetrievalOut | None        # NEW: from the retriever wrapper
    model_calls: tuple[ModelCallOut, ...] # NEW: calls made during this node

class ModelCallTotalsOut(_Event):         # + records
    records: tuple[ModelCallOut, ...]     # NEW: every row of model_calls, in order
                                          # (I/O never present here — it is not stored)

class TurnFinishedEvent(_Event):          # + three fields
    memory_audit: tuple[MemoryAuditOut, ...]   # NEW
    started_at: datetime; finished_at: datetime  # NEW, from the runs row
```

`EVENT_TYPES` gains `"context"` (ten members). Nothing is removed or renamed.

### 6.3 `GET /api/runs/{trace_id}` (Phase 4)

Same keys, same content, now a pydantic `RunReportOut` in `web/protocol.py`
(`run: {trace_id, actor, project_code, outcome, started_at, finished_at,
state: AgentState}`, `events`, `audit_rows`, `model_calls: {records, totals}`,
`memory_audit`) so `ui/src/protocol.ts::RunReport` has a Python model to be
drift-tested against (a field-name test, §11). Add `answer: {text, citations}`
via the existing `parse_citations(state.response, state)` so a hydrated turn can
render chips without a TypeScript re-implementation of the trailer parser.

---

## 7. Instrumentation changes (what is genuinely new capture)

### 7.1 `RunOrchestrator.on_start` (engine, 12 lines)

```python
on_start: Callable[[AgentState], object] | None = None
```
Called in `handle()` immediately before `self.runtime.run(...)` with the
declared state (history, memories, contract attached), and in `resume()`
immediately before `self.runtime.resume_approval(...)` with the claimed paused
state. Same never-fail discipline as `WorkflowRuntime._observe`: an observer that
raises is logged and dropped. Why the orchestrator and not a wrapper: it is the
one place that knows "the state the graph will start from"; the engine's
observer fires only *after* the first node, and the declaration call must be
attributed to the prelude, not to node 1.

### 7.2 `TurnStream` (web)

- `__init__(loop, *, start_seq=0, baseline: AgentState | None = None)` —
  `baseline` is the paused state on a resumed stream (`PendingDecision` gains
  `paused: AgentState`; `events_already_seen` becomes `len(paused.events)`).
  `_last: AgentState | None = baseline` is the delta baseline.
- `context(state)` — emits `ContextEvent` from `state.history/memories/contract`
  plus every pending model call (the declaration). Sets `_last = state` **without
  streaming its events** (the orchestrator's `history_recalled` / `memory_recalled`
  / `contract_declared` rows are streamed by the first `step()` exactly as today;
  `_emitted` is untouched).
- `model_call(record, request=None, response=None)` — appends a `ModelCallOut`
  to `_pending_calls` (worker thread; consumed on the same thread).
- `retrieval(query, limit, outcome, minimum_similarity)` — stores
  `_pending_retrieval` as a `RetrievalOut`.
- `trace_event(event)` — stamps `step = (_last.step_count if _last else 0) + 1`.
- `step(state)` — streams new rows (unchanged), then builds the `StepEvent`
  with: `node` = the `node` of the last `node_entered` among the new rows (or
  `None`), `elapsed_ms` from `time.perf_counter()` deltas, the state delta
  against `_last` (`evidence` when the tuple differs, `observations[len(last):]`,
  `response` when it went from `None` to set, `failure`/`error_detail`/`draft`/
  `redirected_needs`/`retry_count` as they stand), `retrieval` and `model_calls`
  from the pending slots (cleared). Then `_last = state`.
  If `context()` was never called (a caller without `on_start`), the first
  `step()` emits the `ContextEvent` itself first, with no model calls attributed.
- `flush_events` unchanged.

### 7.3 `LLMGateway.inspector` (llm) + `llm/inspection.py` **NEW**

```python
@dataclass(frozen=True)
class ModelRequestSnapshot:
    kind: Literal["answer", "tools"]; messages: tuple[tuple[str, str], ...]  # (role, content ≤ cap)
    message_chars: tuple[int, ...]; tools: tuple[str, ...]; tool_choice: str | None
    temperature: float

@dataclass(frozen=True)
class ModelResponseSnapshot:
    content: str | None; content_chars: int | None
    tool_name: str | None; arguments: Mapping[str, Any] | None; stop_reason: str | None

class ModelCallInspector(Protocol):
    def model_call(self, record: ModelCallRecord, request: ModelRequestSnapshot | None,
                   response: ModelResponseSnapshot | None) -> None: ...   # must not raise
```

`LLMGateway` gains `inspector: ModelCallInspector | None = None` and
`inspect_io: bool = False`. `_record(..., request=None, response=None)` builds
the `ModelCallRecord` exactly as today, calls `self.telemetry.record(record)`,
then — if `inspector` is set — calls `inspector.model_call(record, request if
self.inspect_io else None, response if self.inspect_io else None)` inside
`try/except Exception: logger.warning(...)`. The four call sites pass what they
have: `answer()` → `snapshot_request("answer", messages, (), None, temperature)`
and, on success/invalid_schema, `snapshot_response(content=response["text"],
stop_reason=response["stop_reason"])`; `call_tools()` → `("tools", messages,
[t.name for t in tools], tool_choice, temperature)` and `snapshot_response(
tool_name=decision.tool_name, arguments=decision.arguments, content=decision.content)`;
`_check_budget(estimated, request=...)` → request only. Failure paths pass
`response=None`.

Why a second hook next to `telemetry` rather than widening `ModelCallRecord`:
the record is the persisted cost evidence (`model_calls` table) and its
docstring fixes `detail` to "one line"; prompts in an exported evidence table
are exactly what `state/events.py` warns against ("read as if they had been
reviewed"). The inspector is a *live, developer-only* view, gated by a toggle,
and nothing about it touches a table. Why the same `TurnStream` object is the
inspector: the record and its I/O are produced in one `_record` call on one
thread, so pairing is by argument, not by ordering.

Both gateways in `build_turn` get the inspector (the answering one and the
memory one). D5 (ADR 0015) forbids the memory gateway a `stream` **sink** so a
proposal can never reach the chat bubble; the inspector never calls `delta()`,
only `model_call()`, so it is not a sink and the proposal call becomes
inspectable in the Consolidation phase — which is where "what memory was
proposed and why was it rejected" is answered. A test pins that the memory
gateway's `stream` is still `None` (`tests/composition/test_turn.py::test_the_memory_gateway_carries_no_sink` already does).

### 7.4 `InspectedRetriever` (composition)

```python
@dataclass(frozen=True)
class InspectedRetriever:          # satisfies DocumentRetrieverPort structurally
    inner: HybridRetriever
    stream: TurnStreamLike
    def search(self, query: str, *, limit: int) -> tuple[EvidenceSnippet, ...]:
        outcome = self.inner.search_detailed(query, limit=limit)      # same raises as search()
        self.stream.retrieval(query, limit, outcome, self.inner.minimum_similarity)  # never-fail
        return tuple(hit.chunk.as_snippet() for hit in outcome.hits)  # identical to search()
```
Wired only when a stream is given. The port and the node are untouched; the
fake retrievers in `tests/engine/` keep working because nothing asks them for
`search_detailed`.

### 7.5 `DEV_TRACE_MODEL_IO` (composition/settings)

`Settings.dev_trace_model_io: bool = False`, read from `DEV_TRACE_MODEL_IO=1`,
logged by `log_dev_toggles`, documented in `.env.example` and CLAUDE.md's
`DEV_*` list. `build_turn` sets `inspect_io=settings.dev_trace_model_io` on both
gateways. Off by default: the model I/O is the largest payload and the one
that repeats evidence/history/memory verbatim.

### 7.6 Service tail (web)

`_emit_tail` additionally reads `queries.memory_audit(trace_id)` and
`queries.run(trace_id)` (both already exist) and fills the three new
`TurnFinishedEvent` fields; `ModelCallTotalsOut.records` is built from the
`records` it already fetches.

### 7.7 Nothing else

No change to `TraceEvent`, `EventKind`, `AgentState`, the schema, the ports,
`GraphNodes`, `WorkflowRuntime`, `RunTelemetry`, `ModelCallRecord`, the
approval flow, or the pause store.

---

## 8. Data flow (after the plan)

```
composition/turn.py::build_turn(stream=TurnStream)
   ├─ RunOrchestrator.on_start ─────────▶ stream.context(state)      → SSE context
   ├─ WorkflowRuntime.observer ─────────▶ stream.step(state)         → SSE trace* + step
   ├─ ToolGateway.on_event ─────────────▶ stream.trace_event(ev)     → SSE trace (source=tool_gateway, step=k)
   ├─ LLMGateway.inspector (both) ──────▶ stream.model_call(rec, io) → folded into the next context/step,
   │                                                                     or left for turn_finished.records
   ├─ InspectedRetriever ───────────────▶ stream.retrieval(...)      → folded into the next step
   └─ LLMGateway.stream (answering) ───▶ stream.delta/reset          → SSE token/reset  (unchanged)
web/service.py::run_turn / resume_turn
   ├─ stream.flush_events(final)                                     → SSE trace (post-run rows)
   └─ _emit_tail: answer | approval_required, then turn_finished     → + memory_audit, records, clocks
ui/src/turnReducer.ts (pure) ── TurnView { context, events, steps, answer, summary … }
ui/src/lib/executionTree.ts (pure) ── ExecutionTree { phases[] }
ui/src/components/trace/* ── render
Phase 4: GET /api/runs/{id} ── hydrate.ts ── the same TurnView shape ── the same tree builder
```

Attachment rules in `executionTree.ts` (Phase 2 form; Phase 1 heuristics in §12):
1. A node span opens at an engine `node_entered` row and closes at the matching
   `node_exited`. Its ordinal is the `step_count` of the `step` that closes it
   (the engine increments `step_count` together with `node_entered`, so this is
   exact on fresh and resumed views alike); a span still streaming — no closing
   step yet — is numbered `last step_count + 1`.
2. A `step` with `node !== null` is the post-state of the most recently opened
   span not yet paired with a step; a `step` with `node === null` is an
   engine-level card (Approval / Loop guard), placed in sequence.
3. A live row with `step === k` (gateway) attaches to span `k`, whether or not
   span `k`'s rows have arrived yet (it usually has not — see §1.4).
4. `model_calls` and `retrieval` on a step belong to that span; `context.model_calls`
   to the prelude; records in `turn_finished.model_calls.records` beyond the count
   attributed to prelude + spans are the Consolidation phase's (they are the last
   rows, in order).
5. Rows with node `history`/`memory`/`contract` before the first span are the
   prelude; rows with node `history`/`memory` after the last terminal step are
   Consolidation; `turn_finished.memory_audit` rows join them.
6. The transition line of a span is `steps[k].route` with the reason chosen from
   the span's rows in this priority: last `route_selected` detail → last
   `contract_enforced` → last `tool_called`/`failed`/`rate_limited`/
   `approval_requested` detail → none.

---

## 9. UI interaction / design proposal

- **Cards, not a tree widget.** One `PhaseCard`/`NodeCard` per phase, in run
  order, each a shadcn `Collapsible`. Defaults: node cards **open**, with their
  Input/Decision/Output rows visible; payload blocks (evidence text, tool
  summary, model request/reply, memory statements, history turns) **collapsed**
  behind a labelled trigger with a count or size (`request · 7 blocks · 5.1k
  chars`). Context and Consolidation cards **collapsed** unless they carry
  something notable (a rejection, a redirect, a non-empty contract). Raw events
  **collapsed**.
- **The node header** is the scannable line: `#k · Planner (think) · 1 LLM call ·
  812 ms · → call_tool`. Tone via the existing `traceTone` of the span's worst
  row (`failed`/`run_failed` → destructive, `approval*` → warning, `tool_called`
  → success). Retries: `attempt 2 of 2` from the `retry_scheduled` detail; a
  throttled re-execution is its own card with `retry_count`.
- **Input / Decision / Output** rows use `KeyValueList`; arguments and JSON use
  `JsonBlock`; free text uses the new `TextBlock` (mono, `whitespace-pre-wrap`,
  collapsed beyond 12 lines, "show all" toggle, copy button, `truncated` badge
  when the server clipped it).
- **Evidence** renders as the chips the chat already uses (`CitationChips`'
  document chip → opens `/api/documents/{id}?actor=`), each expandable to the
  snippet text; the retrieval block lists hits as a dense table (`chunk_id ·
  fused · vector rank/score · lexical rank/score`) with the floor and candidate
  counts in a footer line; `gated` renders as a warning line ("closest chunk
  0.29 under the 0.30 floor").
- **Model call** rows: `model · outcome · in/out tokens · cost · latency · attempts`
  with two collapsed blocks, `request` (one `TextBlock` per role block, role as a
  mono chip; the tool names and `tool_choice` as chips) and `reply` (tool call
  name + `JsonBlock` arguments, or the content `TextBlock`). When I/O is off
  (default) the row shows a muted `set DEV_TRACE_MODEL_IO=1 to capture the
  prompt` hint once per turn, not per call.
- **Memory**: recalled records as `kind · key · statement · confidence ·
  recorded_at · memory_id` rows (memory tone); audit rows as `decision` badge
  (`write`/`update` success, `reject` warning + the `rejection` rule in mono) ·
  `kind` · `statement_summary` · `reason`.
- **History**: numbered turns `request → response (clipped)` with route/failure
  chips, `trace_id` as `MonoId`.
- **Transition row** at the bottom of each node card: `→ think — get_project_status
  -> ok` with the route badge tone.
- **Answer card**: text (`answer.text`), citations chips, `FailureBlock` when
  `failure !== 'none'`, `error_detail`.
- **Streaming state**: while `status === 'running'` the last span is "open"
  (spinner in the header, `StreamingCaret` in the Output row); pending gateway
  rows render inside it immediately (they arrive before the span's own rows).
- **Keyboard/a11y**: collapsible triggers are buttons with `aria-expanded`; the
  raw table keeps `data-source` and the gateway tooltip (handbook E2).
- **Removed:** the "Trace" section block and its `/api/runs` link; the
  "Decisions" section. **Kept:** the trace id as a `MonoId` with copy in the
  header — `docs/manual-test.md` §1 and §4.8 take `<trace_id>` from the panel and
  removing it would break the handbook's own verification loop (judgment call,
  flagged for the author).

---

## 10. Payload-size / debug-data strategy

**Server-side bounds (one constant, one helper):** `TEXT_MAX_CHARS = 4_000`
and `clip_text()` in `web/protocol.py` bound every free-text block on the wire
— evidence text, tool summaries, prompt blocks, model reply content. Each block
carries `truncated` and `chars` so the client says exactly what it is not
showing. Structured data (`tool_arguments`, tool-call `arguments`, ranks) is
sent whole; it is small by construction (`StrictArguments` schemas).

**What is not sent:** tool JSON schemas (static code; names suffice), full
`AgentState` snapshots (the delta is sent), `session_summary`/intent bodies
beyond `statement ≤ 400`, embedding vectors, anything from `pauses.state`.

**Duplication policy:** every payload is sent **once, on the event of the
node that produced it** — evidence on the retrieve step, an outcome on its
tool step, memories/history/contract on `context`, audit rows on
`turn_finished`. Two deliberate exceptions: `turn_finished.model_calls.records`
repeats the telemetry of calls already folded into steps (it is the DB's
authoritative list, cheap, and lets the client size the Consolidation phase;
it never repeats I/O), and — only when `DEV_TRACE_MODEL_IO=1` — a prompt block
re-renders evidence/observations/history/memory text, because "what the model
actually received" is the point of that toggle. The client shows both without
merging; a `same as Context ▸` shortcut is not attempted (it would claim an
equality the renderer cannot verify).

**Worst-case per-event size:** a `step` for a planner node with the toggle
off ≈ 2–20 KB (evidence 4 × ≤ 4 KB only on the retrieve step); with the
toggle on, three planner calls × 7 blocks × ≤ 4 KB ≈ 85 KB. Acceptable for a
local dev tool over SSE; `ui/src/sse.ts` buffers on `\n\n` and
`model_dump_json` emits one line, so frame size is not a correctness concern.

**Client-side:** long text collapsed beyond 12 lines; payload blocks lazy —
rendered only when the trigger is opened (a closed `Collapsible` keeps its
content unmounted); the raw table renders `EventRow` exactly as today; nothing
is stored in `localStorage` except, optionally, which phases the developer
left open (per-viewer convenience, try/catch-wrapped).

**Sensitivity:** prompts can contain scope-restricted document text; they are
streamed only to the requester's own turn, only when the toggle is on, and are
never written to any table. The plan does not add prompt capture to
`scripts/run_turn.py` or to eval output.

---

## 11. Testing strategy

### Python (`uv run pytest -q`)

- `tests/web/test_protocol.py` — new models validate; `clip_text` clips at the
  cap with `truncated`/`chars` right; `EVENT_TYPES` has ten members; `TraceRow.step`
  and `StepEvent.node` accept `None`; `ContextEvent` with an empty history/memory
  and `contract=None` is valid (missing optional payloads).
- `tests/web/test_protocol_drift.py` — unchanged test, now green only when
  `protocol.ts` names `context` too. Add a field-name drift test for the new
  interfaces (regex over `interface X {` blocks vs. `model_fields`) so a widened
  `StepEvent` cannot ship without its TS mirror.
- `tests/web/test_stream.py` — `context()` emits one `ContextEvent` with the
  state's history/memories/contract and the pending declaration call; a resumed
  stream (`baseline=`) emits `context` with `resumed=True` and a first `step`
  whose observation delta is only the new outcome; `step` sets `node` from the
  `node_entered` row and `None` for an engine-level observation; evidence is
  emitted on the step where it changed and `None` on the next; `retrieval` and
  `model_call` are folded into the next step and cleared; `trace_event` stamps
  `step = last.step_count + 1`; `elapsed_ms >= 0`; without `context()` the first
  `step` emits a `ContextEvent` itself.
- `tests/web/test_service.py` — `turn_finished` carries `memory_audit`,
  `model_calls.records` and the run clocks (`FakeQueries` grows `memory_audit()`
  and `run()`); `resume_turn` passes the paused state as the stream baseline.
- `tests/engine/test_orchestrator.py` — `on_start` receives the declared state
  (history + memories + contract attached) before `runtime.run`; receives the
  paused state on `resume`; a raising `on_start` is logged and the turn
  completes and files exactly as before.
- `tests/llm/test_gateway.py` — the inspector receives `(record, request,
  response)` on `answered`, `routed`, `invalid_schema`, `provider_failure` and
  `budget_exceeded` (response `None` where nothing came back); with
  `inspect_io=False` both are `None`; request blocks are clipped; an inspector
  that raises never fails the call; `telemetry.record` is still called first.
- `tests/llm/test_inspection.py` — snapshot builders bound text and keep
  `message_chars`; `arguments` is a plain dict copy.
- `tests/composition/test_turn.py` — with a stream, the runtime's retriever is an
  `InspectedRetriever` over the bound `HybridRetriever` and it satisfies
  `DocumentRetrieverPort`; without a stream it is the bare retriever; both
  gateways carry the inspector and `inspect_io` follows the setting; the memory
  gateway still has `stream is None`; `on_start is stream.context`.
- `tests/composition/test_settings.py` — `DEV_TRACE_MODEL_IO=1` parses; absent
  → `False`; logged by `log_dev_toggles`.
- Phase 4: `tests/web/test_app.py` — `GET /api/runs/{id}` matches `RunReportOut`
  and includes `answer.citations` parsed from the filed state.

### UI (`npm --prefix ui run typecheck && npm --prefix ui test`)

- `turnReducer.test.ts` — replaces the "decision de-duplication" block: every
  `step` is kept in order; `context` sets `view.context` and is idempotent on a
  resumed stream; `turn_finished` stores `memory_audit`; the D4 tests are
  untouched.
- `executionTree.test.ts` — spans open/close and are numbered from `step_count`
  on fresh and resumed views; a step with `node: null` becomes an engine card;
  gateway rows with `step: k` attach to span `k` even when they arrive first;
  pending `model_calls` land on the right span; consolidation gets the unattributed
  tail of `records`; the transition reason priority; an unterminated span (still
  streaming) is marked open; an empty view yields no phases.
- `transitionReason.test.ts`, `nodeLabel.test.ts` — pure maps.
- Component tests (`@testing-library/react`): `NodeCard` renders order, label,
  header counts, expands and collapses; `ContextBlock` renders history turns,
  memories and the contract, and "unchecked" for `null`; `EvidenceBlock` renders
  tags and the clipped text with the `truncated` badge; `RetrievalBlock` renders
  hits, floor, and the gated warning; `ToolOutcomeBlock` renders every status
  including `rate_limited` with `retry_after_seconds`; `ModelCallBlock` renders
  with and without I/O, and the toggle hint once; `MemoryAuditBlock` renders
  `reject` with its rule and `write` without one; `TextBlock` collapses long text
  and shows the char counts; every block renders with `null`/empty optional fields
  without throwing (a table-driven test over the optional fields).
- `hydrate.test.ts` (Phase 4) — a filed report becomes a `TurnView` whose tree
  matches the documented attribution rule; observations beyond the rule land in
  an "unattributed" bucket, never on the wrong span.
- `EventRow.test.tsx` unchanged.

### Regression

- `uv run python -m compileall -q src && uv run python -c "import agentic_erp_assistant" && uv run pytest -q`
  (the `postgres` and `live` markers skip themselves without their resources,
  as today).
- `npm --prefix ui run typecheck && npm --prefix ui test && npm --prefix ui run build && npm --prefix ui run lint`.
- Browser walk (`uv run agentic-erp-assistant serve`), per phase: R1
  (documents; evidence block + retrieval hits + composer call), T1 (tool; the
  Planner card shows the arguments *before* the Execute-tool card appears), T6
  (denied read → `failed` tone), A1/A2 (pause, approve from the same bubble and
  from the queue — the queue path must hydrate or show `context` from the resumed
  stream), A3 (deny), M1/M2/M4 (memory written / recalled / rejected with the
  rule), M7 with `DEV_HISTORY_TURN_LIMIT=2` (promotion), T11 with
  `DEV_FLAKY_STATUS=1` (gateway retry rows inside the tool card), G1 with
  `DEV_MAX_STEPS=2` (Loop guard card), and one run with `DEV_TRACE_MODEL_IO=1`.
  Console must be error-free; E1/E2 of the handbook must still pass against the
  raw table.

---

## 12. Implementation phases

Each phase ends green on both suites and is one commit (`ui:`/`web:`/`engine:`/
`llm:`/`composition:`/`docs:` prefixes per CLAUDE.md), on `fpt-bao/refactor-UI`
or a branch off it.

### Phase 0 — Baseline
Record `uv run pytest -q` and `npm --prefix ui test` counts; screenshot the
current panel for R1 and T1. No changes.

### Phase 1 — UI-only: the execution tree from what already streams
- `turnReducer.ts`: keep `steps` (all), drop `decisions`.
- `lib/executionTree.ts`, `lib/nodeLabel.ts`, `lib/transitionReason.ts` with the
  Phase-1 heuristics (no protocol change yet): spans from `node_entered/
  node_exited`; a `step` closes the latest open span not yet paired with a step
  when one exists **and** its `step_count` is greater than the last paired
  step's, otherwise it is engine-level (`_out_of_budget` re-observes with the
  same count; `resume_approval`'s recorded decision arrives with the paused count
  and no span open); gateway rows (`seq: null`) attach to the *next* span to
  open, because they are produced during a node whose own rows only arrive when
  it returns (§1.4).
- `TurnTrace.tsx` → the inspector shell: header (badge, `MonoId` + copy, no
  link), overview strip, node cards with Input (from the previous step / the
  `trace` details), Decision (`route_selected`/`contract_enforced` rows + the
  step's tool/arguments `JsonBlock`), Result (the span's `tool_called`/
  `evidence_retrieved`/`failed` rows), Transition; Consolidation card from the
  post-run rows; Answer card; Raw events (collapsed). Delete `DecisionList.tsx`.
- `docs/manual-test.md` §3: replace the "Decisions sub-list" sentence and the
  "headed by the trace_id, with a link" clause; T1's "Decision row" wording.
- Tests: reducer, tree builder, `NodeCard`, `TransitionRow`, `RawEventsTable`.
- Verify: R1, T1, A1/A2, G1 in the browser.
- Commit: `ui: trace panel as an execution inspector built from the streamed spans`.

### Phase 2 — Protocol: context, node deltas, audit rows (backend + UI)
- `web/protocol.py`: `TextOut`, `HistoryTurnOut`, `MemoryOut`, `ContractOut`,
  `EvidenceOut`, `ToolOutcomeOut`, `MemoryAuditOut`, `ModelCallOut` (no I/O yet),
  `ContextEvent`, widened `TraceRow`/`StepEvent`/`ModelCallTotalsOut`/
  `TurnFinishedEvent`; `EVENT_TYPES += "context"`.
- `engine/orchestrator.py`: `on_start`. `web/stream.py`: `_last`, `baseline`,
  `context()`, delta, `node`, `elapsed_ms`, `step` stamping. `web/service.py`:
  tail fields, `PendingDecision.paused`. `composition/turn.py`: `TurnStreamLike.
  context`, wire `on_start`.
- UI: `protocol.ts` mirror; reducer handles `context`; tree builder switches to
  `StepEvent.node` / `TraceRow.step`; `ContextBlock`, `HistoryBlock`,
  `MemoryBlock`, `ContractBlock`, `EvidenceBlock`, `ToolOutcomeBlock`,
  `MemoryAuditBlock`, `TextBlock`; overview strip gets the duration from the run
  clocks and per-node `elapsed_ms`.
- Tests as in §11 for protocol/stream/service/orchestrator/composition and the
  new blocks; the drift field test.
- Verify: R1 (evidence text visible), T1 (outcome summary + source_ids), A2 from
  the queue (context on the resumed stream), M1/M4 (audit rows with the rule).
- Commits (two): `web: stream the context, each node's delta, and the memory
  audit rows the panel needs` (backend, with the orchestrator hook — one logical
  change: the projection) and `ui: node cards read their payloads from the
  stream`.

### Phase 3 — Model calls, retrieval diagnostics, and the I/O toggle
- `llm/inspection.py`, `LLMGateway.inspector`/`inspect_io`, `_record` /
  `_check_budget` signatures; `composition/settings.py` `DEV_TRACE_MODEL_IO`;
  `.env.example`; `InspectedRetriever`; `TurnStreamLike.model_call/retrieval`;
  `TurnStream.model_call/retrieval` + folding; `ModelRequestOut`/
  `ModelResponseOut`/`RetrievalOut` on the wire.
- UI: `ModelCallBlock`, `RetrievalBlock`; the toggle hint.
- Tests: gateway, inspection, settings, composition, stream folding, blocks.
- Verify: R1 with and without the toggle (retrieval hits with ranks; the
  composer's seven blocks), T1 (two planner calls attributed to spans 1 and 3,
  proposal call in Consolidation), R9 as `guest` (gated retrieval line).
- Commits: `llm: an inspector hook beside telemetry, carrying bounded prompt and
  reply snapshots behind a dev toggle`; `composition: bind the retriever's
  diagnostics and the model-call inspector to the turn's stream`; `ui: model call
  and retrieval blocks`.

### Phase 4 — Hydration of filed runs
- `RunReportOut` + `answer.citations` on `GET /api/runs/{id}`; `protocol.ts::
  RunReport`; `api.ts::getRun`; `hydrate.ts`; `appState.inspectedId` +
  `turn_selected` (clicking an assistant bubble inspects it); `TracePanel`
  fetches and hydrates when `events.length === 0` (history-loaded and
  queue-resumed turns) with a "reconstructed from the filed run" banner and the
  documented attribution: evidence → the `retrieve_project_documents` span;
  observations → one per `call_tool` span in order, then one per `think` span
  containing `approval_requested`; leftovers → an "unattributed" list at run
  level; model calls → run level; audit rows → Consolidation.
- Tests: app route, hydrate, panel selection.
- Verify: open a session from the list and inspect an old R1 and an old A2.
- Commit: `web+ui: a filed run hydrates the same inspector, with its attribution
  labelled as reconstructed`.

### Phase 5 — Optional diagnostics (only if wanted after using Phases 1–4)
- `ConversationMemory.recall_in_detail` (mirror of `SessionMemory`'s), composition
  wrappers that hand `MemorySelection.skipped`/`plan.excluded` and
  `HistorySelection.dropped` to `stream.context` → `ContextEvent.memory_skipped`,
  `history_dropped`.
- `LLMGateway` passes `on_retry` to `retry_with_backoff` and forwards per-attempt
  errors to the inspector (`ModelCallOut.retries: [{attempt, delay, error}]`).
- ADR 0022 — "The inspector widens the projection, not the record: prompts are
  captured live behind a toggle and never filed; a node's input and output are
  the observer's adjacent states, projected by the stream" — and the CLAUDE.md
  web-layer bullet (ten event types, `DEV_TRACE_MODEL_IO`).

---

## 13. Risks and compatibility concerns

| Risk | Mitigation |
|---|---|
| A widened `StepEvent` breaks the reducer's D4 rule or the approval flow | Additive fields with defaults; the reducer's `answer`/`approval` cases are untouched and their tests stay. |
| `protocol.ts` and `protocol.py` drift on *fields*, which the current test does not check | The field-name drift test in Phase 2. |
| Mid-node ordering: gateway rows / model calls arrive before their span's rows | `TraceRow.step` stamping and pending-slot folding make attachment explicit; the Phase-1 "next span" heuristic is replaced, not kept. |
| A resumed decision from the **queue** (fresh assistant bubble, `start_seq` skips the first half) has no first-half spans | `context` is re-emitted on every stream (idempotent); Phase 4 hydrates the first half from the filed run; until then the panel shows the resumed half plus the context, which is what today's panel shows too. |
| `elapsed_ms` is not node execution time exactly — it includes engine bookkeeping and, for the first step, recall + declaration | Labelled `≈` and documented in the card tooltip; the run clocks come from the `runs` row. |
| Prompt capture leaks restricted text | Toggle default off; live-only; never stored; the stream is the requester's own; no eval/script path captures it. |
| SSE frame size with the toggle on | ≤ ~85 KB per step in the worst case; measured in the Phase 3 browser walk; `TEXT_MAX_CHARS` is one constant to lower. |
| `ObservationOut` is shared with the chat's tool chips | Left slim; the full outcome is `ToolOutcomeOut` on the step. |
| `turn_finished` grows (`records`, `memory_audit`) | Both are bounded by the run (tens of rows); no I/O in `records`. |
| `EvidenceQueries.memory_audit`/`run` in `_emit_tail` add two queries per turn | Same connection, already used for `model_calls`; sub-millisecond on nine tables. |
| Hydration attribution can mis-place an observation on unusual runs (a think-node preflight refusal) | The rule is documented; leftovers go to an explicit "unattributed" bucket; the banner says the view is reconstructed; the live view is exact. |
| `docs/manual-test.md` quotes the old panel | Updated in Phases 1–2; E1/E2 remain valid through the raw table. |
| Removing the trace id would break the handbook's `<trace_id>` loop | The id stays in the header (flagged in §9). |
| `test_the_memory_gateway_carries_no_sink` vs. wiring the inspector on the memory gateway | The inspector is not the `stream` sink; the test still asserts `stream is None`; a new test asserts the inspector never calls `delta`. |
| The engine's four-port surface | Untouched — the retriever wrapper lives in composition and satisfies the same port; `on_start` is on the orchestrator, which is already the composition seam. |

---

## 14. Explicitly not changed

- `state/events.py`: `TraceEvent`'s three fields, the 280-char cap, the absence
  of a timestamp, and the closed `EventKind` set (and therefore the
  `trace_events` CHECK constraint).
- `state/agent_state.py`: no new fields, no `STATE_VERSION` bump.
- `engine/nodes.py`, `engine/workflow.py`, `engine/transitions.py`,
  `engine/ports.py`: the loop, the step budget, the transition table, the four
  ports and their fakes.
- `trace/`: `RunRecord`, `TraceStore`/`PauseStore`, `RunTelemetry`, the single
  `save_run` write; `persistence/schema.py` and every table.
- `llm/telemetry.py::ModelCallRecord` (the cost evidence row) and
  `llm/ports.py`.
- `tools/gateway.py`'s eight-step order and its `on_event` hook.
- The approval flow: `precheck_approval`'s 403/404/409 contract, `claim`
  atomicity, `resume_approval`.
- ADR 0015's D4 (the `answer` event is authoritative over streamed tokens) and
  D5 (the memory gateway has no stream sink); D11 (stop = client gives up).
- The chat column, `RouteBadge` labels, `CitationChips`, `FailureBlock`,
  `ApprovalCard`; `EventRow` and its `data-source`/tooltip markers.
- Existing `/api/*` response shapes (Phase 4 types `/api/runs` without changing
  its keys or content; it adds `answer`).
- `scripts/run_turn.py`, the eval harnesses, and `evidence/` outputs.
- The nine existing SSE event names; every existing field on them.
