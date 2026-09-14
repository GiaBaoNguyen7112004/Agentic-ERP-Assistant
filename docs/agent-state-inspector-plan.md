# AgentState Inspector Plan — the full state on every step

**Status:** plan only; no code changed. Written against commit `47f58a4` on
`fpt-bao/tracing-refactor`, after Phases 1–4 of `docs/trace-inspector-plan.md`
shipped. Every file, field and function named below was checked against the
source; anything that does not exist yet is marked **NEW**.

The question this plan answers:

> After each node ran, what was the *whole* `AgentState` — not just the fields
> the stream chose to project — and what exactly did that node change?

## 0. What exists today, in five sentences

1. `state/agent_state.py::AgentState` is the one object a node reads and
   returns (26 fields, frozen, `STATE_VERSION = 2`); its docstring says
   "reviewing a run means reading them" — the sequence of states *is* the
   review artifact.
2. The engine already hands every post-node state to an observer
   (`engine/workflow.py::WorkflowRuntime.observer`), and the orchestrator hands
   the pre-graph state to `on_start`; composition wires both to
   `web/stream.py::TurnStream.step` / `.context`.
3. `TurnStream.step` keeps that state in `self._last` and puts only a
   **delta projection** on the wire (`StepEvent`: `evidence` once,
   `observations[new:]`, `response`, `failure`, …) — the earlier plan's §10
   explicitly decided "full `AgentState` snapshots are not sent".
4. The UI (`ui/src/lib/executionTree.ts`, `components/trace/NodeCard.tsx`)
   renders those deltas as per-topic blocks; nothing shows a state, and
   `AgentState` appears in `ui/src/` only as the REST mirror
   `protocol.ts::AgentStateSnapshot`, used by `hydrate.ts` for the filed
   **final** state (`runs.state`).
5. A filed run holds one state (the final one); intermediate states are not
   persisted anywhere.

## 1. Decisions (and what was rejected)

**D1 — Reverse §10 of the trace-inspector plan: send the full state on every
`step`, and the starting state on `context`.** The delta projection made the
reader reconstruct the state mentally across cards; a developer asking "what
did node 3 see?" has to sum four earlier deltas. The state is what the engine
already holds in `_last`; putting it on the wire is one field. The delta
fields on `StepEvent` **stay** — the existing blocks read them and they are
cheap; removing them is a separate cleanup after the state block has proven
itself.

**D2 — The wire type is `AgentState` itself, not a clipped `AgentStateOut`.**
`RunOut.state: AgentState` already crosses the REST wire unclipped, its TS
mirror `AgentStateSnapshot` already exists, and
`tests/web/test_protocol_drift.py` already drift-tests it field-for-field
against `AgentState.model_fields`. A second, lossy shape would be two
AgentStates on the wire and a place for them to disagree. Rejected: a
`TextOut`-clipped mirror (§10's bound policy) — "full" means full; the size is
bounded by construction (see §7, Risks) and measured in the browser walk.

**D3 — The diff is computed client-side, per top-level field, against the
previous step's state (or the `context` state for step 1).** The server does
not send a diff: it would be a third shape, and the client already has both
states. `ui/src/lib/stateDiff.ts` **NEW** is a pure function with three
outcomes per field: `unchanged`, `changed`, `appended` (an array field the
node only extended — `evidence`, `observations`, `events`, `history`,
`memories`).

**D4 — Filed runs are honest: only the last node's card carries a state.**
The record holds the final state only; `hydrate.ts` puts it on the last step
and `null` everywhere else, and the state block says why. Persisting a
snapshot per step is designed in §8 as an optional later phase, **not**
part of this change — it is a schema and a `TraceStore` port change, and the
live view (the thing the user asked for) needs neither.

**D5 — TS `state` is `AgentStateSnapshot | null`; Python `state` is
`AgentState`, required.** The server never sends `null`; the client type
admits it only because a hydrated step has nothing to put there (D4). The
drift test checks field *names*, so the two stay in sync where it matters; the
asymmetry is written down in `protocol.ts` next to the field.

**D6 — Collapsed by default, with an informative trigger line.** A full state
is 20–30 KB of JSON; open by default it would push every other block off
screen. The trigger reads `State · 4 changed · route, tool_name,
tool_arguments, step_count`, so the scan line already answers "what did this
node touch" without expanding.

## 2. Wire contract (`web/protocol.py` ↔ `ui/src/protocol.ts`)

### 2.1 Python — two fields, no new event type

```python
class ContextEvent(_Event):
    ...
    model_calls: tuple[ModelCallOut, ...]
    state: AgentState                       # NEW
    """The state the engine is about to run from -- history, memories and
    contract attached, no node executed yet (or, on a resume, the paused
    state exactly as the pause store returned it). The baseline every
    later ``StepEvent.state`` is read against; the same object
    ``TurnStream._last`` is seeded with."""


class StepEvent(_Event):
    ...
    model_calls: tuple[ModelCallOut, ...]
    state: AgentState                       # NEW
    """The whole state the node returned -- the exact object the engine's
    observer was handed, events included. The delta fields above are
    projections of this one for the blocks that read them; this is the
    record itself. Sent whole on every step, deliberately (see
    ``docs/agent-state-inspector-plan.md`` D1/D2): a reader asking "what
    did node 3 see" should not have to sum three earlier deltas."""
```

`EVENT_TYPES` is unchanged (ten members). Update the `StepEvent` class
docstring's first paragraph — it currently says the event "is a projection of
the state a screen needs, not the state itself"; that sentence is now false
and must go.

`AgentState` serializes cleanly already (`RunOut.state` proves it):
`tool_arguments` has a `field_serializer`, `scopes` (frozenset) and
`redirected_needs` (frozenset) become JSON arrays in unspecified order.

### 2.2 TypeScript — mirror, with the one documented asymmetry

```ts
export interface ContextEvent {
  ...
  model_calls: ModelCallOut[]
  /** The state the engine starts from. Always sent live; `null` only on a
   * view hydrated from a filed run, which files no starting state
   * (docs/agent-state-inspector-plan.md D4/D5). */
  state: AgentStateSnapshot | null
}

export interface StepEvent {
  ...
  model_calls: ModelCallOut[]
  /** The whole state after this node. Always sent live; `null` only on a
   * hydrated step other than the last (the record holds the final state
   * only -- see hydrate.ts). */
  state: AgentStateSnapshot | null
}
```

`AgentStateSnapshot` and the interfaces it embeds (`TraceEvent`,
`EvidenceSnippet`, `MemoryRecord`, `ConversationTurn`, `ReplyContract`,
`ToolOutcome`) already exist further down `protocol.ts`; TypeScript hoists
interface declarations, so nothing moves.

### 2.3 Every place that builds one of these by hand (grep-verified)

| File | What to add |
|---|---|
| `src/agentic_erp_assistant/web/stream.py` (`context()`, `step()`) | `state=state` |
| `tests/web/test_protocol.py` lines ~214 and ~225 (`ContextEvent(...)`) | `state=<an AgentState>` — add a small `_state()` builder at the top of the file (copy the one in `tests/web/test_stream.py:42`) |
| `ui/src/__tests__/fixtures.ts::step()` | `state: null` default |
| `ui/src/__tests__/fixtures.ts` | **NEW** `context(overrides)` builder: `{ type: 'context', request: 'hi', history: [], memories: [], contract: null, model_calls: [], state: null, ...overrides }` |
| `ui/src/__tests__/turnReducer.test.ts` lines 82, 95, 96; `ui/src/__tests__/ContextBlock.test.tsx` line 8 | switch to `context({...})` from fixtures (or add `state: null`) |
| `ui/src/hydrate.ts` (`baseStep`, the `context` literal) | `state: null` in `baseStep`; `state: null` on context; the last step gets `report.run.state` (§5.4) |

No other file constructs a `StepEvent` or `ContextEvent` literal
(`grep -rn "StepEvent(\|ContextEvent(" src tests scripts`,
`grep -rln "type: 'step'\|type: 'context'" ui/src`).

## 3. Stream (`web/stream.py`)

Two one-line additions; nothing else moves.

```python
def context(self, state: AgentState) -> None:
    ...
    self._put(
        ContextEvent(
            request=state.request,
            history=...,
            memories=...,
            contract=contract,
            model_calls=model_calls,
            state=state,                    # NEW
        )
    )
    self._last = state

def step(self, state: AgentState) -> None:
    ...
    self._put(
        StepEvent(
            ...,
            model_calls=model_calls,
            state=state,                    # NEW
        )
    )
    self._last = state
```

The resumed path needs nothing: `engine/orchestrator.py::resume` already
calls `on_start` with the claimed paused state (line ~384), so a resumed
stream's `context.state` is the paused state and its first `step.state` is
the state after the recorded decision — exactly the "before → after" pair the
diff wants.

`flush_events` stays trace-rows-only: consolidation appends `memory_*` /
`history_promoted` rows to `events` and nothing else, and the filed state is
`RunOut.state` for anyone who needs it.

## 4. Python tests

`tests/web/test_protocol.py`

- `test_context_event_requires_a_state` — constructing without `state` raises
  `ValidationError`.
- `test_step_event_carries_the_state_and_round_trips_through_json` — build a
  `StepEvent` with a state that has `tool_arguments={"milestone_id": "M2"}`,
  `scopes`, one `EvidenceSnippet`, one `ToolOutcome`, two events;
  `AgentState.model_validate_json(json.loads(event.model_dump_json())["state"])
  == state` (the `field_serializer` on `tool_arguments` and the frozensets
  survive the wire).

`tests/web/test_stream.py` (use the existing `state()` and `run_and_collect`)

- `test_context_carries_the_state_the_engine_starts_from` —
  `events[0].state == the state passed to context()`.
- `test_step_carries_the_whole_state_the_observer_was_handed` — call
  `context(s0)`, `step(s1)`, `step(s2)` where `s2 = s1.evolve(observations=...,
  step_count=2)`; assert `steps[0].state == s1`, `steps[1].state == s2`, and
  `steps[1].state.observations` is the **full** tuple while
  `steps[1].observations` (the delta) has length 1 — the two coexist by
  design.
- `test_a_resumed_streams_context_state_is_the_paused_state` — `context()`
  with a state whose `approval == "pending"`, `tool_name` set; assert the
  event's `state.approval == "pending"`.

`tests/web/test_protocol_drift.py` — no change; `AgentStateSnapshot ↔
AgentState` is already in `_MIRRORED_MODELS`, and `StepEvent`/`ContextEvent`
are too, so a missing `state` on either side fails the suite.

`tests/web/test_service.py` — run it; it drives a real `TurnStream` through a
fake orchestrator and should pass untouched. If a test there asserts on an
exact `StepEvent` field set, add `state`.

## 5. UI

### 5.1 `ui/src/lib/stateDiff.ts` **NEW** (pure, tested)

```ts
import type { AgentStateSnapshot } from '../protocol'

export type StateField = keyof AgentStateSnapshot

/** The four sections of state/agent_state.py, in its own order, so the
 * block reads like the model. A test asserts every key of
 * AgentStateSnapshot appears here exactly once. */
export const STATE_FIELD_GROUPS: readonly { title: string; fields: readonly StateField[] }[] = [
  { title: 'Turn', fields: ['request', 'actor', 'project_code', 'scopes', 'trace_id', 'session_id'] },
  {
    title: 'Decided and gathered',
    fields: [
      'route', 'evidence', 'memories', 'history', 'contract', 'redirected_needs',
      'draft', 'observations', 'tool_name', 'tool_arguments', 'tool_mutating', 'approval',
    ],
  },
  { title: 'How it ended', fields: ['response', 'failure', 'error_detail', 'terminal'] },
  { title: 'Record', fields: ['step_count', 'retry_count', 'events', 'state_version'] },
]

/** Fields the server serializes from a frozenset -- order carries no
 * meaning, so they are compared as sorted arrays. */
const SET_FIELDS: ReadonlySet<StateField> = new Set(['scopes', 'redirected_needs'])

export type ChangeKind = 'unchanged' | 'changed' | 'appended'

export interface FieldChange {
  field: StateField
  kind: ChangeKind
  before: unknown   // null when there is no previous state
  after: unknown
  /** For `appended`: the elements `after` has beyond `before`. */
  appended: unknown[] | null
}

/** Structural equality, key-order-insensitive for objects. Small and local
 * on purpose -- no dependency for a 20-line function. */
export function deepEqual(a: unknown, b: unknown): boolean { ... }

/**
 * One entry per field of `next`, in STATE_FIELD_GROUPS order. With no
 * `previous` every field is `changed` with `before: null` (the initial
 * state: everything is new). An array field whose previous value is a
 * strict prefix of the next one is `appended`, never `changed` -- that is
 * what "a node added an observation" looks like, and the block renders
 * only the tail.
 */
export function diffState(previous: AgentStateSnapshot | null, next: AgentStateSnapshot): FieldChange[] { ... }

/** The `changed`/`appended` entries only -- what the trigger line lists. */
export function changedFields(changes: FieldChange[]): FieldChange[]
```

Prefix rule for `appended`: `Array.isArray(before) && Array.isArray(after) &&
after.length > before.length && before.every((item, i) => deepEqual(item,
after[i]))`. Equal-length or shorter arrays fall through to `changed` /
`unchanged`.

### 5.2 `ui/src/lib/executionTree.ts` — one field on `ExecutionEntry`

```ts
export interface ExecutionEntry {
  ...
  /** The state this entry started from: the previous closed entry's
   * `step.state`, else the context's `state`, else `null` (no context
   * arrived, or a hydrated view). Computed here, not in the card, so the
   * "before" of every diff is one pure, tested rule. */
  stateBefore: AgentStateSnapshot | null
}
```

In `buildExecutionTree`, keep `let lastState: AgentStateSnapshot | null =
view.context?.state ?? null`; when pushing an entry set `stateBefore:
lastState` and then `lastState = item.step.state ?? lastState` (a hydrated
`null` state must not erase a known baseline). The two trailing
still-streaming entries get `stateBefore: lastState` too.

### 5.3 `ui/src/components/trace/StateBlock.tsx` **NEW**

Props: `{ state: AgentStateSnapshot | null; previous: AgentStateSnapshot |
null; title?: string }`.

Rendering, top to bottom:

1. `state === null` → one muted line: *"State not carried — a reconstructed
   run files only its final state (runs.state); see the last node."* Nothing
   else.
2. A `Collapsible` (shadcn, like `NodeCard`), **closed by default**. Trigger
   line: chevron · `State` · `{n} changed` badge (`ToneBadge`, `info` when
   n > 0, `neutral` when 0) · the changed field names as mono chips (first 6,
   then `+k`). With `previous === null` the badge reads `initial` instead of
   a count.
3. Inside, a toolbar row: a `Button variant="ghost" size="sm"` toggling
   `changed only` / `all fields` (default `changed only` when `previous !==
   null`, `all fields` when it is `null`), and a `Copy JSON` button
   (`navigator.clipboard?.writeText(JSON.stringify(state, null, 2))`, same
   pattern as `MonoId`).
4. For each group in `STATE_FIELD_GROUPS` (skipped entirely when it has
   nothing to show under the current filter): `SectionHeading` with the
   group title, then one row per field, `data-field={field}` and
   `data-change={kind}` on the row for tests:
   - name in mono (`text-[11px] text-muted-foreground`), a small `changed` /
     `+n` marker next to it when applicable;
   - value via `renderValue(value)`: `null` → muted `null`; boolean/number →
     mono inline; string ≤ 120 chars and no newline → mono inline; longer
     string → `TextBlock`; array/object → `JsonBlock` (an empty array →
     mono `[]`);
   - `changed` scalar (string/number/boolean/null on both sides) → one line
     `before → after` (before struck through / muted);
   - `changed` non-scalar → `before` and `after` stacked as two `JsonBlock`s
     with 10px labels;
   - `appended` → `JsonBlock` of `appended` only, labelled `+n appended`,
     plus a `show all m` ghost button that swaps in the full array.

Every value path must render with empty arrays, `null`, and a state whose
`tool_arguments` is `null` (the table-driven "optional fields" test pattern
already used by the other blocks).

### 5.4 Wiring

- `NodeCard.tsx`: after the `Call` section and before `<TransitionRow>`:
  ```tsx
  {entry.step && (
    <StateBlock state={entry.step.state} previous={entry.stateBefore} />
  )}
  ```
  While streaming (`entry.step === null`) render nothing — the state arrives
  with the closing step.
- `ContextBlock.tsx`: a last section `SectionHeading` "Initial state" →
  `<StateBlock state={context.state} previous={null} />`.
- `hydrate.ts`: `baseStep` gets `state: null`; after the "filed reply closes
  the last span" block add `steps[lastIndex].state = state`; the `context`
  literal gets `state: null`. Extend the header comment's attribution list
  with `state -> the filed (final) state on the LAST span only; every other
  step and the context carry null -- nothing intermediate is filed`.
- `TurnTrace.tsx`: in the reconstructed banner append *" Only the final
  state is filed; earlier nodes show no state."*
- `docs/manual-test.md` §3: one sentence — each node card has a collapsed
  `State` row listing the fields that node changed; expanding it shows the
  whole `AgentState` after that node.

### 5.5 UI tests (`npm --prefix ui test`)

- `stateDiff.test.ts` — `STATE_FIELD_GROUPS` covers every key of a full
  `AgentStateSnapshot` fixture exactly once (build the fixture in
  `fixtures.ts` as `agentState(overrides)` — **NEW** builder, all 26 fields);
  `previous === null` → every field `changed` with `before: null`;
  identical states → all `unchanged`; `route: null → 'call_tool'` →
  `changed` with the pair; `observations: [] → [o1]` → `appended` with
  `[o1]`; `[o1, o2] → [o1]` → `changed`; `scopes` in a different order →
  `unchanged`; `tool_arguments` with keys in a different order →
  `unchanged`; `changedFields` drops `unchanged`.
- `executionTree.test.ts` — with a context carrying `state: s0` and two steps
  carrying `s1`, `s2`: `entries[0].stateBefore === s0`,
  `entries[1].stateBefore === s1`; with no context, `entries[0].stateBefore
  === null`; a hydrated `null` step state does not erase the baseline for
  the next entry; the still-streaming trailing entry has `stateBefore` set.
- `StateBlock.test.tsx` — `state: null` renders the "not carried" line and
  no trigger; the trigger shows `2 changed` and the two field chips; clicking
  it reveals rows with `data-change="changed"` only; toggling `all fields`
  reveals `data-change="unchanged"` rows too; `previous: null` shows
  `initial` and opens in `all fields` mode; an `appended` observations row
  shows `+1 appended`; the copy button calls `clipboard.writeText` with the
  JSON (stub `navigator.clipboard` as `MonoId`'s test does, if one exists —
  otherwise `vi.stubGlobal`).
- `NodeCard.test.tsx` — a step with `state` and an entry with `stateBefore`
  renders the `State` trigger with the changed count; a streaming entry
  (`step: null`) renders no `State` trigger.
- `ContextBlock.test.tsx` — a context with `state` renders the "Initial
  state" heading; with `state: null` renders the "not carried" line.
- `hydrate.test.ts` — the last step's `state` is `report.run.state`; every
  earlier step's is `null`; `context.state` is `null`.
- `turnReducer.test.ts` — untouched except the fixture switch (§2.3).

## 6. Phases, commits, verification

Each phase ends green on both suites (`uv run python -m compileall -q src &&
uv run python -c "import agentic_erp_assistant" && uv run pytest -q`; `npm
--prefix ui run typecheck && npm --prefix ui test && npm --prefix ui run
build`) and is one commit on `fpt-bao/tracing-refactor`.

### Phase 1 — `web: carry the whole AgentState on context and every step`

§2.1, §3, §4, plus the `protocol.ts` mirror and the fixture defaults (§2.2,
§2.3) so `typecheck` stays green — the TS side has to change in the same
commit or the drift test fails. No rendering yet.

Commit body must record D1 (the reversal of trace-inspector-plan §10 and
why), D2 (why `AgentState` itself and not a clipped mirror), and the measured
per-step frame size from one T1 run (Chrome DevTools → Network → the SSE
response → size of a `step` frame).

Verify: `uv run agentic-erp-assistant serve`, run T1, confirm in the Network
tab that each `event: step` frame carries `"state":{...}` with
`"state_version":2`, and the chat/approval flow is unchanged.

### Phase 2 — `ui: a State block on every node card, diffed against the state before it`

§5.1–§5.5, `docs/manual-test.md`.

Verify in the browser (console must stay error-free):

| Case (handbook id) | What the State row must show |
|---|---|
| R1 (documents) | node 1 `Planner`: `route`, `contract`-related fields; node 2 `Retrieve & compose`: `evidence` `+n appended`, `response` changed `null → "…"`, `terminal` `false → true`, `events` appended |
| T1 (tool) | node 1: `route`, `tool_name`, `tool_arguments`, `step_count`; node 2: `observations +1`, `events`; node 3: `response`, `terminal` |
| A1/A2 (pause, approve from the same bubble) | the paused step: `approval` `not_required → pending`; the resumed stream's Context "Initial state" shows `approval: pending`; the first resumed card: `approval → approved`, `observations +1` |
| A3 (deny) | the engine card `Approval`: `approval → denied`, `failure`, `response` set |
| G1 with `DEV_MAX_STEPS=2` | the `Loop guard` engine card: `failure` changed, `terminal` true |
| an old run from history | the last card has the filed state, every earlier card says "not carried", the banner's extra sentence is present |
| `changed only` ↔ `all fields` | toggling on node 1 of T1 reveals `request`, `actor`, `scopes`, … as `unchanged` |

### Phase 3 (optional, only if wanted after using Phase 2) — see §8

## 7. Risks

| Risk | Mitigation |
|---|---|
| Frame size: a `step` now repeats evidence text, every observation summary, the growing `events` list, history and memories on every node | Bounded by construction: evidence ≤ retriever `limit` chunks, observations ≤ the step budget, `events` ≤ ~20 rows × 280 chars, history/memories are already clipped/≤ 400 chars. Worst case measured in Phase 1 and written in the commit; if it exceeds ~100 KB per step, the fallback is to drop `events` from the wire copy via a `field_serializer`-free `AgentState.model_dump(exclude={"events"})` **on a dedicated `StateOut` — which is D2's rejected path, so only with the measurement to justify it**. |
| `scopes`/`redirected_needs` serialize in set order and could differ between two frames | `diffState` compares them sorted; a test pins it. |
| The reducer keeps every step's state in memory for the session (`TurnView.steps`) | Tens of KB per step, a handful of steps per turn; the chat already keeps every turn's `TurnView`. Not a concern at dev scale; noted, not solved. |
| A hydrated view's `null` states read as "the node had no state" | The block's own wording says *not carried* and why; the banner repeats it. |
| The `StepEvent` docstring promised "a projection, not the state itself" | Rewritten in Phase 1 (§2.1); ADR-adjacent reasoning goes in the commit body. |
| `test_protocol_drift` now guards a field whose *type* differs across the wire (`AgentState` vs `AgentStateSnapshot \| null`) | It only ever checked names; the `\| null` is commented at the field (D5). |

## 8. Optional Phase 3 — filing a snapshot per step (design only, not scheduled)

Only worth doing if, after Phase 2, hydrated runs showing one state turns out
to matter in practice. It is a real record change, so it gets its own ADR.

- **Where the states come from:** the engine's observer already sees every
  one. In `engine/orchestrator.py::handle`/`resume`, wrap the runtime's
  observer so each observed state is also appended to a local list (the
  orchestrator owns the run; `WorkflowRuntime` stays untouched — composition
  already passes `observer=stream.step`, so the fan-out is `observer =
  _both(stream.step, collected.append)` built in the orchestrator, never in
  `web/`).
- **Record:** `trace/records.py::RunRecord` gains `snapshots: tuple[StateSnapshot, ...] = ()`
  where `StateSnapshot(step_count: int, node: str | None, state: AgentState)`,
  frozen. The default keeps every existing constructor call valid.
- **Table:** `persistence/schema.py` — a tenth table
  `state_snapshots (trace_id REFERENCES runs ON DELETE CASCADE, step_count
  integer, node text, state jsonb, PRIMARY KEY (trace_id, step_count))`,
  written inside `PostgresTraceStore.save_run` after the `runs` upsert (same
  transaction as `trace_events`, `ON CONFLICT DO NOTHING`); `InMemoryTraceStore`
  keeps them in a dict. `scripts/init_postgres.py` needs no change (the
  template is applied whole).
- **Read model:** `EvidenceQueries.state_snapshots(trace_id) ->
  tuple[StateSnapshot, ...]`; `RunReportOut.state_snapshots`; `protocol.ts::
  RunReport.state_snapshots`; drift-test the new model.
- **Hydration:** `hydrate.ts` matches snapshots to spans by `step_count` (the
  engine numbers them identically) and sets `steps[i].state`; the "not
  carried" wording then applies only to runs filed before the table existed.
- **Why the resumed segment is fine:** `save_run` is called once per segment
  (handle / resume), each with its own observed states; the primary key on
  `(trace_id, step_count)` is exactly the engine's own numbering, so the two
  segments never collide.
- **Cost:** one jsonb row per node execution, ~20–30 KB each; `runs.state` is
  already the same size. Retention is not a question this project has had to
  answer yet.

## 9. Explicitly not changed

- `state/agent_state.py` — no new field, no `STATE_VERSION` bump.
- `engine/` — nothing; the observer and `on_start` already carry the state.
- `trace/`, `persistence/`, the schema — nothing (until §8, if ever).
- `TurnStream`'s delta computation, `_pending_*` folding, `trace_event`
  stamping, `flush_events`.
- Every existing `StepEvent`/`ContextEvent` field; the ten event types.
- `hydrate.ts`'s attribution rule for evidence/observations/tool call —
  only `state` is added to it.
- `DEV_TRACE_MODEL_IO` and the model-call inspector — the state block is
  not behind a toggle: the state is already computed and already held by the
  stream; nothing new is captured, only sent.
