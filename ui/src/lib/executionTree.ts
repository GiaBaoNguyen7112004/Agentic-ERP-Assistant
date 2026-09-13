import type { StepEvent, TraceRow } from '../protocol'
import type { TurnView } from '../turnReducer'

/**
 * One executed node, or one engine-level state change that closed with no
 * node span of its own (an approval decision being recorded, a denial's
 * refusal, the loop guard). `kind` distinguishes them for labelling and
 * numbering; the shape is otherwise the same, because a reader wants both
 * treated as "a step the engine took" -- see `nodeLabel.ts` for what a
 * missing `nodeName` renders as.
 */
export interface ExecutionEntry {
  kind: 'node' | 'engine'
  /** `step.step_count` when this entry closed with a step; best-effort
   * (one past the last closed entry) while still streaming. */
  ordinal: number
  /** The `node_entered` row's `node` for a node entry; the first row's
   * `node` for a bare engine entry with at least one row; `null` for one
   * with none (nothing to name it by). */
  nodeName: string | null
  /** Every row this entry owns, in arrival order -- live gateway rows first
   * (they are produced, and therefore arrive, before the node they belong
   * to has flushed anything -- see the module doc below), then the node's
   * own persisted rows. */
  rows: TraceRow[]
  /** The step that closed this entry, or `null` while it is still the one
   * being streamed (no `node_exited`/closing step has arrived yet). */
  step: StepEvent | null
}

export interface ExecutionTree {
  /** Rows filed before the first node ran -- `history_recalled`,
   * `memory_recalled`, `contract_declared`. Emitted by the orchestrator
   * before the engine is touched, so they arrive folded into the very first
   * batch this module ever sees, ahead of that batch's own `node_entered`
   * row -- split off here rather than left as part of node 1. */
  prelude: TraceRow[]
  /** One entry per node execution or bare engine-level step, in run order. */
  entries: ExecutionEntry[]
  /** Rows filed after the engine finished -- `memory_written`,
   * `memory_rejected`, `history_promoted`. These arrive with no step of
   * their own (`TurnStream.flush_events` streams trace rows only), so they
   * are recognized as every row still pending once the timeline runs out of
   * steps to close them with. */
  consolidation: TraceRow[]
}

const NODE_ENTERED = 'node_entered'

function isNodeEntered(row: TraceRow): boolean {
  return row.source === 'engine' && row.kind === NODE_ENTERED
}

/** The name this span is known by: the `node_entered` row's own `node`, when
 * there is one -- never `rows[0]`, which is often a live gateway row that
 * arrived ahead of the node it belongs to (see the module doc) and would
 * otherwise misname the whole span after its gateway prefix. Falls back to
 * the first row's `node` only for a bare engine entry, which has no
 * `node_entered` row to name it by at all. */
function nodeNameOf(rows: readonly TraceRow[]): string | null {
  return rows.find(isNodeEntered)?.node ?? rows[0]?.node ?? null
}

/** The three kinds the orchestrator files before the engine ever runs (see
 * `engine/orchestrator.py`'s `_with_history` / `_recalled` / `_declared`,
 * in that order). Only these -- never "the first batch has no node_entered"
 * -- mark a leading run of rows as prelude: a stream that opens with, say,
 * an `approval_recorded` row and no node span before it (a fresh view built
 * for a queue-resumed decision, `App.tsx::decideFromQueue`) is a real first
 * entry, not prelude that happens to look like one. */
const PRELUDE_KINDS = new Set(['history_recalled', 'memory_recalled', 'contract_declared'])

/** Split the leading prelude rows off a batch of rows, only from the very
 * first batch this timeline produces. */
function splitPrelude(rows: TraceRow[]): { prelude: TraceRow[]; rest: TraceRow[] } {
  let splitAt = 0
  while (splitAt < rows.length && PRELUDE_KINDS.has(rows[splitAt].kind)) splitAt += 1
  return { prelude: rows.slice(0, splitAt), rest: rows.slice(splitAt) }
}

/**
 * The tool call this entry is actually about, or `null` when it has none to
 * show.
 *
 * `StepEvent.tool_name`/`tool_arguments` are not, by themselves, a safe
 * signal: they project `AgentState.tool_name`/`tool_arguments`, and nothing
 * clears those fields when a later decision moves on to a route that calls
 * nothing (`advance(state, "answer", ...)` leaves them exactly as a prior
 * `call_tool` decision set them -- see `engine/nodes.py::think`). Read
 * naively, the *next* planner card after a tool ran would appear to be
 * calling that same tool again. A step's tool fields are this entry's own
 * only when the entry is actually about one: the node that executed a call
 * (`call_tool`), an approval decision (`approval`), or a decision that just
 * chose to make or escalate one (`step.route` is `call_tool` or
 * `request_approval`).
 */
export function entryToolCall(
  entry: ExecutionEntry,
): { name: string; arguments: Record<string, unknown> | null } | null {
  const step = entry.step
  if (step === null || step.tool_name === null) return null

  const relevant =
    entry.nodeName === 'call_tool' ||
    entry.nodeName === 'approval' ||
    step.route === 'call_tool' ||
    step.route === 'request_approval'
  return relevant ? { name: step.tool_name, arguments: step.tool_arguments } : null
}

/**
 * Group a turn's interleaved trace rows and step events into the spans a
 * developer reads as "what the graph did, in order".
 *
 * This is a *heuristic* over the wire shape the engine and the web layer
 * already produce (see `engine/workflow.py::WorkflowRuntime.run` and
 * `web/stream.py::TurnStream.step`), not a new contract: a node's whole
 * batch of trace rows -- its own `node_entered`, everything it did, and its
 * `node_exited` -- is flushed in one `step()` call, immediately followed by
 * exactly one `StepEvent`, so the row batch accumulated since the previous
 * step closes when the next step arrives. A live gateway row (no `seq`) is
 * pushed to the stream the instant it happens, which is *during* the node
 * that produced it -- before that node's own rows have flushed -- so it
 * always lands in `timeline` ahead of the batch it belongs to, never inside
 * a prior, already-closed one.
 *
 * `docs/trace-inspector-plan.md` §7.2 replaces this grouping with an
 * explicit `StepEvent.node` / `TraceRow.step` once the protocol carries
 * them; nothing here should get more elaborate than this doc explains.
 */
export function buildExecutionTree(view: TurnView): ExecutionTree {
  let prelude: TraceRow[] = []
  const entries: ExecutionEntry[] = []
  let pending: TraceRow[] = []
  let sawFirstBatch = false
  let sawAnyStep = false

  for (const item of view.timeline) {
    if (item.kind === 'row') {
      pending.push(item.row)
      continue
    }

    sawAnyStep = true
    let rows = pending
    pending = []

    if (!sawFirstBatch) {
      sawFirstBatch = true
      const split = splitPrelude(rows)
      prelude = split.prelude
      rows = split.rest
    }

    entries.push({
      kind: rows.some(isNodeEntered) ? 'node' : 'engine',
      ordinal: item.step.step_count,
      nodeName: nodeNameOf(rows),
      rows,
      step: item.step,
    })
  }

  // Whatever is left in `pending` never got closed by a step. Two honest
  // readings: the run is still going and this is the node currently
  // streaming (it has a node_entered row of its own), or the run finished
  // and the orchestrator's post-run rows arrived with no step to pair them
  // with at all (consolidation).
  let consolidation: TraceRow[] = []
  if (pending.length > 0) {
    if (!sawAnyStep) {
      // Still inside the very first, still-open batch -- no step has closed
      // it yet, but any prelude rows already folded in ahead of it can
      // still be split off.
      const split = splitPrelude(pending)
      prelude = split.prelude
      entries.push({ kind: 'node', ordinal: 1, nodeName: nodeNameOf(split.rest), rows: split.rest, step: null })
    } else if (pending.some(isNodeEntered)) {
      const lastOrdinal = entries.length > 0 ? entries[entries.length - 1].ordinal : 0
      entries.push({
        kind: 'node',
        ordinal: lastOrdinal + 1,
        nodeName: nodeNameOf(pending),
        rows: pending,
        step: null,
      })
    } else {
      consolidation = pending
    }
  }

  return { prelude, entries, consolidation }
}
