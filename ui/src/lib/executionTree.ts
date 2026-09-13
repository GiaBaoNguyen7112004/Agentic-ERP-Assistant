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
  /** `step.node` for a real node span -- the `node_entered` row's own
   * name, read directly off the field the server stamps it with, never
   * guessed from row order. A bare engine entry has no node to be named
   * by that way, so it falls back to its own first row's `node` (e.g.
   * `approval_recorded`'s row says "approval"), or `null` when it has no
   * rows at all. For a still-streaming entry (no step yet) this falls
   * back further, to searching its own rows for a `node_entered` row --
   * the one case nothing has told the client the name yet. */
  nodeName: string | null
  /** Every row this entry owns, in arrival order -- live gateway rows
   * first (attached by their own `step` stamp -- see `TraceRow.step` --
   * because they are produced, and therefore arrive, *during* the node
   * they belong to, before that node's own rows have flushed), then the
   * node's own persisted rows. */
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
   * row -- split off here rather than left as part of node 1. A richer
   * reading of the same prelude is `TurnView.context` (the `ContextEvent`),
   * when the stream carried one -- these flat rows are the fallback for a
   * view that never got one (an old session, a caller with no `on_start`
   * wired) and always what the raw events table shows regardless. */
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

/** The name a still-streaming entry is known by, before it has a step of
 * its own to read `.node` off: the `node_entered` row's own name, when
 * there is one -- never `rows[0]`, which is often a live gateway row that
 * arrived ahead of the node it belongs to and would otherwise misname the
 * whole span after its gateway prefix. */
function openEntryNodeName(rows: readonly TraceRow[]): string | null {
  return rows.find(isNodeEntered)?.node ?? null
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
 * Node identity now comes straight off the wire: `StepEvent.node` names the
 * span a step closes, and `TraceRow.step` names which node execution a live
 * gateway row (no `seq`) was produced during -- both stamped server-side
 * (`web/stream.py`), so this module no longer *infers* either one the way
 * its first version had to (see git history / `docs/trace-inspector-plan.md`
 * §7.2 for what that heuristic was and why it was replaced: a gateway row
 * always arrives before the node it belongs to has flushed anything, which
 * made "attach it to the next span to open" the only heuristic available
 * before `step` existed, and briefly a source of real bugs -- naming a span
 * after a gateway row that happened to arrive first).
 *
 * What is still inferred, and has to be: prelude rows (no `ContextEvent` to
 * read structured history/memory/contract from -- see `ExecutionTree.
 * prelude`'s own doc) and a still-open node's name (no step has arrived for
 * it yet to read `.node` off).
 */
export function buildExecutionTree(view: TurnView): ExecutionTree {
  let prelude: TraceRow[] = []
  const entries: ExecutionEntry[] = []
  let pending: TraceRow[] = []
  const gatewayByStep = new Map<number, TraceRow[]>()
  let sawFirstBatch = false
  let sawAnyStep = false

  for (const item of view.timeline) {
    if (item.kind === 'row') {
      if (item.row.source === 'tool_gateway' && item.row.step !== null) {
        const bucket = gatewayByStep.get(item.row.step)
        if (bucket) bucket.push(item.row)
        else gatewayByStep.set(item.row.step, [item.row])
      } else {
        pending.push(item.row)
      }
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
      kind: item.step.node !== null ? 'node' : 'engine',
      ordinal: item.step.step_count,
      // A real node span is named by the step itself, always correctly now
      // (no more guessing from row order). A bare entry has no node to be
      // named by -- fall back to its own first row's `node`, e.g.
      // `approval_recorded`'s row says "approval" even though nothing
      // about it is a graph node execution.
      nodeName: item.step.node ?? rows[0]?.node ?? null,
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
      entries.push({
        kind: 'node',
        ordinal: 1,
        nodeName: openEntryNodeName(split.rest),
        rows: split.rest,
        step: null,
      })
    } else if (pending.some(isNodeEntered)) {
      const lastOrdinal = entries.length > 0 ? entries[entries.length - 1].ordinal : 0
      entries.push({
        kind: 'node',
        ordinal: lastOrdinal + 1,
        nodeName: openEntryNodeName(pending),
        rows: pending,
        step: null,
      })
    } else {
      consolidation = pending
    }
  }

  // Attach every live gateway row to the node execution it was stamped
  // for -- always a `node` entry (the tool gateway only ever runs
  // synchronously inside `call_tool`'s own node, never during a bare
  // engine-level step), matched by step_count rather than array position
  // so a bare engine entry sitting between two nodes can never shift the
  // match.
  for (const entry of entries) {
    if (entry.kind !== 'node') continue
    const bucket = gatewayByStep.get(entry.ordinal)
    if (bucket) entry.rows = [...bucket, ...entry.rows]
  }

  return { prelude, entries, consolidation }
}
