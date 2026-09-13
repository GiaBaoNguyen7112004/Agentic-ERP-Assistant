// Rebuilding a TurnView from a filed run (docs/trace-inspector-plan.md §6.3,
// Phase 4): a turn loaded from session history -- or an approval decided from
// the queue, whose fresh bubble never saw the first half -- carries no trace
// events, so the panel hydrates it from GET /api/runs/{trace_id} instead of
// showing an empty tree.
//
// The reconstruction is honest about what a filed run cannot say back. A live
// stream carries one `step` per node execution with its payload deltas; a
// filed run carries only the event log and the FINAL state, so this module
// re-derives the steps from the `node_entered`/`node_exited` spans and
// attaches payloads by a documented best-effort rule -- never by claiming
// something the record does not say:
//
//   evidence     -> the `retrieve_project_documents` span (there is at most
//                   one per turn; the snippets are the filed state's, once)
//   observations -> one per `call_tool` span in order, then one per `think`
//                   span holding an `approval_requested` row (a preflight
//                   escalation's refusal); anything left over goes to an
//                   explicit `unattributed` bucket at run level -- never onto
//                   the wrong span
//   model calls  -> run level (the summary strip); no filed row says which
//                   node made a call, so no span claims one
//   tool call    -> the filed state's `tool_name`/`tool_arguments` name the
//                   LAST call the turn made, so they decorate the last
//                   `call_tool` span and nothing else
//   response     -> the filed reply closes the LAST span, terminal, exactly
//                   where a live terminal step closes the turn
//
// Rows the engine files outside any span (prelude recall/declaration rows,
// consolidation write/reject rows) stay stepless on purpose:
// `lib/executionTree.ts` files them as the prelude and consolidation phases
// from the same timeline shape a live turn produces, so both views build
// through the one tree builder.

import type {
  AnswerEvent,
  ContextEvent,
  EvidenceOut,
  MemoryOut,
  RunReport,
  StepEvent,
  ToolOutcome,
  ToolOutcomeOut,
  TraceRow,
  TurnFinishedEvent,
} from './protocol'
import { initialTurn, type TimelineEntry, type TurnView } from './turnReducer'

export interface HydratedTurn {
  turn: TurnView
  /** Tool outcomes the attribution rule above could not place on any span.
   * Empty for every ordinary run; non-empty only when a run's shape defeats
   * the rule (the panel says so, next to its reconstruction banner). */
  unattributed: ToolOutcomeOut[]
}

interface Span {
  /** The `node_entered` row's own name, or `null` for an engine-level card
   * (an `approval_recorded`/`run_failed` row filed outside any span). */
  nodeName: string | null
  rows: TraceRow[]
  /** The route a `route_selected` row inside the span chose, when one did. */
  route: string | null
}

const NODE_ENTERED = 'node_entered'
const NODE_EXITED = 'node_exited'
const ENGINE_CARD_NODES = new Set(['approval', 'engine'])

/** `route_selected`'s detail is "<route>: <rationale>"; the part before the
 * first colon is the route. The rationale is not parsed here -- it is the
 * row itself, which the card already renders. */
function routeFrom(rows: readonly TraceRow[]): string | null {
  for (let i = rows.length - 1; i >= 0; i -= 1) {
    const item = rows[i]
    if (item.kind !== 'route_selected') continue
    const colon = item.detail.indexOf(':')
    return colon === -1 ? item.detail : item.detail.slice(0, colon)
  }
  return null
}

/** Walk the filed event log the way the engine ran it: a `node_entered`
 * opens a span, the matching `node_exited` closes it, and a row filed
 * outside any span is either an engine card (`approval`, `engine`) or a
 * prelude/consolidation row the tree builder files stepless. Returns the
 * spans in run order and, parallel to `rows`, which span (if any) each row
 * belongs to -- so the caller can interleave rows and closing steps without
 * walking the log a second time. */
export function spansFromEvents(
  rows: readonly TraceRow[],
): { spans: Span[]; rowSpans: (number | null)[] } {
  const spans: Span[] = []
  const rowSpans: (number | null)[] = []
  let open: { nodeName: string; rows: TraceRow[] } | null = null

  const close = () => {
    if (open === null) return
    for (let i = 0; i < open.rows.length; i += 1) rowSpans.push(spans.length)
    spans.push({ nodeName: open.nodeName, rows: open.rows, route: routeFrom(open.rows) })
    open = null
  }

  for (const row of rows) {
    if (row.kind === NODE_ENTERED) {
      close()
      open = { nodeName: row.node, rows: [row] }
      continue
    }
    if (row.kind === NODE_EXITED) {
      if (open) open.rows.push(row)
      close()
      continue
    }
    if (open) {
      open.rows.push(row)
      continue
    }
    if (ENGINE_CARD_NODES.has(row.node)) {
      // An approval decision recorded on resume, or the loop guard: a real
      // state change with no node span around it -- an engine card, the way
      // a live `step` with `node: null` becomes one.
      rowSpans.push(spans.length)
      spans.push({ nodeName: null, rows: [row], route: null })
      continue
    }
    // Prelude or consolidation: no step closes these, by design.
    rowSpans.push(null)
  }
  close()
  return { spans, rowSpans }
}

function baseStep(stepCount: number, overrides: Partial<StepEvent>): StepEvent {
  return {
    type: 'step',
    route: null,
    tool_name: null,
    tool_arguments: null,
    tool_mutating: null,
    approval: 'not_required',
    step_count: stepCount,
    terminal: false,
    node: null,
    elapsed_ms: 0,
    evidence: null,
    observations: [],
    response: null,
    failure: 'none',
    error_detail: null,
    draft: null,
    redirected_needs: [],
    retry_count: 0,
    retrieval: null,
    model_calls: [],
    ...overrides,
  }
}

function evidenceOut(state: RunReport['run']['state']): EvidenceOut[] {
  return state.evidence.map((snippet) => ({
    source_id: snippet.source_id,
    locator: snippet.locator,
    tag: `[${snippet.source_id}#${snippet.locator}]`,
    text: { text: snippet.text, truncated: false, chars: snippet.text.length },
  }))
}

function observationsOut(state: RunReport['run']['state']): ToolOutcomeOut[] {
  return state.observations.map((outcome: ToolOutcome) => ({
    tool_name: outcome.tool_name,
    arguments_summary: outcome.arguments_summary,
    status: outcome.status,
    summary: { text: outcome.summary, truncated: false, chars: outcome.summary.length },
    source_ids: outcome.source_ids,
    error: outcome.error,
    attempts: outcome.attempts,
    retry_after_seconds: outcome.retry_after_seconds,
  }))
}

function memoriesOut(state: RunReport['run']['state']): MemoryOut[] {
  return state.memories.map((record) => ({
    memory_id: record.memory_id,
    kind: record.kind,
    key: record.key,
    statement: record.statement,
    confidence: record.confidence,
    recorded_in_run: record.recorded_in_run,
    recorded_at: record.recorded_at,
    supersedes: record.supersedes,
    links: record.links,
    required_scope: record.required_scope,
    actor: record.actor,
    project_code: record.project_code,
    session_id: record.session_id,
  }))
}

/**
 * The one way a filed run becomes a `TurnView` -- the same shape a live turn
 * built from the stream, with whoever renders it labelling the attribution
 * as reconstructed (the panel shows the banner; what this module could not
 * place it says in `unattributed`).
 */
export function hydrateTurn(report: RunReport): HydratedTurn {
  const state = report.run.state
  const rows: TraceRow[] = report.events.map((event) => ({
    type: 'trace',
    seq: null,
    node: event.node,
    kind: event.kind,
    detail: event.detail,
    source: 'engine',
    step: null,
  }))

  const { spans, rowSpans } = spansFromEvents(rows)
  const evidence = evidenceOut(state)
  const queue = [...observationsOut(state)]
  const lastCallTool = [...spans].reverse().find((span) => span.nodeName === 'call_tool') ?? null

  // Steps, one per span, numbered the way the engine numbers them (real node
  // executions only; engine cards stay unnumbered, ordinal 0).
  const steps: StepEvent[] = spans.map((span) =>
    baseStep(0, {
      node: span.nodeName,
      route: span.route,
      evidence: span.nodeName === 'retrieve_project_documents' ? evidence : null,
    }),
  )
  let ordinal = 0
  steps.forEach((_, index) => {
    if (spans[index].nodeName !== null) {
      ordinal += 1
      steps[index].step_count = ordinal
    }
  })

  // The interleaved timeline, in run order: each span's rows followed by the
  // step that closes it, leftover (prelude/consolidation) rows stepless --
  // the exact shape a live turn's reducer keeps. A span's rows are
  // contiguous in the event log, so "the next row belongs to a different
  // span (or none, or there is none)" is exactly where its closing step goes.
  const timeline: TimelineEntry[] = []
  rows.forEach((row, index) => {
    timeline.push({ kind: 'row', row })
    const spanIndex = rowSpans[index]
    if (spanIndex === null) return
    const next = rowSpans[index + 1]
    if (next !== spanIndex) timeline.push({ kind: 'step', step: steps[spanIndex] })
  })

  // -- payload attribution, exactly as the header comment documents --------
  spans.forEach((span, index) => {
    if (span.nodeName === 'call_tool') {
      const next = queue.shift()
      if (next) steps[index].observations = [next]
    }
  })
  // A think span that escalated to a human takes the next leftover outcome,
  // in order -- the refusal an approver's denial produced.
  spans.forEach((span, index) => {
    if (span.nodeName !== 'think' && span.nodeName !== 'start') return
    if (!span.rows.some((row) => row.kind === 'approval_requested')) return
    if (steps[index].observations.length > 0) return
    const next = queue.shift()
    if (next) steps[index].observations = [next]
  })
  const unattributed = queue

  if (lastCallTool !== null && state.tool_name !== null) {
    const index = spans.indexOf(lastCallTool)
    steps[index].tool_name = state.tool_name
    steps[index].tool_arguments = state.tool_arguments
    steps[index].tool_mutating = state.tool_mutating
  }

  // The filed reply closes the last span, terminal -- where a live terminal
  // step closes the turn.
  const lastIndex = spans.length - 1
  if (lastIndex >= 0) {
    steps[lastIndex].response = report.answer.text || state.response
    steps[lastIndex].failure = state.failure
    steps[lastIndex].error_detail = state.error_detail
    steps[lastIndex].terminal = state.terminal
  }

  const context: ContextEvent = {
    type: 'context',
    request: state.request,
    history: state.history,
    memories: memoriesOut(state),
    contract: state.contract
      ? { needs: [...state.contract.needs], document_query: state.contract.document_query }
      : null,
    // Model calls are run-level in a reconstruction: no filed row names the
    // node that made it, so none is claimed here -- the records ride the
    // summary (`summary.model_calls`), which is where "run level" shows.
    model_calls: [],
  }

  const answer: AnswerEvent = {
    type: 'answer',
    text: report.answer.text,
    route: state.route,
    failure: state.failure,
    error_detail: state.error_detail,
    citations: report.answer.citations,
  }

  const summary: TurnFinishedEvent = {
    type: 'turn_finished',
    outcome: report.run.outcome === 'paused' ? 'paused' : 'terminal',
    route: state.route,
    failure: state.failure,
    step_count: state.step_count,
    evidence: [...new Set(state.evidence.map((snippet) => snippet.source_id))],
    memories_recalled: state.memories.length,
    history_shown: state.history.length,
    observations: state.observations.map((outcome) => ({
      tool: outcome.tool_name,
      status: outcome.status,
      attempts: outcome.attempts,
    })),
    model_calls: report.model_calls,
    memory_audit: report.memory_audit,
    started_at: report.run.started_at,
    finished_at: report.run.finished_at,
  }

  const turn: TurnView = {
    ...initialTurn,
    traceId: report.run.trace_id,
    status: report.run.outcome === 'paused' ? 'paused' : 'done',
    streamed: '',
    answer,
    approval: null,
    context,
    events: rows,
    steps,
    timeline,
    summary,
    error: null,
  }
  return { turn, unattributed }
}