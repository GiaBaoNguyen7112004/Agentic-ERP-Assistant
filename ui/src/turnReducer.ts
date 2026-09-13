import type {
  AnswerEvent,
  ApprovalRequiredEvent,
  ServerEvent,
  StepEvent,
  TraceRow,
  TurnFinishedEvent,
} from './protocol'

export type TurnStatus = 'starting' | 'running' | 'paused' | 'done' | 'error'

/**
 * One arrival on the trace half of the stream, in the exact order it was
 * received. `events` and `steps` below are convenience projections of this
 * same sequence -- kept because the raw events table and the reducer's own
 * tests read them directly -- but `timeline` is the one array that still
 * says which trace rows arrived *before* a given step event, which a flat
 * `events`/`steps` split cannot: a node's own rows and any live gateway rows
 * around it interleave with the `step` events that close each node
 * execution, and `lib/executionTree.ts` groups rows into node spans by
 * walking exactly this interleaving (see that module for why: a step can
 * close a batch of zero new trace rows, e.g. a denied approval's refusal,
 * and nothing about `events.length` at that point says so on its own).
 */
export type TimelineEntry = { kind: 'row'; row: TraceRow } | { kind: 'step'; step: StepEvent }

export interface TurnView {
  traceId: string | null
  status: TurnStatus
  /** The preview -- appended to by `token`, cleared by `reset`. Once
   * `answer` is set, a renderer must show `answer.text`, never this (D4). */
  streamed: string
  answer: AnswerEvent | null
  approval: ApprovalRequiredEvent | null
  /** Every trace row, engine and gateway, in arrival order. */
  events: TraceRow[]
  /** Every `step` event, in arrival order -- not deduplicated by route (the
   * execution tree needs one entry per node execution, not one per distinct
   * route reached). */
  steps: StepEvent[]
  /** `events` and `steps`, interleaved as they arrived. See the type doc. */
  timeline: TimelineEntry[]
  summary: TurnFinishedEvent | null
  error: string | null
}

export const initialTurn: TurnView = {
  traceId: null,
  status: 'starting',
  streamed: '',
  answer: null,
  approval: null,
  events: [],
  steps: [],
  timeline: [],
  summary: null,
  error: null,
}

/**
 * The whole client-side interpretation of a turn, as one pure function.
 * Every other place in this app that shows a turn reads a `TurnView`; only
 * this function ever builds one from the raw event stream.
 */
export function turnReducer(view: TurnView, event: ServerEvent): TurnView {
  switch (event.type) {
    case 'turn_started':
      return {
        // A resumed turn keeps everything the client already saw about the
        // first half; a fresh chat starts a clean view even if this
        // function is reused across turns in the same session.
        ...(event.resumed ? view : initialTurn),
        traceId: event.trace_id,
        status: 'running',
        approval: event.resumed ? null : view.approval,
      }

    case 'trace':
      return {
        ...view,
        events: [...view.events, event],
        timeline: [...view.timeline, { kind: 'row', row: event }],
      }

    case 'step':
      return {
        ...view,
        steps: [...view.steps, event],
        timeline: [...view.timeline, { kind: 'step', step: event }],
      }

    case 'token':
      return { ...view, streamed: view.streamed + event.text }

    case 'reset':
      return { ...view, streamed: '' }

    case 'approval_required':
      return { ...view, approval: event, status: 'paused' }

    case 'answer':
      return { ...view, answer: event, status: 'done' }

    case 'turn_finished':
      return { ...view, summary: event }

    case 'error':
      return { ...view, status: 'error', error: event.message }

    default:
      return view
  }
}
