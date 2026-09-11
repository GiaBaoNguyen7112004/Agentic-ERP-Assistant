import type {
  AnswerEvent,
  ApprovalRequiredEvent,
  ServerEvent,
  StepEvent,
  TraceRow,
  TurnFinishedEvent,
} from './protocol'

export type TurnStatus = 'starting' | 'running' | 'paused' | 'done' | 'error'

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
  /** `step` events kept only when the route changed from the previous one
   * kept -- the decision sequence, not every intermediate observation. */
  decisions: StepEvent[]
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
  decisions: [],
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
      return { ...view, events: [...view.events, event] }

    case 'step': {
      const last = view.decisions[view.decisions.length - 1]
      if (last && last.route === event.route) {
        return view
      }
      return { ...view, decisions: [...view.decisions, event] }
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
