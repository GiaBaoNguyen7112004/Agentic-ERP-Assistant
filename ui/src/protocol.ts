// Mirrors src/agentic_erp_assistant/web/protocol.py field for field.
//
// tests/web/test_protocol_drift.py reads this file and fails the Python
// suite the day the two disagree -- a renamed event fails in `pytest`, not
// in a browser console. Keep every `type: "..."` literal below in exact sync
// with EVENT_TYPES in protocol.py, and add nothing here that is not also a
// field on the corresponding pydantic model.

export interface Citation {
  source_id: string
  locator: string | null
  tag: string
  kind: 'document' | 'erp'
}

export interface Observation {
  tool: string
  status: string
  attempts: number
}

export interface ModelCallTotals {
  count: number
  input_tokens: number
  output_tokens: number
  cost_usd: number | null
  unpriced: number
}

export interface TurnStartedEvent {
  type: 'turn_started'
  trace_id: string
  session_id: string | null
  actor: string
  resumed: boolean
}

export interface TraceRow {
  type: 'trace'
  seq: number | null
  node: string
  kind: string
  detail: string
  source: 'engine' | 'tool_gateway'
}

export interface StepEvent {
  type: 'step'
  route: string | null
  tool_name: string | null
  tool_arguments: Record<string, unknown> | null
  tool_mutating: boolean | null
  approval: string
  step_count: number
  terminal: boolean
}

export interface TokenEvent {
  type: 'token'
  text: string
}

export interface ResetEvent {
  type: 'reset'
}

export interface ApprovalRequiredEvent {
  type: 'approval_required'
  trace_id: string
  tool_name: string
  arguments: Record<string, unknown>
  summary: string
  actor: string
}

export interface AnswerEvent {
  type: 'answer'
  text: string
  route: string | null
  failure: string
  error_detail: string | null
  citations: Citation[]
}

export interface TurnFinishedEvent {
  type: 'turn_finished'
  outcome: 'paused' | 'terminal'
  route: string | null
  failure: string
  step_count: number
  evidence: string[]
  memories_recalled: number
  history_shown: number
  observations: Observation[]
  model_calls: ModelCallTotals
}

export interface ErrorEvent {
  type: 'error'
  message: string
}

export type ServerEvent =
  | TurnStartedEvent
  | TraceRow
  | StepEvent
  | TokenEvent
  | ResetEvent
  | ApprovalRequiredEvent
  | AnswerEvent
  | TurnFinishedEvent
  | ErrorEvent

export const EVENT_TYPES: readonly ServerEvent['type'][] = [
  'turn_started',
  'trace',
  'step',
  'token',
  'reset',
  'approval_required',
  'answer',
  'turn_finished',
  'error',
] as const

// -- the JSON routes, mirroring web/app.py's response shapes ---------------

export interface UserSummary {
  actor: string
  display_name: string
  role: string
  project_code: string
  scopes: string[]
  can_approve: boolean
}

export interface SessionSummary {
  session_id: string
  first_request: string
  last_started_at: string
}

export interface SessionTurn {
  trace_id: string
  session_id: string
  actor: string
  request: string
  response: string | null
  route: string | null
  failure: string
  tool_name: string | null
  approval: string
  started_at: string
  finished_at: string
}

export interface PendingApproval {
  trace_id: string
  actor: string
  tool_name: string
  arguments_summary: string
  created_at: string
  session_id: string | null
}
