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

export interface TextOut {
  text: string
  truncated: boolean
  chars: number
}

export interface HistoryTurnOut {
  trace_id: string
  request: string
  response: string | null
  route: string | null
  failure: string
  tool_name: string | null
  approval: string
  started_at: string
  finished_at: string
}

export interface MemoryOut {
  memory_id: string
  kind: string
  key: string
  statement: string
  confidence: number
  recorded_in_run: string
  recorded_at: string
  supersedes: string[]
  links: string[]
  required_scope: string
  actor: string
  project_code: string
  session_id: string
}

export interface ContractOut {
  needs: string[]
  document_query: string | null
}

export interface EvidenceOut {
  source_id: string
  locator: string
  tag: string
  text: TextOut
}

export interface ToolOutcomeOut {
  tool_name: string
  arguments_summary: string
  status: string
  summary: TextOut
  source_ids: string[]
  error: string | null
  attempts: number
  retry_after_seconds: number | null
}

export interface MemoryAuditOut {
  occurred_at: string
  memory_id: string
  kind: string
  decision: string
  rejection: string | null
  reason: string
  statement_summary: string
}

export interface MessageOut {
  role: string
  content: TextOut
}

export interface ModelRequestOut {
  kind: 'answer' | 'tools'
  messages: MessageOut[]
  tools: string[]
  tool_choice: string | null
  temperature: number
}

export interface ModelResponseOut {
  content: TextOut | null
  tool_name: string | null
  arguments: Record<string, unknown> | null
  stop_reason: string | null
}

export interface RetrievalHitOut {
  chunk_id: string
  document_id: string
  locator: string
  title: string
  score: number
  ranks: Record<string, number>
  scores: Record<string, number>
}

export interface RetrievalOut {
  query: string
  limit: number
  hits: RetrievalHitOut[]
  best_similarity: number | null
  minimum_similarity: number
  dense_candidates: number
  lexical_candidates: number
  gated: boolean
}

export interface ModelCallOut {
  model: string
  outcome: string
  estimated_input_tokens: number
  input_tokens: number
  output_tokens: number
  cost_usd: number | null
  latency_seconds: number
  attempts: number
  occurred_at: string
  detail: string | null
  request: ModelRequestOut | null
  response: ModelResponseOut | null
}

export interface ModelCallTotals {
  count: number
  input_tokens: number
  output_tokens: number
  cost_usd: number | null
  unpriced: number
  records: ModelCallOut[]
}

export interface TurnStartedEvent {
  type: 'turn_started'
  trace_id: string
  session_id: string | null
  actor: string
  resumed: boolean
}

export interface ContextEvent {
  type: 'context'
  request: string
  history: HistoryTurnOut[]
  memories: MemoryOut[]
  contract: ContractOut | null
  model_calls: ModelCallOut[]
  // Always sent live -- the state the engine is about to run from. `null`
  // only on a view hydrated from a filed run, which files no starting
  // state (docs/agent-state-inspector-plan.md D4/D5); the server-side
  // field is required (never actually null on the wire).
  state: AgentStateSnapshot | null
}

export interface TraceRow {
  type: 'trace'
  seq: number | null
  node: string
  kind: string
  detail: string
  source: 'engine' | 'tool_gateway'
  step: number | null
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
  node: string | null
  elapsed_ms: number
  evidence: EvidenceOut[] | null
  observations: ToolOutcomeOut[]
  response: string | null
  failure: string
  error_detail: string | null
  draft: string | null
  redirected_needs: string[]
  retry_count: number
  retrieval: RetrievalOut | null
  model_calls: ModelCallOut[]
  // Always sent live -- the whole state this node returned. `null` only on
  // a step hydrated from a filed run other than the last one (the record
  // holds the final state only -- see hydrate.ts); the server-side field
  // is required (never actually null on the wire).
  state: AgentStateSnapshot | null
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
  memory_audit: MemoryAuditOut[]
  started_at: string
  finished_at: string
}

export interface ErrorEvent {
  type: 'error'
  message: string
}

export type ServerEvent =
  | TurnStartedEvent
  | ContextEvent
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
  'context',
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

// -- the filed-run report, mirroring web/protocol.py's RunReportOut (and the
// state models it embeds), so a filed run hydrates into the same TurnView a
// live turn built from the stream -- see ui/src/hydrate.ts. Every interface
// below is drift-tested field-for-field against its Python model in
// tests/web/test_protocol_drift.py. -----------------------------------------

export interface TraceEvent {
  node: string
  kind: string
  detail: string
}

export interface EvidenceSnippet {
  source_id: string
  locator: string
  text: string
}

export interface MemoryRecord {
  memory_id: string
  kind: string
  key: string
  statement: string
  project_code: string
  required_scope: string
  actor: string
  session_id: string
  recorded_in_run: string
  recorded_at: string
  confidence: number
  supersedes: string[]
  superseded_at: string | null
  links: string[]
}

export interface ConversationTurn {
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

export interface ReplyContract {
  needs: string[]
  document_query: string | null
}

export interface ToolOutcome {
  tool_name: string
  arguments_summary: string
  status: string
  summary: string
  source_ids: string[]
  error: string | null
  attempts: number
  retry_after_seconds: number | null
}

export interface AgentStateSnapshot {
  request: string
  actor: string
  project_code: string
  scopes: string[]
  trace_id: string
  session_id: string | null
  route: string | null
  evidence: EvidenceSnippet[]
  memories: MemoryRecord[]
  history: ConversationTurn[]
  contract: ReplyContract | null
  redirected_needs: string[]
  draft: string | null
  observations: ToolOutcome[]
  tool_name: string | null
  tool_arguments: Record<string, unknown> | null
  tool_mutating: boolean
  approval: string
  response: string | null
  failure: string
  error_detail: string | null
  terminal: boolean
  step_count: number
  retry_count: number
  events: TraceEvent[]
  state_version: number
}

export interface AuditRow {
  trace_id: string
  occurred_at: string
  actor: string
  tool_name: string
  arguments_summary: string
  approval: string
  status: string
  source_ids: string[]
}

export interface RunOut {
  trace_id: string
  actor: string
  project_code: string | null
  outcome: string
  started_at: string
  finished_at: string
  state: AgentStateSnapshot
}

export interface RunAnswerOut {
  text: string
  citations: Citation[]
}

export interface RunReport {
  run: RunOut
  events: TraceEvent[]
  audit_rows: AuditRow[]
  model_calls: ModelCallTotals
  memory_audit: MemoryAuditOut[]
  answer: RunAnswerOut
}
