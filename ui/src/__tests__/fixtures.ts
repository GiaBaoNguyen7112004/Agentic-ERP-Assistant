// Shared builders for the protocol's two widest event shapes, so a field
// added to either one is a default added here once -- not a missing-field
// type error rediscovered independently in every test file that builds one.

import type { StepEvent, TraceRow } from '../protocol'

export function row(overrides: Partial<TraceRow> = {}): TraceRow {
  return {
    type: 'trace',
    seq: 0,
    node: 'think',
    kind: 'route_selected',
    detail: '',
    source: 'engine',
    step: null,
    ...overrides,
  }
}

export function step(overrides: Partial<StepEvent> = {}): StepEvent {
  return {
    type: 'step',
    route: null,
    tool_name: null,
    tool_arguments: null,
    tool_mutating: null,
    approval: 'not_required',
    step_count: 1,
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
