import { describe, expect, it } from 'vitest'
import { transitionReason } from '../lib/transitionReason'
import type { TraceRow } from '../protocol'

function row(kind: string, detail: string): TraceRow {
  return { type: 'trace', seq: 0, node: 'think', kind, detail, source: 'engine', step: null }
}

describe('transitionReason', () => {
  it('prefers route_selected over every other kind', () => {
    const reason = transitionReason([
      row('tool_called', 'get_project_status -> ok'),
      row('route_selected', 'answer: answered without calling a tool'),
    ])
    expect(reason).toBe('answer: answered without calling a tool')
  })

  it('falls back to contract_enforced when there is no route_selected', () => {
    const reason = transitionReason([
      row('failed', 'planner raised RuntimeError'),
      row('contract_enforced', 'erp_field: a call was required, search withheld'),
    ])
    expect(reason).toBe('erp_field: a call was required, search withheld')
  })

  it('falls back through tool_called, failed, rate_limited, approval_requested in order', () => {
    expect(transitionReason([row('tool_called', 'X -> ok')])).toBe('X -> ok')
    expect(transitionReason([row('failed', 'boom')])).toBe('boom')
    expect(transitionReason([row('rate_limited', 'throttled')])).toBe('throttled')
    expect(transitionReason([row('approval_requested', 'needs a human')])).toBe('needs a human')
  })

  it('picks the most recent match among rows of the same kind', () => {
    const reason = transitionReason([
      row('route_selected', 'first'),
      row('tool_called', 'unrelated'),
      row('route_selected', 'second'),
    ])
    expect(reason).toBe('second')
  })

  it('returns null when no row matches any known kind', () => {
    expect(transitionReason([row('node_entered', '')])).toBeNull()
  })

  it('returns null for an empty row list', () => {
    expect(transitionReason([])).toBeNull()
  })
})
