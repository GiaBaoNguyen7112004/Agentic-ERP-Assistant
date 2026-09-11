import { describe, expect, it } from 'vitest'
import { initialTurn, turnReducer } from '../turnReducer'
import type { ServerEvent } from '../protocol'

function apply(events: ServerEvent[]) {
  return events.reduce(turnReducer, initialTurn)
}

describe('token / reset', () => {
  it('appends tokens to the streamed preview, in order', () => {
    const view = apply([
      { type: 'token', text: 'Hel' },
      { type: 'token', text: 'lo' },
    ])
    expect(view.streamed).toBe('Hello')
  })

  it('reset clears the streamed preview', () => {
    const view = apply([
      { type: 'token', text: 'Hel' },
      { type: 'reset' },
      { type: 'token', text: 'Hi' },
    ])
    expect(view.streamed).toBe('Hi')
  })
})

describe('answer replaces the streamed preview (D4)', () => {
  it('once answer arrives, the authoritative text is what a renderer must show', () => {
    const view = apply([
      { type: 'token', text: 'Prev' },
      { type: 'token', text: 'iew' },
      {
        type: 'answer',
        text: 'The real answer.',
        route: 'answer',
        failure: 'none',
        error_detail: null,
        citations: [],
      },
    ])
    expect(view.answer?.text).toBe('The real answer.')
    expect(view.status).toBe('done')
    // The preview is still there for a caller who wants it, but the D4 rule
    // is enforced by whichever component renders the view, not by clearing
    // `streamed` here -- a reset mid-retry must not wipe an answer that
    // already arrived on some other path.
    expect(view.streamed).toBe('Preview')
  })
})

describe('decision de-duplication', () => {
  const step = (route: string): ServerEvent => ({
    type: 'step',
    route,
    tool_name: null,
    tool_arguments: null,
    tool_mutating: null,
    approval: 'not_required',
    step_count: 1,
    terminal: false,
  })

  it('keeps only the steps whose route differs from the previous one kept', () => {
    const view = apply([step('think'), step('think'), step('call_tool'), step('call_tool'), step('answer')])
    expect(view.decisions.map((d) => d.route)).toEqual(['think', 'call_tool', 'answer'])
  })

  it('an empty stream has no decisions', () => {
    expect(apply([]).decisions).toEqual([])
  })
})

describe('approval_required', () => {
  it('sets approval and status paused', () => {
    const view = apply([
      {
        type: 'approval_required',
        trace_id: 'run-1',
        tool_name: 'create_risk',
        arguments: { project_id: 'atlas' },
        summary: 'create_risk(project_id=atlas)',
        actor: 'priya',
      },
    ])
    expect(view.status).toBe('paused')
    expect(view.approval?.tool_name).toBe('create_risk')
  })
})

describe('turn_started: paused then resumed', () => {
  it('a second turn_started with resumed:true keeps the earlier events and decisions', () => {
    const view = apply([
      { type: 'turn_started', trace_id: 'run-1', session_id: 's1', actor: 'priya', resumed: false },
      { type: 'trace', seq: 0, node: 'start', kind: 'node_entered', detail: '', source: 'engine' },
      {
        type: 'approval_required',
        trace_id: 'run-1',
        tool_name: 'create_risk',
        arguments: {},
        summary: 'create_risk()',
        actor: 'priya',
      },
      { type: 'turn_started', trace_id: 'run-1', session_id: 's1', actor: 'priya', resumed: true },
      { type: 'trace', seq: 1, node: 'approval', kind: 'approval_recorded', detail: 'approved by priya', source: 'engine' },
    ])

    expect(view.events).toHaveLength(2)
    expect(view.status).toBe('running')
    // The approval card is no longer waiting on anyone once resumed.
    expect(view.approval).toBeNull()
  })

  it('a fresh (non-resumed) turn_started resets everything', () => {
    const view = apply([
      { type: 'token', text: 'stale' },
      { type: 'turn_started', trace_id: 'run-2', session_id: 's1', actor: 'priya', resumed: false },
    ])
    expect(view.streamed).toBe('')
    expect(view.traceId).toBe('run-2')
  })
})

describe('turn_finished and error', () => {
  it('turn_finished sets summary without changing status', () => {
    const view = apply([
      {
        type: 'answer',
        text: 'done',
        route: 'answer',
        failure: 'none',
        error_detail: null,
        citations: [],
      },
      {
        type: 'turn_finished',
        outcome: 'terminal',
        route: 'answer',
        failure: 'none',
        step_count: 2,
        evidence: [],
        memories_recalled: 0,
        history_shown: 0,
        observations: [],
        model_calls: { count: 1, input_tokens: 10, output_tokens: 5, cost_usd: 0.01, unpriced: 0 },
      },
    ])
    expect(view.status).toBe('done')
    expect(view.summary?.outcome).toBe('terminal')
  })

  it('error sets status error and carries the message', () => {
    const view = apply([{ type: 'error', message: 'boom' }])
    expect(view.status).toBe('error')
    expect(view.error).toBe('boom')
  })
})
