import { describe, expect, it } from 'vitest'
import { initialTurn, turnReducer } from '../turnReducer'
import type { ServerEvent } from '../protocol'
import { row, step as stepFixture } from './fixtures'

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

describe('steps', () => {
  const step = (route: string, stepCount = 1): ServerEvent => stepFixture({ route, step_count: stepCount })

  it('keeps every step, including repeats of the same route', () => {
    // Unlike the old deduplicated `decisions` list, the execution tree needs
    // one entry per node execution -- two consecutive `think` steps are two
    // different node runs, not one decision shown twice.
    const view = apply([step('think', 1), step('think', 2), step('call_tool', 3)])
    expect(view.steps.map((s) => s.route)).toEqual(['think', 'think', 'call_tool'])
  })

  it('an empty stream has no steps', () => {
    expect(apply([]).steps).toEqual([])
  })
})

describe('timeline', () => {
  it('interleaves trace rows and steps in exact arrival order', () => {
    const tick = (seq: number): ServerEvent => row({ seq, detail: 'answer' })
    const closing: ServerEvent = stepFixture({ route: 'answer', terminal: true })
    const view = apply([tick(0), tick(1), closing, tick(2)])
    expect(view.timeline.map((entry) => entry.kind)).toEqual(['row', 'row', 'step', 'row'])
  })
})

describe('context', () => {
  it('sets view.context', () => {
    const view = apply([
      {
        type: 'context',
        request: 'Why is milestone M2 late?',
        history: [],
        memories: [],
        contract: { needs: ['document_passage'], document_query: 'why milestone M2 is late' },
      },
    ])
    expect(view.context?.contract?.needs).toEqual(['document_passage'])
  })

  it('a second context event (a resumed stream) overwrites the first', () => {
    const view = apply([
      { type: 'context', request: 'first', history: [], memories: [], contract: null },
      { type: 'context', request: 'second', history: [], memories: [], contract: null },
    ])
    expect(view.context?.request).toBe('second')
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
      row({ seq: 0, node: 'start', kind: 'node_entered', detail: '' }),
      {
        type: 'approval_required',
        trace_id: 'run-1',
        tool_name: 'create_risk',
        arguments: {},
        summary: 'create_risk()',
        actor: 'priya',
      },
      { type: 'turn_started', trace_id: 'run-1', session_id: 's1', actor: 'priya', resumed: true },
      row({ seq: 1, node: 'approval', kind: 'approval_recorded', detail: 'approved by priya' }),
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
        model_calls: {
          count: 1, input_tokens: 10, output_tokens: 5, cost_usd: 0.01, unpriced: 0, records: [],
        },
        memory_audit: [],
        started_at: '2026-01-01T00:00:00Z',
        finished_at: '2026-01-01T00:00:00Z',
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
