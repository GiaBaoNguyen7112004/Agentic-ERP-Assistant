import { describe, expect, it } from 'vitest'
import {
  changedFields,
  deepEqual,
  diffState,
  STATE_FIELD_GROUPS,
  type StateField,
} from '../lib/stateDiff'
import type { AgentStateSnapshot } from '../protocol'
import { agentState } from './fixtures'

describe('STATE_FIELD_GROUPS', () => {
  it('covers every field of AgentStateSnapshot exactly once', () => {
    const fullState = agentState()
    const expected = Object.keys(fullState).sort()
    const grouped = STATE_FIELD_GROUPS.flatMap((group) => group.fields).sort()
    expect(grouped).toEqual(expected)
  })

  it('lists no field twice', () => {
    const grouped = STATE_FIELD_GROUPS.flatMap((group) => group.fields)
    expect(new Set(grouped).size).toBe(grouped.length)
  })
})

describe('deepEqual', () => {
  it('treats primitives, arrays, and plain objects structurally', () => {
    expect(deepEqual(1, 1)).toBe(true)
    expect(deepEqual('a', 'b')).toBe(false)
    expect(deepEqual([1, 2], [1, 2])).toBe(true)
    expect(deepEqual([1, 2], [2, 1])).toBe(false)
    expect(deepEqual({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(true)
    expect(deepEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false)
    expect(deepEqual(null, null)).toBe(true)
    expect(deepEqual(null, {})).toBe(false)
  })
})

describe('diffState', () => {
  it('marks every field changed, with before null, when there is no previous state', () => {
    const state = agentState({ route: 'call_tool' })
    const changes = diffState(null, state)

    expect(changes.every((c) => c.kind === 'changed' && c.before === null)).toBe(true)
    const routeChange = changes.find((c) => c.field === 'route')
    expect(routeChange?.after).toBe('call_tool')
  })

  it('marks identical states unchanged, field for field', () => {
    const state = agentState({ route: 'answer', response: 'done', terminal: true })
    const changes = diffState(state, state)

    expect(changes.every((c) => c.kind === 'unchanged')).toBe(true)
  })

  it('marks a changed scalar field with its before/after pair', () => {
    const before = agentState({ route: null })
    const after = agentState({ route: 'call_tool' })
    const changes = diffState(before, after)

    const routeChange = changes.find((c) => c.field === 'route')
    expect(routeChange).toEqual({ field: 'route', kind: 'changed', before: null, after: 'call_tool', appended: null })
  })

  it('marks an array field that only grew as appended, with just the new tail', () => {
    const observation = {
      tool_name: 'get_project_status',
      arguments_summary: 'milestone_id=M2',
      status: 'ok',
      summary: 'M2 is on track',
      source_ids: ['m2'],
      error: null,
      attempts: 1,
      retry_after_seconds: null,
    }
    const before = agentState({ observations: [] })
    const after = agentState({ observations: [observation] })
    const changes = diffState(before, after)

    const change = changes.find((c) => c.field === 'observations')
    expect(change?.kind).toBe('appended')
    expect(change?.appended).toEqual([observation])
  })

  it('marks an array field that shrank or was reordered as changed, not appended', () => {
    const o1 = { tool_name: 'a', arguments_summary: '', status: 'ok', summary: '', source_ids: [], error: null, attempts: 1, retry_after_seconds: null }
    const o2 = { ...o1, tool_name: 'b' }

    const grew = diffState(
      agentState({ observations: [o1, o2] }),
      agentState({ observations: [o1] }),
    )
    expect(grew.find((c) => c.field === 'observations')?.kind).toBe('changed')
  })

  it('compares scopes and redirected_needs as sets, ignoring wire order', () => {
    const before = agentState({ scopes: ['a', 'b'] })
    const after = agentState({ scopes: ['b', 'a'] })
    const changes = diffState(before, after)

    expect(changes.find((c) => c.field === 'scopes')?.kind).toBe('unchanged')
  })

  it('compares tool_arguments structurally, ignoring key order', () => {
    const before = agentState({ tool_name: 't', tool_arguments: { a: 1, b: 2 } })
    const after = agentState({ tool_name: 't', tool_arguments: { b: 2, a: 1 } })
    const changes = diffState(before, after)

    expect(changes.find((c) => c.field === 'tool_arguments')?.kind).toBe('unchanged')
  })

  it('produces one entry per field, in STATE_FIELD_GROUPS order', () => {
    const changes = diffState(null, agentState())
    const fields = changes.map((c) => c.field)
    const expected: StateField[] = STATE_FIELD_GROUPS.flatMap((g) => g.fields)
    expect(fields).toEqual(expected)
  })
})

describe('changedFields', () => {
  it('drops unchanged entries and keeps changed/appended ones', () => {
    const state: AgentStateSnapshot = agentState({ route: 'answer' })
    const changes = diffState(state, agentState({ route: 'answer', terminal: true }))
    const kept = changedFields(changes)

    expect(kept.every((c) => c.kind !== 'unchanged')).toBe(true)
    expect(kept.some((c) => c.field === 'terminal')).toBe(true)
    expect(kept.some((c) => c.field === 'route')).toBe(false)
  })
})
