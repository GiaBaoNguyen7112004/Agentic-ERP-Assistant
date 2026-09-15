import { describe, expect, it } from 'vitest'
import { traceTone } from '../lib/traceTone'

// Same substring rules as the old kindClass(); the tones are what changed.
describe('traceTone', () => {
  it('maps the kinds the handbook names', () => {
    expect(traceTone('route_selected')).toBe('info')
    expect(traceTone('tool_called')).toBe('success')
    expect(traceTone('approval_requested')).toBe('warning')
    expect(traceTone('memory_written')).toBe('memory')
    // 'tool_called' is checked before 'failed' -- the old kindClass() did the
    // same, so a failed tool call still reads success-toned 'tool_called'.
    expect(traceTone('tool_called_failed')).toBe('success')
    expect(traceTone('run_failed')).toBe('destructive')
  })

  it('keeps the substring behaviour for compound kinds', () => {
    expect(traceTone('tool_retry_failed')).toBe('destructive')
    expect(traceTone('memory_recalled')).toBe('memory')
    expect(traceTone('approval_granted')).toBe('warning')
  })

  it('everything else is neutral', () => {
    expect(traceTone('retrieval_done')).toBe('neutral')
    expect(traceTone('compose_answer')).toBe('neutral')
  })
})