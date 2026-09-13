import { describe, expect, it } from 'vitest'
import { nodeLabel } from '../lib/nodeLabel'

describe('nodeLabel', () => {
  it('maps both planner-route span names to Planner', () => {
    expect(nodeLabel('start')).toBe('Planner')
    expect(nodeLabel('think')).toBe('Planner')
  })

  it('maps the retrieval and tool routes', () => {
    expect(nodeLabel('retrieve_project_documents')).toBe('Retrieve & compose')
    expect(nodeLabel('call_tool')).toBe('Execute tool')
  })

  it('maps the engine-level names', () => {
    expect(nodeLabel('approval')).toBe('Approval')
    expect(nodeLabel('engine')).toBe('Loop guard')
  })

  it('maps the three context-phase names to one label', () => {
    expect(nodeLabel('history')).toBe('Context')
    expect(nodeLabel('memory')).toBe('Context')
    expect(nodeLabel('contract')).toBe('Context')
  })

  it('falls back to the raw name for anything unmapped, and to "Turn ended" for null', () => {
    expect(nodeLabel('retrieve')).toBe('retrieve')
    expect(nodeLabel(null)).toBe('Turn ended')
  })
})
