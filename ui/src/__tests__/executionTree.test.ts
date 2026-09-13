import { describe, expect, it } from 'vitest'
import { buildExecutionTree, entryToolCall, type ExecutionEntry } from '../lib/executionTree'
import { initialTurn, type TimelineEntry, type TurnView } from '../turnReducer'
import type { StepEvent, TraceRow } from '../protocol'

function row(overrides: Partial<TraceRow>): TraceRow {
  return { type: 'trace', seq: 0, node: 'think', kind: 'route_selected', detail: '', source: 'engine', ...overrides }
}

function step(overrides: Partial<StepEvent>): StepEvent {
  return {
    type: 'step',
    route: null,
    tool_name: null,
    tool_arguments: null,
    tool_mutating: null,
    approval: 'not_required',
    step_count: 1,
    terminal: false,
    ...overrides,
  }
}

const r = (partial: Partial<TraceRow>): TimelineEntry => ({ kind: 'row', row: row(partial) })
const s = (partial: Partial<StepEvent>): TimelineEntry => ({ kind: 'step', step: step(partial) })

function viewOf(timeline: TimelineEntry[]): TurnView {
  return { ...initialTurn, timeline }
}

describe('buildExecutionTree', () => {
  it('groups a three-node tool turn into three node entries, in order', () => {
    // node_entered start / route_selected / node_exited start -> step(call_tool)
    // node_entered call_tool / tool_called / node_exited call_tool -> step(think)
    // node_entered think / route_selected / node_exited think -> step(answer, terminal)
    const timeline: TimelineEntry[] = [
      r({ seq: 0, node: 'start', kind: 'node_entered', detail: '' }),
      r({ seq: 1, node: 'think', kind: 'route_selected', detail: 'call_tool: called get_project_status' }),
      r({ seq: 2, node: 'start', kind: 'node_exited', detail: '' }),
      s({ route: 'call_tool', tool_name: 'get_project_status', step_count: 1 }),

      r({ seq: 3, node: 'call_tool', kind: 'node_entered', detail: '' }),
      r({ seq: 4, node: 'execute_tool', kind: 'tool_called', detail: 'get_project_status -> ok' }),
      r({ seq: 5, node: 'call_tool', kind: 'node_exited', detail: '' }),
      s({ route: 'think', step_count: 2 }),

      r({ seq: 6, node: 'think', kind: 'node_entered', detail: '' }),
      r({ seq: 7, node: 'think', kind: 'route_selected', detail: 'answer: answered without calling a tool' }),
      r({ seq: 8, node: 'think', kind: 'node_exited', detail: '' }),
      s({ route: 'answer', step_count: 3, terminal: true }),
    ]

    const tree = buildExecutionTree(viewOf(timeline))

    expect(tree.prelude).toEqual([])
    expect(tree.consolidation).toEqual([])
    expect(tree.entries).toHaveLength(3)
    expect(tree.entries.map((e) => e.kind)).toEqual(['node', 'node', 'node'])
    expect(tree.entries.map((e) => e.nodeName)).toEqual(['start', 'call_tool', 'think'])
    expect(tree.entries.map((e) => e.ordinal)).toEqual([1, 2, 3])
    expect(tree.entries[0].rows).toHaveLength(3)
    expect(tree.entries[1].rows).toHaveLength(3)
    expect(tree.entries[2].step?.terminal).toBe(true)
  })

  it('attaches a live gateway row to the node span it precedes, not the one before it', () => {
    // The gateway's own trace_event fires *during* call_tool's execution,
    // before that node's own rows (node_entered included) have flushed --
    // so it arrives on the wire ahead of call_tool's batch, and must not be
    // read as trailing the previous (start) span.
    const timeline: TimelineEntry[] = [
      r({ seq: 0, node: 'start', kind: 'node_entered' }),
      r({ seq: 1, node: 'think', kind: 'route_selected', detail: 'call_tool: called get_project_status' }),
      r({ seq: 2, node: 'start', kind: 'node_exited' }),
      s({ route: 'call_tool', step_count: 1 }),

      r({ seq: null, node: 'tool_gateway', kind: 'tool_called', detail: 'get_project_status -> ok', source: 'tool_gateway' }),
      r({ seq: 3, node: 'call_tool', kind: 'node_entered' }),
      r({ seq: 4, node: 'execute_tool', kind: 'tool_called', detail: 'get_project_status -> ok' }),
      r({ seq: 5, node: 'call_tool', kind: 'node_exited' }),
      s({ route: 'think', step_count: 2 }),
    ]

    const tree = buildExecutionTree(viewOf(timeline))

    expect(tree.entries).toHaveLength(2)
    expect(tree.entries[0].rows).toHaveLength(3) // start's own three rows only
    expect(tree.entries[1].rows).toHaveLength(4) // the gateway row + call_tool's three
    expect(tree.entries[1].rows[0].source).toBe('tool_gateway')
    // The span is named after its own node_entered row, never the gateway
    // row that happens to sit first -- a leading gateway row must not
    // misname the whole span after itself.
    expect(tree.entries[1].nodeName).toBe('call_tool')
  })

  it('splits prelude rows folded into the very first batch away from node 1', () => {
    const timeline: TimelineEntry[] = [
      r({ seq: 0, node: 'history', kind: 'history_recalled', detail: '2 prior turn(s)' }),
      r({ seq: 1, node: 'memory', kind: 'memory_recalled', detail: '1 memor(y|ies) recalled' }),
      r({ seq: 2, node: 'contract', kind: 'contract_declared', detail: 'needs=(none)' }),
      r({ seq: 3, node: 'start', kind: 'node_entered' }),
      r({ seq: 4, node: 'think', kind: 'route_selected', detail: 'answer: answered without calling a tool' }),
      r({ seq: 5, node: 'start', kind: 'node_exited' }),
      s({ route: 'answer', step_count: 1, terminal: true }),
    ]

    const tree = buildExecutionTree(viewOf(timeline))

    expect(tree.prelude.map((row) => row.kind)).toEqual([
      'history_recalled',
      'memory_recalled',
      'contract_declared',
    ])
    expect(tree.entries).toHaveLength(1)
    expect(tree.entries[0].rows).toHaveLength(3)
  })

  it('collects trailing rows filed after the last step as consolidation', () => {
    const timeline: TimelineEntry[] = [
      r({ seq: 0, node: 'start', kind: 'node_entered' }),
      r({ seq: 1, node: 'think', kind: 'route_selected', detail: 'answer: answered without calling a tool' }),
      r({ seq: 2, node: 'start', kind: 'node_exited' }),
      s({ route: 'answer', step_count: 1, terminal: true }),
      // flush_events streams these with no step of its own.
      r({ seq: 3, node: 'memory', kind: 'memory_written', detail: 'write (…)' }),
      r({ seq: 4, node: 'history', kind: 'history_promoted', detail: '1 turn(s) folded into the session summary' }),
    ]

    const tree = buildExecutionTree(viewOf(timeline))

    expect(tree.entries).toHaveLength(1)
    expect(tree.consolidation.map((row) => row.kind)).toEqual(['memory_written', 'history_promoted'])
  })

  it('treats a bare approval_recorded as an engine entry, not a node', () => {
    const timeline: TimelineEntry[] = [
      r({ seq: 0, node: 'start', kind: 'node_entered' }),
      r({ seq: 1, node: 'think', kind: 'approval_requested', detail: 'create_risk needs a human' }),
      r({ seq: 2, node: 'start', kind: 'node_exited' }),
      s({ route: 'request_approval', approval: 'pending', step_count: 1 }),

      r({ seq: 3, node: 'approval', kind: 'approval_recorded', detail: 'create_risk approved by priya' }),
      s({ route: 'request_approval', approval: 'approved', step_count: 1 }),

      r({ seq: 4, node: 'call_tool', kind: 'node_entered' }),
      r({ seq: 5, node: 'execute_tool', kind: 'tool_called', detail: 'create_risk -> ok' }),
      r({ seq: 6, node: 'call_tool', kind: 'node_exited' }),
      s({ route: 'think', step_count: 2 }),
    ]

    const tree = buildExecutionTree(viewOf(timeline))

    expect(tree.entries.map((e) => e.kind)).toEqual(['node', 'engine', 'node'])
    expect(tree.entries[1].nodeName).toBe('approval')
    expect(tree.entries[1].rows).toHaveLength(1)
  })

  it('treats a denial\'s zero-row refusal step as its own bare engine entry', () => {
    // resume_approval's denial path observes twice: once for the recorded
    // approval_recorded row, and once more for the refusal -- which adds no
    // new trace row at all before its step fires.
    const timeline: TimelineEntry[] = [
      r({ seq: 0, node: 'approval', kind: 'approval_recorded', detail: 'create_risk denied by priya' }),
      s({ route: 'request_approval', approval: 'denied', step_count: 1 }),
      s({ route: 'refuse', step_count: 1, terminal: true }),
    ]

    const tree = buildExecutionTree(viewOf(timeline))

    expect(tree.entries).toHaveLength(2)
    expect(tree.entries[0].rows).toHaveLength(1)
    expect(tree.entries[1].kind).toBe('engine')
    expect(tree.entries[1].rows).toEqual([])
    expect(tree.entries[1].nodeName).toBeNull()
    expect(tree.entries[1].step?.route).toBe('refuse')
  })

  it('marks a still-streaming node span as open, with no step yet', () => {
    const timeline: TimelineEntry[] = [
      r({ seq: 0, node: 'start', kind: 'node_entered' }),
      r({ seq: 1, node: 'think', kind: 'route_selected', detail: 'call_tool: called get_project_status' }),
      r({ seq: 2, node: 'start', kind: 'node_exited' }),
      s({ route: 'call_tool', step_count: 1 }),

      r({ seq: 3, node: 'call_tool', kind: 'node_entered' }),
      // no node_exited/step yet -- the tool call is still running
    ]

    const tree = buildExecutionTree(viewOf(timeline))

    expect(tree.entries).toHaveLength(2)
    expect(tree.entries[1].step).toBeNull()
    expect(tree.entries[1].ordinal).toBe(2)
    expect(tree.entries[1].nodeName).toBe('call_tool')
  })

  it('an empty timeline has no prelude, entries, or consolidation', () => {
    const tree = buildExecutionTree(viewOf([]))
    expect(tree).toEqual({ prelude: [], entries: [], consolidation: [] })
  })
})

describe('entryToolCall', () => {
  function entry(overrides: Partial<ExecutionEntry>): ExecutionEntry {
    return { kind: 'node', ordinal: 1, nodeName: null, rows: [], step: null, ...overrides }
  }

  it('is relevant for the node that executed the call, whatever route it produced', () => {
    const e = entry({
      nodeName: 'call_tool',
      step: step({ route: 'think', tool_name: 'get_project_status', tool_arguments: { milestone_id: 'M2' } }),
    })
    expect(entryToolCall(e)).toEqual({ name: 'get_project_status', arguments: { milestone_id: 'M2' } })
  })

  it('is relevant for a planner decision that calls or escalates a tool', () => {
    const calling = entry({
      nodeName: 'think',
      step: step({ route: 'call_tool', tool_name: 'list_risks', tool_arguments: { project_id: 'atlas' } }),
    })
    const escalating = entry({
      nodeName: 'think',
      step: step({ route: 'request_approval', tool_name: 'create_risk', tool_arguments: {} }),
    })
    expect(entryToolCall(calling)?.name).toBe('list_risks')
    expect(entryToolCall(escalating)?.name).toBe('create_risk')
  })

  it('is relevant for an approval entry', () => {
    const e = entry({
      kind: 'engine',
      nodeName: 'approval',
      step: step({ route: 'request_approval', approval: 'approved', tool_name: 'create_risk', tool_arguments: {} }),
    })
    expect(entryToolCall(e)?.name).toBe('create_risk')
  })

  it('is null for a planner decision that answers, even though the state still carries a prior tool_name', () => {
    // AgentState.tool_name/tool_arguments are never cleared once set (see
    // engine/nodes.py::think's "answer" branch), so a later think node that
    // decided to answer still reports the *previous* call's tool on its
    // StepEvent -- this must not be read as "this node called a tool".
    const e = entry({
      nodeName: 'think',
      step: step({ route: 'answer', tool_name: 'get_project_status', tool_arguments: { milestone_id: 'M2' } }),
    })
    expect(entryToolCall(e)).toBeNull()
  })

  it('is null for a bare refusal entry carrying a stale tool_name', () => {
    const e = entry({ kind: 'engine', nodeName: null, step: step({ route: 'refuse', tool_name: 'create_risk' }) })
    expect(entryToolCall(e)).toBeNull()
  })

  it('is null when the step names no tool at all', () => {
    expect(entryToolCall(entry({ nodeName: 'call_tool', step: step({ route: 'think', tool_name: null }) }))).toBeNull()
  })

  it('is null while the entry is still streaming (no step yet)', () => {
    expect(entryToolCall(entry({ nodeName: 'call_tool', step: null }))).toBeNull()
  })
})
