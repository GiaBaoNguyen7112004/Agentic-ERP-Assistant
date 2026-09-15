import { describe, expect, it } from 'vitest'
import { buildExecutionTree } from '../lib/executionTree'
import { hydrateTurn } from '../hydrate'
import type { RunReport, ToolOutcomeOut } from '../protocol'
import type { AgentStateSnapshot } from '../protocol'

const STAMP = '2026-01-01T00:00:00Z'

function aState(overrides: Partial<AgentStateSnapshot> = {}): AgentStateSnapshot {
  return {
    request: 'Why is milestone M2 late?',
    actor: 'priya',
    project_code: 'atlas',
    scopes: ['project.status.read'],
    trace_id: 'run-1',
    session_id: 'sess-1',
    route: 'answer',
    evidence: [],
    memories: [],
    history: [],
    contract: null,
    redirected_needs: [],
    draft: null,
    observations: [],
    tool_name: null,
    tool_arguments: null,
    tool_mutating: false,
    approval: 'not_required',
    response: 'M2 is two days late.',
    failure: 'none',
    error_detail: null,
    terminal: true,
    step_count: 3,
    retry_count: 0,
    events: [],
    state_version: 2,
    ...overrides,
  }
}

function aReport(overrides: {
  state?: AgentStateSnapshot
  events?: RunReport['events']
  observations?: ToolOutcomeOut[]
  outcome?: 'terminal' | 'paused'
} = {}): RunReport {
  const state = overrides.state ?? aState()
  return {
    run: {
      trace_id: 'run-1',
      actor: 'priya',
      project_code: 'atlas',
      outcome: overrides.outcome ?? 'terminal',
      started_at: STAMP,
      finished_at: STAMP,
      state,
    },
    events: overrides.events ?? [],
    audit_rows: [],
    model_calls: {
      count: 2,
      input_tokens: 100,
      output_tokens: 50,
      cost_usd: 0.01,
      unpriced: 0,
      records: [],
    },
    memory_audit: [],
    answer: { text: state.response ?? '', citations: [] },
  }
}

// The event log of a full R1-style run: prelude rows, planner, retrieval,
// planner, terminal -- exactly the shape the engine files (state.events has
// no seq; the query orders by the table's seq column).
function aToolTurnEvents(): RunReport['events'] {
  return [
    { node: 'history', kind: 'history_recalled', detail: '2 turn(s)' },
    { node: 'memory', kind: 'memory_recalled', detail: '1 record(s)' },
    { node: 'contract', kind: 'contract_declared', detail: 'erp_field' },
    { node: 'start', kind: 'node_entered', detail: '' },
    { node: 'think', kind: 'route_selected', detail: 'retrieve_project_documents: the reply needs a document passage' },
    { node: 'start', kind: 'node_exited', detail: '' },
    { node: 'retrieve_project_documents', kind: 'node_entered', detail: '' },
    { node: 'retrieve', kind: 'evidence_retrieved', detail: "2 passage(s) for 'M2 delay'" },
    { node: 'retrieve_project_documents', kind: 'node_exited', detail: '' },
    { node: 'think', kind: 'node_entered', detail: '' },
    { node: 'think', kind: 'route_selected', detail: 'answer: answered without calling a tool' },
    { node: 'think', kind: 'node_exited', detail: '' },
    { node: 'memory', kind: 'memory_written', detail: 'write (task in flight)' },
  ]
}

describe('hydrateTurn', () => {
  it('rebuilds a TurnView a live turn would have produced: tree, context, summary', () => {
    const state = aState({
      evidence: [
        { source_id: 'sprint-12-report.md', locator: '3.2', text: 'M2 slipped two days.' },
        { source_id: 'sprint-12-report.md', locator: '4.1', text: 'M2 recovery plan.' },
      ],
    })
    const { turn, unattributed } = hydrateTurn(aReport({ state, events: aToolTurnEvents() }))

    expect(unattributed).toEqual([])
    expect(turn.traceId).toBe('run-1')
    expect(turn.status).toBe('done')
    expect(turn.answer?.text).toBe('M2 is two days late.')
    expect(turn.summary?.outcome).toBe('terminal')
    expect(turn.summary?.started_at).toBe(STAMP)
    expect(turn.context?.type).toBe('context')
    expect(turn.context?.request).toBe('Why is milestone M2 late?')

    const tree = buildExecutionTree(turn)
    expect(tree.prelude.map((row) => row.kind)).toEqual([
      'history_recalled',
      'memory_recalled',
      'contract_declared',
    ])
    expect(tree.entries.map((entry) => entry.nodeName)).toEqual([
      'start',
      'retrieve_project_documents',
      'think',
    ])
    expect(tree.entries.map((entry) => entry.ordinal)).toEqual([1, 2, 3])
    // The filed reply closes the last span, terminal.
    expect(tree.entries[2].step?.terminal).toBe(true)
    expect(tree.entries[2].step?.response).toBe('M2 is two days late.')
    // Consolidation stays stepless, the way a live view files it.
    expect(tree.consolidation.map((row) => row.kind)).toEqual(['memory_written'])
  })

  it('places evidence on the retrieve span and nowhere else', () => {
    const state = aState({
      evidence: [{ source_id: 'sprint-12-report.md', locator: '3.2', text: 'M2 slipped.' }],
    })
    const { turn } = hydrateTurn(aReport({ state, events: aToolTurnEvents() }))
    const tree = buildExecutionTree(turn)

    const retrieve = tree.entries.find((entry) => entry.nodeName === 'retrieve_project_documents')
    expect(retrieve?.step?.evidence).toHaveLength(1)
    expect(retrieve?.step?.evidence?.[0].tag).toBe('[sprint-12-report.md#3.2]')
    expect(tree.entries[0].step?.evidence).toBeNull()
    expect(tree.entries[2].step?.evidence).toBeNull()
  })

  it('gives the planner spans their routes, read off the route_selected rows', () => {
    const { turn } = hydrateTurn(aReport({ events: aToolTurnEvents() }))
    const tree = buildExecutionTree(turn)

    expect(tree.entries[0].step?.route).toBe('retrieve_project_documents')
    expect(tree.entries[1].step?.route).toBeNull()
    expect(tree.entries[2].step?.route).toBe('answer')
  })

  it('attributes one observation per call_tool span, then think spans holding an approval_requested row', () => {
    const state = aState({
      observations: [
        {
          tool_name: 'get_project_status',
          arguments_summary: 'milestone_id=M2',
          status: 'ok',
          summary: 'M2 is at risk.',
          source_ids: ['milestone-m2'],
          error: null,
          attempts: 1,
          retry_after_seconds: null,
        },
        {
          tool_name: 'get_budget_summary',
          arguments_summary: 'project=atlas',
          status: 'ok',
          summary: 'Budget 80% used.',
          source_ids: [],
          error: null,
          attempts: 1,
          retry_after_seconds: null,
        },
      ],
      tool_name: 'get_budget_summary',
      tool_arguments: { project: 'atlas' },
    })
    const events: RunReport['events'] = [
      { node: 'start', kind: 'node_entered', detail: '' },
      { node: 'think', kind: 'route_selected', detail: 'call_tool: called get_project_status' },
      { node: 'start', kind: 'node_exited', detail: '' },
      { node: 'call_tool', kind: 'node_entered', detail: '' },
      { node: 'execute_tool', kind: 'tool_called', detail: 'get_project_status -> ok' },
      { node: 'call_tool', kind: 'node_exited', detail: '' },
      { node: 'think', kind: 'node_entered', detail: '' },
      { node: 'think', kind: 'route_selected', detail: 'call_tool: called get_budget_summary' },
      { node: 'think', kind: 'node_exited', detail: '' },
      { node: 'call_tool', kind: 'node_entered', detail: '' },
      { node: 'execute_tool', kind: 'tool_called', detail: 'get_budget_summary -> ok' },
      { node: 'call_tool', kind: 'node_exited', detail: '' },
      { node: 'think', kind: 'node_entered', detail: '' },
      { node: 'think', kind: 'route_selected', detail: 'answer: done' },
      { node: 'think', kind: 'node_exited', detail: '' },
    ]
    const { turn, unattributed } = hydrateTurn(aReport({ state, events }))
    const tree = buildExecutionTree(turn)

    expect(unattributed).toEqual([])
    const callSpans = tree.entries.filter((entry) => entry.nodeName === 'call_tool')
    expect(callSpans).toHaveLength(2)
    expect(callSpans[0].step?.observations[0].tool_name).toBe('get_project_status')
    expect(callSpans[1].step?.observations[0].tool_name).toBe('get_budget_summary')
    // The LAST call_tool span alone carries the filed state's tool fields.
    expect(callSpans[0].step?.tool_name).toBeNull()
    expect(callSpans[1].step?.tool_name).toBe('get_budget_summary')
    expect(tree.entries[4].step?.observations).toEqual([])
  })

  it('takes a refusal observation on the think span that escalated, before calling anything unattributed', () => {
    const state = aState({
      observations: [
        {
          tool_name: 'create_risk',
          arguments_summary: 'title=…',
          status: 'denied',
          summary: 'The approver denied the write.',
          source_ids: [],
          error: null,
          attempts: 1,
          retry_after_seconds: null,
        },
      ],
    })
    const events: RunReport['events'] = [
      { node: 'start', kind: 'node_entered', detail: '' },
      { node: 'think', kind: 'approval_requested', detail: 'create_risk needs a human' },
      { node: 'think', kind: 'route_selected', detail: 'request_approval: mutating tool' },
      { node: 'start', kind: 'node_exited', detail: '' },
      { node: 'approval', kind: 'approval_recorded', detail: 'create_risk denied by priya' },
      { node: 'think', kind: 'node_entered', detail: '' },
      { node: 'think', kind: 'route_selected', detail: 'fail: the write was denied' },
      { node: 'think', kind: 'node_exited', detail: '' },
    ]
    const { turn, unattributed } = hydrateTurn(aReport({ state, events }))
    const tree = buildExecutionTree(turn)

    expect(unattributed).toEqual([])
    const thinkWithEscalation = tree.entries[0]
    expect(thinkWithEscalation.step?.observations).toHaveLength(1)
    expect(thinkWithEscalation.step?.observations[0].status).toBe('denied')
    // The approval decision itself is an engine card, as a live step with
    // `node: null` becomes one.
    expect(tree.entries[1].kind).toBe('engine')
    expect(tree.entries[1].nodeName).toBe('approval')
  })

  it('leaves observations the rule cannot place in the unattributed bucket, never on a wrong span', () => {
    const state = aState({
      observations: [
        {
          tool_name: 'ghost_call',
          arguments_summary: '',
          status: 'ok',
          summary: 'no span claims this',
          source_ids: [],
          error: null,
          attempts: 1,
          retry_after_seconds: null,
        },
      ],
    })
    const events: RunReport['events'] = [
      { node: 'start', kind: 'node_entered', detail: '' },
      { node: 'think', kind: 'route_selected', detail: 'answer: done' },
      { node: 'start', kind: 'node_exited', detail: '' },
    ]
    const { turn, unattributed } = hydrateTurn(aReport({ state, events }))
    const tree = buildExecutionTree(turn)

    expect(unattributed).toHaveLength(1)
    expect(unattributed[0].tool_name).toBe('ghost_call')
    for (const entry of tree.entries) {
      expect(entry.step?.observations).toEqual([])
    }
  })

  it('keeps model calls at run level: no span claims one, the summary carries them', () => {
    const { turn } = hydrateTurn(aReport({ events: aToolTurnEvents() }))

    expect(turn.steps.every((step) => step.model_calls.length === 0)).toBe(true)
    expect(turn.summary?.model_calls.count).toBe(2)
    expect(turn.context?.model_calls).toEqual([])
  })

  it('hydrates a paused run as a paused view', () => {
    const state = aState({
      route: 'request_approval',
      tool_name: 'create_risk',
      tool_arguments: { title: 'x' },
      tool_mutating: true,
      approval: 'pending',
      response: null,
      terminal: false,
    })
    const { turn } = hydrateTurn(aReport({ state, outcome: 'paused' }))

    expect(turn.status).toBe('paused')
    expect(turn.summary?.outcome).toBe('paused')
  })

  it('survives runs with no events at all and with no contract', () => {
    const { turn } = hydrateTurn(aReport({ state: aState({ contract: null }) }))
    expect(turn.context?.contract).toBeNull()
    expect(turn.steps).toEqual([])
    expect(buildExecutionTree(turn).entries).toEqual([])
  })

  it('marks a contract the run actually declared', () => {
    const state = aState({
      contract: { needs: ['erp_field', 'document_passage'], document_query: 'M2 delay' },
    })
    const { turn } = hydrateTurn(aReport({ state }))
    expect(turn.context?.contract).toEqual({
      needs: ['erp_field', 'document_passage'],
      document_query: 'M2 delay',
    })
  })

  it('carries the filed state only on the last step; context and every earlier step get null', () => {
    const state = aState({ response: 'M2 is two days late.' })
    const { turn } = hydrateTurn(aReport({ state, events: aToolTurnEvents() }))

    expect(turn.context?.state).toBeNull()
    expect(turn.steps.length).toBeGreaterThan(1)
    expect(turn.steps.slice(0, -1).every((s) => s.state === null)).toBe(true)
    expect(turn.steps.at(-1)?.state).toEqual(state)
  })
})