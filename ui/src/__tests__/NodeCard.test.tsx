import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { NodeCard } from '../components/trace/NodeCard'
import type { ExecutionEntry } from '../lib/executionTree'
import { agentState, row, step } from './fixtures'

describe('NodeCard', () => {
  it('renders a numbered node entry with its label, raw name, and rows, open by default', () => {
    const entry: ExecutionEntry = {
      kind: 'node',
      ordinal: 2,
      nodeName: 'call_tool',
      rows: [row({ kind: 'tool_called', detail: 'get_project_status -> ok' })],
      step: step({ route: 'think', tool_name: 'get_project_status', step_count: 2 }),
      stateBefore: null,
    }
    render(<NodeCard entry={entry} />)

    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText('Execute tool')).toBeInTheDocument()
    expect(screen.getByText('(call_tool)')).toBeInTheDocument()
    expect(screen.getByText('→ get_project_status')).toBeInTheDocument()
    // Shows up twice: once as the row itself, once as the transition's own
    // reason (the same row is the highest-priority match for both).
    expect(screen.getAllByText('get_project_status -> ok')).toHaveLength(2)
    // Open by default -- the transition is visible without expanding anything.
    expect(screen.getByText('think')).toBeInTheDocument()
  })

  it('does not number a bare engine entry', () => {
    const entry: ExecutionEntry = {
      kind: 'engine',
      ordinal: 1,
      nodeName: 'approval',
      rows: [row({ node: 'approval', kind: 'approval_recorded', detail: 'create_risk approved by priya' })],
      step: step({ route: 'request_approval', approval: 'approved' }),
      stateBefore: null,
    }
    render(<NodeCard entry={entry} />)

    expect(screen.getByText('Approval')).toBeInTheDocument()
    expect(screen.queryByText('1')).not.toBeInTheDocument()
  })

  it('renders the tool call arguments as JSON when the step names a tool', () => {
    const entry: ExecutionEntry = {
      kind: 'node',
      ordinal: 1,
      nodeName: 'call_tool',
      rows: [],
      step: step({
        route: 'call_tool',
        tool_name: 'create_risk',
        tool_arguments: { project_id: 'atlas', severity: 'high' },
      }),
      stateBefore: null,
    }
    render(<NodeCard entry={entry} />)

    expect(screen.getByText(/"project_id": "atlas"/)).toBeInTheDocument()
  })

  it('shows a "no new trace rows" note and a spinner for a still-streaming node with no rows yet', () => {
    const entry: ExecutionEntry = { kind: 'node', ordinal: 3, nodeName: 'call_tool', rows: [], step: null, stateBefore: null }
    render(<NodeCard entry={entry} />)

    expect(screen.getByText('no new trace rows')).toBeInTheDocument()
  })

  it('shows a State trigger with the changed-field count when the step carries a state', () => {
    const previous = agentState({ route: null })
    const next = agentState({ route: 'call_tool', step_count: 1 })
    const entry: ExecutionEntry = {
      kind: 'node',
      ordinal: 1,
      nodeName: 'think',
      rows: [],
      step: step({ route: 'call_tool', step_count: 1, state: next }),
      stateBefore: previous,
    }
    render(<NodeCard entry={entry} />)

    expect(screen.getByText('2 changed')).toBeInTheDocument()
  })

  it('shows no State trigger for a still-streaming entry with no step yet', () => {
    const entry: ExecutionEntry = { kind: 'node', ordinal: 1, nodeName: 'think', rows: [], step: null, stateBefore: null }
    render(<NodeCard entry={entry} />)

    expect(screen.queryByText(/State not carried/)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^State/ })).not.toBeInTheDocument()
  })

  it('collapses on click, hiding its rows', async () => {
    const user = userEvent.setup()
    const entry: ExecutionEntry = {
      kind: 'node',
      ordinal: 1,
      nodeName: 'think',
      rows: [row({ detail: 'answer: answered without calling a tool' })],
      step: step({ route: 'answer', terminal: true }),
      stateBefore: null,
    }
    render(<NodeCard entry={entry} />)

    // 'route_selected' (the row's kind label) appears once, in the row
    // list -- the transition row below it shows the route badge and the
    // same detail text as its reason, so that text alone would match twice.
    expect(screen.getByText('route_selected')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { expanded: true }))
    expect(screen.queryByText('route_selected')).not.toBeInTheDocument()
  })
})
