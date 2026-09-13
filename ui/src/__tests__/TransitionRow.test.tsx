import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { TransitionRow } from '../components/trace/TransitionRow'
import type { ExecutionEntry } from '../lib/executionTree'
import { row, step } from './fixtures'

describe('TransitionRow', () => {
  it('renders the route and the best-matching reason', () => {
    const entry: ExecutionEntry = {
      kind: 'node',
      ordinal: 1,
      nodeName: 'start',
      rows: [row({ detail: 'call_tool: called get_project_status' })],
      step: step({ route: 'call_tool' }),
    }
    render(<TransitionRow entry={entry} />)
    expect(screen.getByText('call_tool')).toBeInTheDocument()
    expect(screen.getByText('call_tool: called get_project_status')).toBeInTheDocument()
  })

  it('renders the route badge alone when no row explains the reason', () => {
    const entry: ExecutionEntry = { kind: 'engine', ordinal: 1, nodeName: null, rows: [], step: step({ route: 'refuse' }) }
    render(<TransitionRow entry={entry} />)
    expect(screen.getByText('refuse')).toBeInTheDocument()
  })

  it('renders nothing for a still-streaming entry with no step yet', () => {
    const entry: ExecutionEntry = { kind: 'node', ordinal: 1, nodeName: 'call_tool', rows: [], step: null }
    const { container } = render(<TransitionRow entry={entry} />)
    expect(container).toBeEmptyDOMElement()
  })
})
