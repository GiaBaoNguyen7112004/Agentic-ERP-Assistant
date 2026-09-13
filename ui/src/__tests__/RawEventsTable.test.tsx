import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { RawEventsTable } from '../components/trace/RawEventsTable'
import type { TraceRow } from '../protocol'

function row(overrides: Partial<TraceRow>): TraceRow {
  return { type: 'trace', seq: 0, node: 'think', kind: 'route_selected', detail: '', source: 'engine', ...overrides }
}

describe('RawEventsTable', () => {
  it('renders nothing for an empty event list', () => {
    const { container } = render(<RawEventsTable events={[]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('is collapsed by default, with the row count visible', () => {
    render(<RawEventsTable events={[row({ detail: 'first' }), row({ seq: 1, detail: 'second' })]} />)
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.queryByText('first')).not.toBeInTheDocument()
  })

  it('shows every row once expanded', async () => {
    const user = userEvent.setup()
    render(<RawEventsTable events={[row({ detail: 'first' }), row({ seq: 1, detail: 'second' })]} />)
    await user.click(screen.getByRole('button', { name: /Raw events/ }))
    expect(screen.getByText('first')).toBeInTheDocument()
    expect(screen.getByText('second')).toBeInTheDocument()
  })
})
