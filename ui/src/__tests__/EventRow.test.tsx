import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { EventRow } from '../components/EventRow'

describe('EventRow', () => {
  it('renders an engine row with its seq', () => {
    const { container } = render(
      <EventRow
        event={{ type: 'trace', seq: 3, node: 'think', kind: 'route_selected', detail: 'answer', source: 'engine' }}
      />,
    )
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(container.querySelector('.trace-row')).not.toHaveClass('gateway')
  })

  it('renders a gateway row in italics with no seq', () => {
    const { container } = render(
      <EventRow
        event={{
          type: 'trace',
          seq: null,
          node: 'tool_gateway',
          kind: 'retry_scheduled',
          detail: 'attempt 2',
          source: 'tool_gateway',
        }}
      />,
    )
    expect(container.querySelector('.trace-row')).toHaveClass('gateway')
    expect(screen.queryByText('3')).not.toBeInTheDocument()
    // seq renders as the placeholder, not a number
    expect(screen.getByText('·')).toBeInTheDocument()
  })
})
