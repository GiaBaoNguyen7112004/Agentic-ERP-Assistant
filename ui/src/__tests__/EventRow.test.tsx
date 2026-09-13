import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { EventRow } from '../components/EventRow'

describe('EventRow', () => {
  it('renders an engine row with its seq', () => {
    render(
      <EventRow
        event={{
          type: 'trace',
          seq: 3,
          node: 'think',
          kind: 'route_selected',
          detail: 'answer',
          source: 'engine',
          step: null,
        }}
      />,
    )
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(
      document.querySelector('[data-source="tool_gateway"]'),
    ).not.toBeInTheDocument()
    expect(document.querySelector('[data-source="engine"]')).toBeInTheDocument()
  })

  it('renders a gateway row marked live-only with no seq', () => {
    render(
      <EventRow
        event={{
          type: 'trace',
          seq: null,
          node: 'tool_gateway',
          kind: 'retry_scheduled',
          detail: 'attempt 2',
          source: 'tool_gateway',
          step: 1,
        }}
      />,
    )
    // The gateway marker moved from a CSS class to data-source; the
    // behaviour asserted (a distinguishable live-only row) is the same.
    const row = document.querySelector('[data-source="tool_gateway"]')
    expect(row).not.toBeNull()
    expect(row).toHaveAttribute('title', 'live only — tool gateway hook, not persisted')
    expect(screen.queryByText('3')).not.toBeInTheDocument()
    // seq renders as the placeholder, not a number
    expect(screen.getByText('·')).toBeInTheDocument()
  })
})