import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ToolOutcomeBlock } from '../components/trace/ToolOutcomeBlock'
import type { ToolOutcomeOut } from '../protocol'

function outcome(overrides: Partial<ToolOutcomeOut> = {}): ToolOutcomeOut {
  return {
    tool_name: 'get_project_status',
    arguments_summary: 'get_project_status(milestone_id=M2)',
    status: 'ok',
    summary: { text: 'M2 is at risk and 2 days late.', truncated: false, chars: 30 },
    source_ids: ['milestone-m2'],
    error: null,
    attempts: 1,
    retry_after_seconds: null,
    ...overrides,
  }
}

describe('ToolOutcomeBlock', () => {
  it('renders nothing for an empty list', () => {
    const { container } = render(<ToolOutcomeBlock observations={[]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders a successful outcome with its summary and source_ids', () => {
    render(<ToolOutcomeBlock observations={[outcome()]} />)
    expect(screen.getByText('get_project_status')).toBeInTheDocument()
    expect(screen.getByText('ok')).toBeInTheDocument()
    expect(screen.getByText('M2 is at risk and 2 days late.')).toBeInTheDocument()
    expect(screen.getByText(/milestone-m2/)).toBeInTheDocument()
  })

  it('renders a failed outcome with its error, not a summary', () => {
    render(
      <ToolOutcomeBlock
        observations={[
          outcome({
            status: 'denied',
            summary: { text: '', truncated: false, chars: 0 },
            source_ids: [],
            error: "actor 'tomas' does not hold 'project.budget.read'",
          }),
        ]}
      />,
    )
    expect(screen.getByText('denied')).toBeInTheDocument()
    expect(screen.getByText(/does not hold/)).toBeInTheDocument()
  })

  it('renders a rate-limited outcome with how long to wait', () => {
    render(
      <ToolOutcomeBlock
        observations={[
          outcome({
            status: 'rate_limited',
            summary: { text: '', truncated: false, chars: 0 },
            source_ids: [],
            error: 'actor has spent its budget',
            retry_after_seconds: 12.5,
          }),
        ]}
      />,
    )
    expect(screen.getByText('rate_limited')).toBeInTheDocument()
    expect(screen.getByText('retry in 12.5s')).toBeInTheDocument()
  })

  it('renders the attempt count only when more than one attempt was made', () => {
    const { rerender } = render(<ToolOutcomeBlock observations={[outcome({ attempts: 1 })]} />)
    expect(screen.queryByText(/attempts/)).not.toBeInTheDocument()

    rerender(<ToolOutcomeBlock observations={[outcome({ attempts: 3 })]} />)
    expect(screen.getByText('3 attempts')).toBeInTheDocument()
  })
})
