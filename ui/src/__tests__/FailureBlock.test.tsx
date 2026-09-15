import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { FailureBlock } from '../components/FailureBlock'

describe('FailureBlock', () => {
  it('renders nothing for a turn that did not fail', () => {
    const { container } = render(<FailureBlock failure="none" errorDetail={null} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders a real failure as the red block with its mode and detail', () => {
    render(<FailureBlock failure="provider_failure" errorDetail="ValidationError: ..." />)
    const alert = screen.getByRole('alert')
    expect(alert.className).toContain('destructive')
    expect(screen.getByText('provider_failure')).toBeInTheDocument()
    expect(screen.getByText('ValidationError: ...')).toBeInTheDocument()
  })

  it('renders incomplete_reply as a warning caveat, not a failure (ADR 0021)', () => {
    render(
      <FailureBlock
        failure="incomplete_reply"
        errorDetail="contract needs a document passage; the redirected search found 4 passage(s) that did not ground a reply"
      />,
    )
    const alert = screen.getByRole('alert')
    expect(alert.className).toContain('warning')
    expect(alert.className).not.toContain('destructive')
    expect(screen.getByText(/not confirmed by project documents/)).toBeInTheDocument()
    expect(screen.getByText('(incomplete_reply)')).toBeInTheDocument()
    expect(screen.getByText(/did not ground a reply/)).toBeInTheDocument()
  })
})
