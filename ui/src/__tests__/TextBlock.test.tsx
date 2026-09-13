import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { TextBlock } from '../components/shared/TextBlock'

describe('TextBlock', () => {
  it('renders short text with no collapse control', () => {
    render(<TextBlock text="a short passage" />)
    expect(screen.getByText('a short passage')).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('collapses long text behind a "show all" toggle', async () => {
    const user = userEvent.setup()
    const text = Array.from({ length: 20 }, (_, i) => `line ${i}`).join('\n')
    render(<TextBlock text={text} />)

    expect(screen.getByText(/show all 20 lines/)).toBeInTheDocument()
    // The text is still in the DOM (clipped by CSS, not removed) -- a
    // developer can still select/copy it before expanding.
    expect(screen.getByText(/line 0/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /show all/ }))
    expect(screen.getByText(/show less/)).toBeInTheDocument()
  })
})
