import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { EvidenceBlock } from '../components/trace/EvidenceBlock'
import type { EvidenceOut } from '../protocol'

describe('EvidenceBlock', () => {
  it('renders "no passages" for an empty list', () => {
    render(<EvidenceBlock evidence={[]} />)
    expect(screen.getByText('no passages retrieved')).toBeInTheDocument()
  })

  it('renders each passage collapsed behind its citation tag', async () => {
    const user = userEvent.setup()
    const snippet: EvidenceOut = {
      source_id: 'm2-status.md',
      locator: 'p.2',
      tag: '[m2-status.md#p.2]',
      text: { text: 'M2 slipped two weeks after a vendor delay.', truncated: false, chars: 43 },
    }
    render(<EvidenceBlock evidence={[snippet]} />)

    expect(screen.getByText('[m2-status.md#p.2]')).toBeInTheDocument()
    expect(screen.queryByText('M2 slipped two weeks after a vendor delay.')).not.toBeInTheDocument()

    await user.click(screen.getByText('[m2-status.md#p.2]'))
    expect(screen.getByText('M2 slipped two weeks after a vendor delay.')).toBeInTheDocument()
  })

  it('shows a truncation note when the server clipped the text', async () => {
    const user = userEvent.setup()
    const snippet: EvidenceOut = {
      source_id: 'doc-1',
      locator: 'p.1',
      tag: '[doc-1#p.1]',
      text: { text: 'xxxx', truncated: true, chars: 5000 },
    }
    render(<EvidenceBlock evidence={[snippet]} />)
    await user.click(screen.getByText('[doc-1#p.1]'))

    expect(screen.getByText(/showing 4 of 5,000 characters/)).toBeInTheDocument()
  })
})
