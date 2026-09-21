import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { PhaseCard } from '../components/trace/PhaseCard'
import { row } from './fixtures'

describe('PhaseCard', () => {
  it('renders nothing when there are no rows and nothing extra to show', () => {
    const { container } = render(<PhaseCard title="Consolidation" rows={[]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('is collapsed by default and shows the row count', () => {
    render(<PhaseCard title="Consolidation" rows={[row({ detail: 'write (…)' })]} />)
    expect(screen.getByText('Consolidation')).toBeInTheDocument()
    expect(screen.getByText('1')).toBeInTheDocument()
    expect(screen.queryByText('write (…)')).not.toBeInTheDocument()
  })

  it('shows its rows once expanded', async () => {
    const user = userEvent.setup()
    render(<PhaseCard title="Context" rows={[row({ kind: 'history_recalled', detail: '2 prior turn(s)' })]} />)
    await user.click(screen.getByRole('button', { name: /Context/ }))
    expect(screen.getByText('2 prior turn(s)')).toBeInTheDocument()
  })

  it('renders when there are no rows but extra content was passed', () => {
    render(<PhaseCard title="Context" rows={[]} defaultOpen extra={<p>nothing recalled</p>} />)
    expect(screen.getByText('nothing recalled')).toBeInTheDocument()
  })
})
