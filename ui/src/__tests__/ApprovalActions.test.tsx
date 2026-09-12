import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ApprovalActions } from '../components/shared/ApprovalActions'

describe('ApprovalActions', () => {
  it('both buttons are named exactly Approve and Deny', () => {
    render(
      <ApprovalActions canApprove={true} deciding={false} onDecide={() => {}} />,
    )
    expect(screen.getByRole('button', { name: 'Approve' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Deny' })).toBeInTheDocument()
  })

  it('both buttons are disabled while deciding', () => {
    render(<ApprovalActions canApprove={true} deciding={true} onDecide={() => {}} />)
    expect(screen.getByRole('button', { name: 'Approve' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Deny' })).toBeDisabled()
  })

  it('both buttons are disabled for a non-approver', () => {
    render(<ApprovalActions canApprove={false} deciding={false} onDecide={() => {}} />)
    expect(screen.getByRole('button', { name: 'Approve' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Deny' })).toBeDisabled()
  })

  it('clicking Approve reports approved=true; Deny, false', async () => {
    const onDecide = vi.fn()
    const user = userEvent.setup()
    render(<ApprovalActions canApprove={true} deciding={false} onDecide={onDecide} />)
    await user.click(screen.getByRole('button', { name: 'Approve' }))
    await user.click(screen.getByRole('button', { name: 'Deny' }))
    expect(onDecide).toHaveBeenNthCalledWith(1, true)
    expect(onDecide).toHaveBeenNthCalledWith(2, false)
  })
})