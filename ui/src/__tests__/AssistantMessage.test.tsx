import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { AssistantMessage } from '../components/AssistantMessage'
import { initialTurn } from '../turnReducer'
import type { TurnView } from '../turnReducer'

function turn(overrides: Partial<TurnView>): TurnView {
  return { ...initialTurn, ...overrides }
}

describe('AssistantMessage', () => {
  it('shows the streamed preview while the turn is still running', () => {
    render(
      <AssistantMessage
        turn={turn({ status: 'running', streamed: 'Milestone M2 is' })}
        actor="priya"
        canApprove={false}
        deciding={false}
        onDecide={vi.fn()}
      />,
    )
    expect(screen.getByText('Milestone M2 is')).toBeInTheDocument()
  })

  it('shows the authoritative answer text once it arrives, not the preview', () => {
    render(
      <AssistantMessage
        turn={turn({
          status: 'done',
          streamed: 'a stale partial preview',
          answer: {
            type: 'answer',
            text: 'The final, correct answer.',
            route: 'answer',
            failure: 'none',
            error_detail: null,
            citations: [],
          },
        })}
        actor="priya"
        canApprove={false}
        deciding={false}
        onDecide={vi.fn()}
      />,
    )
    expect(screen.getByText('The final, correct answer.')).toBeInTheDocument()
    expect(screen.queryByText('a stale partial preview')).not.toBeInTheDocument()
  })

  it('renders citation chips from the answer', () => {
    render(
      <AssistantMessage
        turn={turn({
          status: 'done',
          answer: {
            type: 'answer',
            text: 'M2 slipped.',
            route: 'answer',
            failure: 'none',
            error_detail: null,
            citations: [{ source_id: 'm2-status.md', locator: 'p.2', tag: '[m2-status.md#p.2]', kind: 'document' }],
          },
        })}
        actor="priya"
        canApprove={false}
        deciding={false}
        onDecide={vi.fn()}
      />,
    )
    expect(screen.getByText('[m2-status.md#p.2]')).toBeInTheDocument()
  })

  it("the approval card's buttons are disabled for a non-approver", () => {
    render(
      <AssistantMessage
        turn={turn({
          status: 'paused',
          approval: {
            type: 'approval_required',
            trace_id: 'run-1',
            tool_name: 'create_risk',
            arguments: { project_id: 'atlas' },
            summary: 'create_risk(project_id=atlas)',
            actor: 'priya',
          },
        })}
        actor="tomas"
        canApprove={false}
        deciding={false}
        onDecide={vi.fn()}
      />,
    )
    expect(screen.getByRole('button', { name: 'Approve' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Deny' })).toBeDisabled()
    expect(screen.getByText(/switch to an approver/)).toBeInTheDocument()
  })

  it('the approval card is enabled for an approver', () => {
    render(
      <AssistantMessage
        turn={turn({
          status: 'paused',
          approval: {
            type: 'approval_required',
            trace_id: 'run-1',
            tool_name: 'create_risk',
            arguments: {},
            summary: 'create_risk()',
            actor: 'priya',
          },
        })}
        actor="priya"
        canApprove={true}
        deciding={false}
        onDecide={vi.fn()}
      />,
    )
    expect(screen.getByRole('button', { name: 'Approve' })).toBeEnabled()
  })
})
