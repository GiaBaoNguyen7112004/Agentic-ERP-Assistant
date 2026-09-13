import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ContextBlock } from '../components/trace/ContextBlock'
import type { ContextEvent } from '../protocol'

function context(overrides: Partial<ContextEvent> = {}): ContextEvent {
  return {
    type: 'context', request: 'hi', history: [], memories: [], contract: null, model_calls: [],
    ...overrides,
  }
}

describe('ContextBlock', () => {
  it('renders "no prior turns" and "nothing remembered" for an empty context', () => {
    render(<ContextBlock context={context()} />)
    expect(screen.getByText('no prior turns in this session')).toBeInTheDocument()
    expect(screen.getByText('nothing remembered that bears on this')).toBeInTheDocument()
  })

  it('renders "unchecked" when the contract is null', () => {
    render(<ContextBlock context={context()} />)
    expect(screen.getByText(/unchecked/)).toBeInTheDocument()
  })

  it('renders a history turn, a memory, and a declared contract', () => {
    render(
      <ContextBlock
        context={context({
          history: [
            {
              trace_id: 'run-0',
              request: 'What is the status of M1?',
              response: 'On track.',
              route: 'answer',
              failure: 'none',
              tool_name: null,
              approval: 'not_required',
              started_at: '2026-01-01T00:00:00Z',
              finished_at: '2026-01-01T00:00:01Z',
            },
          ],
          memories: [
            {
              memory_id: 'mem-1',
              kind: 'preference',
              key: 'reply_language',
              statement: 'Prefers replies in Vietnamese.',
              confidence: 0.9,
              recorded_in_run: 'run-0',
              recorded_at: '2026-01-01T00:00:00Z',
              supersedes: [],
              links: [],
              required_scope: 'project.docs.read',
              actor: 'priya',
              project_code: 'atlas',
              session_id: 'sess-1',
            },
          ],
          contract: { needs: ['document_passage'], document_query: 'why milestone M2 is late' },
        })}
      />,
    )

    expect(screen.getByText('What is the status of M1?')).toBeInTheDocument()
    expect(screen.getByText('On track.')).toBeInTheDocument()
    expect(screen.getByText('Prefers replies in Vietnamese.')).toBeInTheDocument()
    expect(screen.getByText('document_passage')).toBeInTheDocument()
    expect(screen.getByText('why milestone M2 is late')).toBeInTheDocument()
  })
})
