import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { ModelCallBlock } from '../components/trace/ModelCallBlock'
import type { ModelCallOut } from '../protocol'

function call(overrides: Partial<ModelCallOut> = {}): ModelCallOut {
  return {
    model: 'gpt-4o',
    outcome: 'routed',
    estimated_input_tokens: 100,
    input_tokens: 90,
    output_tokens: 10,
    cost_usd: 0.01,
    latency_seconds: 0.8,
    attempts: 1,
    occurred_at: '2026-01-01T00:00:00Z',
    detail: null,
    request: null,
    response: null,
    ...overrides,
  }
}

describe('ModelCallBlock', () => {
  it('renders the cost line', () => {
    render(<ModelCallBlock call={call()} />)
    expect(screen.getByText('gpt-4o')).toBeInTheDocument()
    expect(screen.getByText('routed')).toBeInTheDocument()
    expect(screen.getByText('90/10 tok')).toBeInTheDocument()
    expect(screen.getByText('$0.0100')).toBeInTheDocument()
  })

  it('shows the DEV_TRACE_MODEL_IO hint when no request/response were captured', () => {
    render(<ModelCallBlock call={call()} />)
    expect(screen.getByText('DEV_TRACE_MODEL_IO=1')).toBeInTheDocument()
    expect(screen.queryByText('request & reply')).not.toBeInTheDocument()
  })

  it('expands the request and reply when they were captured', async () => {
    const user = userEvent.setup()
    render(
      <ModelCallBlock
        call={call({
          request: {
            kind: 'answer',
            messages: [{ role: 'user', content: { text: 'hi', truncated: false, chars: 2 } }],
            tools: [],
            tool_choice: null,
            temperature: 0,
          },
          response: {
            content: { text: 'hello', truncated: false, chars: 5 },
            tool_name: null,
            arguments: null,
            stop_reason: 'stop',
          },
        })}
      />,
    )

    expect(screen.queryByText('DEV_TRACE_MODEL_IO=1')).not.toBeInTheDocument()
    await user.click(screen.getByText('request & reply'))
    expect(screen.getByText('hi')).toBeInTheDocument()
    expect(screen.getByText('hello')).toBeInTheDocument()
  })

  it('renders a tool-call reply as a name and JSON arguments', async () => {
    const user = userEvent.setup()
    render(
      <ModelCallBlock
        call={call({
          request: {
            kind: 'tools', messages: [], tools: ['list_risks'], tool_choice: 'auto', temperature: 0,
          },
          response: {
            content: null, tool_name: 'list_risks', arguments: { project_id: 'atlas' }, stop_reason: null,
          },
        })}
      />,
    )
    await user.click(screen.getByText('request & reply'))
    expect(screen.getByText('→ list_risks')).toBeInTheDocument()
    expect(screen.getByText(/"project_id": "atlas"/)).toBeInTheDocument()
  })
})
