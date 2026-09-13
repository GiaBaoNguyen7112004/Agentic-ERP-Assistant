import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { TracePanel } from '../components/TracePanel'
import { hydrateTurn } from '../hydrate'
import { initialTurn, turnReducer, type TurnView } from '../turnReducer'
import type { Message } from '../appState'
import type { RunReport, ServerEvent, StepEvent, TraceRow } from '../protocol'

// TracePanel fetches the filed run through `api.getRun`; every test mocks
// the module so no network is touched.
vi.mock('../api', () => ({ getRun: vi.fn() }))
import { getRun } from '../api'

const STAMP = '2026-01-01T00:00:00Z'

function assistant(traceId: string | null, turn = initialTurn): Message {
  return { role: 'assistant', id: traceId ?? 'assistant-1', traceId, turn }
}

function liveTurn(traceId: string): TurnView {
  // The smallest settled live view: one event row and one step.
  let view = turnReducer(initialTurn, {
    type: 'turn_started',
    trace_id: traceId,
    session_id: 'sess-1',
    actor: 'priya',
    resumed: false,
  } satisfies ServerEvent)
  view = turnReducer(view, {
    type: 'trace',
    seq: 0,
    node: 'start',
    kind: 'node_entered',
    detail: '',
    source: 'engine',
    step: null,
  } satisfies TraceRow)
  return turnReducer(view, {
    type: 'step',
    route: 'answer',
    tool_name: null,
    tool_arguments: null,
    tool_mutating: null,
    approval: 'not_required',
    step_count: 1,
    terminal: true,
    node: 'start',
    elapsed_ms: 5,
    evidence: null,
    observations: [],
    response: 'done.',
    failure: 'none',
    error_detail: null,
    draft: null,
    redirected_needs: [],
    retry_count: 0,
    retrieval: null,
    model_calls: [],
  } satisfies StepEvent)
}

function aReport(): RunReport {
  return {
    run: {
      trace_id: 'run-9',
      actor: 'priya',
      project_code: 'atlas',
      outcome: 'terminal',
      started_at: STAMP,
      finished_at: STAMP,
      state: {
        request: 'Why is M2 late?',
        actor: 'priya',
        project_code: 'atlas',
        scopes: ['project.status.read'],
        trace_id: 'run-9',
        session_id: 'sess-1',
        route: 'answer',
        evidence: [],
        memories: [],
        history: [],
        contract: null,
        redirected_needs: [],
        draft: null,
        observations: [],
        tool_name: null,
        tool_arguments: null,
        tool_mutating: false,
        approval: 'not_required',
        response: 'M2 slipped two days.',
        failure: 'none',
        error_detail: null,
        terminal: true,
        step_count: 1,
        retry_count: 0,
        events: [{ node: 'start', kind: 'node_entered', detail: '' }],
        state_version: 2,
      },
    },
    events: [{ node: 'start', kind: 'node_entered', detail: '' }],
    audit_rows: [],
    model_calls: { count: 0, input_tokens: 0, output_tokens: 0, cost_usd: 0, unpriced: 0, records: [] },
    memory_audit: [],
    answer: { text: 'M2 slipped two days.', citations: [] },
  }
}

beforeEach(() => {
  vi.mocked(getRun).mockReset()
})

describe('TracePanel', () => {
  it('shows the newest assistant turn by default', () => {
    const older = assistant('run-1', liveTurn('run-1'))
    const newer = assistant('run-2', liveTurn('run-2'))
    render(<TracePanel messages={[older, newer]} inspectedId={null} onInspect={vi.fn()} />)

    expect(screen.getByText('run-2')).toBeInTheDocument()
  })

  it('inspects the bubble the user clicked, and offers the way back', () => {
    const older = assistant('run-1', liveTurn('run-1'))
    const newer = assistant('run-2', liveTurn('run-2'))
    render(<TracePanel messages={[older, newer]} inspectedId="run-1" onInspect={vi.fn()} />)

    expect(screen.getByText('run-1')).toBeInTheDocument()
    expect(screen.queryByText('run-2')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /back to the newest turn/ })).toBeInTheDocument()
  })

  it('hydrates a settled turn that carries no events, and labels the view reconstructed', async () => {
    vi.mocked(getRun).mockResolvedValue(aReport())
    const empty = assistant('run-9', {
      ...initialTurn,
      status: 'done',
      answer: null,
    })
    render(<TracePanel messages={[empty]} inspectedId={null} onInspect={vi.fn()} />)

    await waitFor(() => expect(getRun).toHaveBeenCalledWith('run-9'))
    await waitFor(() => expect(screen.getByText(/Reconstructed from the filed run/)).toBeInTheDocument())
    expect(screen.getByText('run-9')).toBeInTheDocument()
  })

  it('does not hydrate a live turn that already has events', () => {
    const live = assistant('run-2', liveTurn('run-2'))
    render(<TracePanel messages={[live]} inspectedId={null} onInspect={vi.fn()} />)

    expect(getRun).not.toHaveBeenCalled()
  })

  it('falls back to the empty state with the run link when the filed run cannot be read', async () => {
    vi.mocked(getRun).mockRejectedValue(new Error('no database'))
    const empty = assistant('run-9', { ...initialTurn, status: 'done' })
    render(<TracePanel messages={[empty]} inspectedId={null} onInspect={vi.fn()} />)

    await waitFor(() =>
      expect(screen.getByText('No live trace for a turn loaded from history')).toBeInTheDocument(),
    )
    expect(screen.getByText(/open the run/)).toBeInTheDocument()
  })

  it('hydrated content goes through the same tree builder as a live view', async () => {
    vi.mocked(getRun).mockResolvedValue(aReport())
    const empty = assistant('run-9', { ...initialTurn, status: 'done' })
    const { container } = render(
      <TracePanel messages={[empty]} inspectedId={null} onInspect={vi.fn()} />,
    )

    await waitFor(() => expect(container.textContent).toContain('Reconstructed'))
    // The planner node card renders from the filed event log.
    expect(screen.getByText(/Planner/)).toBeInTheDocument()
    expect(hydrateTurn(aReport()).turn.traceId).toBe('run-9')
  })
})