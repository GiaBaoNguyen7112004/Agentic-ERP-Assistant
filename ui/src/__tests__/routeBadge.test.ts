import { describe, expect, it } from 'vitest'
import { routeBadge } from '../lib/routeBadge'
import { initialTurn } from '../turnReducer'
import type { TurnView } from '../turnReducer'

function turn(overrides: Partial<TurnView>): TurnView {
  return { ...initialTurn, ...overrides }
}

// The labels are quoted verbatim in docs/manual-test.md §3; changing one
// means changing the handbook too. These tests pin them byte for byte.
describe('routeBadge', () => {
  it("'starting' while no answer has arrived", () => {
    expect(routeBadge(turn({ status: 'starting' }))).toEqual({ label: 'starting', tone: 'neutral' })
  })

  it("'running' while tokens stream", () => {
    expect(routeBadge(turn({ status: 'running', streamed: 'partial' }))).toEqual({
      label: 'running',
      tone: 'neutral',
    })
  })

  it("'waiting for approval' when paused", () => {
    expect(routeBadge(turn({ status: 'paused' }))).toEqual({
      label: 'waiting for approval',
      tone: 'warning',
    })
  })

  it("'documents' for an answer without observations", () => {
    expect(
      routeBadge(
        turn({
          status: 'done',
          answer: { type: 'answer', text: '', route: 'answer', failure: 'none', error_detail: null, citations: [] },
        }),
      ),
    ).toEqual({ label: 'documents', tone: 'info' })
  })

  it("'tool' for an answer that made observations", () => {
    expect(
      routeBadge(
        turn({
          status: 'done',
          answer: { type: 'answer', text: '', route: 'answer', failure: 'none', error_detail: null, citations: [] },
          summary: {
            type: 'turn_finished',
            outcome: 'terminal',
            route: 'answer',
            failure: 'none',
            step_count: 1,
            evidence: [],
            memories_recalled: 0,
            history_shown: 0,
            observations: [{ tool: 'get_status', status: 'ok', attempts: 1 }],
            model_calls: { count: 1, input_tokens: 0, output_tokens: 0, cost_usd: null, unpriced: 0 },
          },
        }),
      ),
    ).toEqual({ label: 'tool', tone: 'info' })
  })

  it("'clarify', 'refused', 'failed' in the same precedence as before", () => {
    const answer = (route: string | null, failure = 'none') => ({
      type: 'answer' as const,
      text: '',
      route,
      failure,
      error_detail: null,
      citations: [],
    })
    expect(routeBadge(turn({ status: 'done', answer: answer('clarify') }))).toEqual({
      label: 'clarify',
      tone: 'neutral',
    })
    expect(routeBadge(turn({ status: 'done', answer: answer('refuse') }))).toEqual({
      label: 'refused',
      tone: 'warning',
    })
    expect(routeBadge(turn({ status: 'done', answer: answer('fail') }))).toEqual({
      label: 'failed',
      tone: 'destructive',
    })
    // a failure with a non-fail route still reads 'failed'
    expect(routeBadge(turn({ status: 'done', answer: answer('answer', 'tool_failure') }))).toEqual({
      label: 'failed',
      tone: 'destructive',
    })
  })

  it("'error' status before any answer", () => {
    expect(routeBadge(turn({ status: 'error', error: 'boom' }))).toEqual({
      label: 'error',
      tone: 'destructive',
    })
  })

  it("'unknown' for a route the map does not name", () => {
    expect(
      routeBadge(
        turn({
          status: 'done',
          answer: { type: 'answer', text: '', route: 'something_new', failure: 'none', error_detail: null, citations: [] },
        }),
      ),
    ).toEqual({ label: 'something_new', tone: 'neutral' })
  })
})