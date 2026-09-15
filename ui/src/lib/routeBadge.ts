import type { TurnView } from '../turnReducer'

export type Tone = 'neutral' | 'info' | 'success' | 'warning' | 'destructive' | 'memory'

export interface RouteBadgeView {
  label: string
  tone: Tone
}

/**
 * The route badge's label AND tone for a turn, as one pure function.
 * Labels are byte-identical to the pre-redesign strings -- they are quoted
 * verbatim in docs/manual-test.md §3 and asserted in the routeBadge tests.
 */
export function routeBadge(turn: TurnView): RouteBadgeView {
  if (turn.status === 'paused') return { label: 'waiting for approval', tone: 'warning' }
  if (!turn.answer) {
    // starting / running / error carry their own treatment
    return { label: turn.status, tone: turn.status === 'error' ? 'destructive' : 'neutral' }
  }
  const { route, failure } = turn.answer
  if (route === 'refuse') return { label: 'refused', tone: 'warning' }
  // ADR 0021: the planner's reply was delivered after a redirected search
  // could not confirm it -- an answer with a caveat, never 'failed'.
  if (route === 'answer' && failure === 'incomplete_reply') {
    return { label: 'incomplete', tone: 'warning' }
  }
  if (route === 'fail' || failure !== 'none') return { label: 'failed', tone: 'destructive' }
  if (route === 'clarify') return { label: 'clarify', tone: 'neutral' }
  if (route === 'answer') {
    const usedTool = (turn.summary?.observations.length ?? 0) > 0
    return usedTool ? { label: 'tool', tone: 'info' } : { label: 'documents', tone: 'info' }
  }
  return { label: route ?? 'unknown', tone: 'neutral' }
}

/** The tone of a planner decision's route, for the decision list's badge. */
export function routeTone(route: string | null): Tone {
  if (route === 'refuse') return 'warning'
  if (route === 'fail') return 'destructive'
  if (route === 'answer') return 'info'
  return 'neutral'
}