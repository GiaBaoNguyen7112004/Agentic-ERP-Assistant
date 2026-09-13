import type { TraceRow } from '../protocol'

/**
 * Which of a node span's own rows best explains why the graph moved on, in
 * priority order. A span can carry several of these at once -- a `think`
 * node that redirects a need emits `route_selected` for the redirected call
 * *and* `contract_enforced` for the redirect itself -- and the reader wants
 * the row that names the actual decision, not the mechanical one beside it.
 */
const REASON_PRIORITY: readonly string[] = [
  'route_selected',
  'contract_enforced',
  'tool_called',
  'failed',
  'rate_limited',
  'approval_requested',
]

/** The detail of the highest-priority, most recent matching row -- or
 * `null` when a span carries none of them (a bare engine card with no rows
 * at all, e.g. the second half of a denied approval). */
export function transitionReason(rows: readonly TraceRow[]): string | null {
  for (const kind of REASON_PRIORITY) {
    for (let i = rows.length - 1; i >= 0; i -= 1) {
      if (rows[i].kind === kind) return rows[i].detail
    }
  }
  return null
}
