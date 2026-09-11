import type { TurnFinishedEvent } from '../protocol'

export function TurnSummary({ summary }: { summary: TurnFinishedEvent }) {
  const { model_calls: calls } = summary

  return (
    <div className="turn-summary">
      <div>
        outcome: {summary.outcome} · route: {summary.route ?? '—'} · steps:{' '}
        {summary.step_count}
      </div>
      <div>
        evidence: {summary.evidence.length} · memories recalled:{' '}
        {summary.memories_recalled} · history shown: {summary.history_shown}
      </div>
      <div>
        model calls: {calls.count} · tokens {calls.input_tokens}/{calls.output_tokens} · cost{' '}
        {calls.cost_usd === null ? 'unknown' : `$${calls.cost_usd.toFixed(4)}`}
        {calls.unpriced > 0 && ` (${calls.unpriced} unpriced)`}
      </div>
      {summary.observations.length > 0 && (
        <div>
          calls:{' '}
          {summary.observations
            .map((o) => `${o.tool} (${o.status}${o.attempts > 1 ? `, ${o.attempts} attempts` : ''})`)
            .join(', ')}
        </div>
      )}
    </div>
  )
}
