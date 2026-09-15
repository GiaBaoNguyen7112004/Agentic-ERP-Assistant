import { ToneBadge } from '@/components/shared/ToneBadge'
import { KeyValueList } from '@/components/shared/KeyValueList'
import { formatCost, formatDuration } from '@/lib/format'
import type { TurnFinishedEvent } from '../protocol'

export function TurnSummary({ summary }: { summary: TurnFinishedEvent }) {
  const { model_calls: calls } = summary

  return (
    <KeyValueList
      className="grid-cols-[max-content_max-content] border-t pt-3 text-xs text-muted-foreground"
      items={[
        {
          key: 'outcome',
          value: (
            <ToneBadge tone={summary.outcome === 'terminal' ? 'success' : 'warning'}>
              {summary.outcome}
            </ToneBadge>
          ),
        },
        { key: 'route', value: <span className="font-mono">{summary.route ?? '—'}</span> },
        { key: 'steps', value: <span className="tabular-nums">{summary.step_count}</span> },
        {
          key: 'duration',
          value: <span className="tabular-nums">{formatDuration(summary.started_at, summary.finished_at)}</span>,
        },
        {
          key: 'evidence',
          value: (
            <span className="tabular-nums" title={summary.evidence.join(', ')}>
              {summary.evidence.length}
            </span>
          ),
        },
        {
          key: 'memories recalled',
          value: <span className="tabular-nums">{summary.memories_recalled}</span>,
        },
        {
          key: 'history shown',
          value: <span className="tabular-nums">{summary.history_shown}</span>,
        },
        { key: 'model calls', value: <span className="tabular-nums">{calls.count}</span> },
        {
          key: 'tokens',
          value: (
            <span className="font-mono tabular-nums">
              {calls.input_tokens}/{calls.output_tokens}
            </span>
          ),
        },
        { key: 'cost', value: <span className="font-mono">{formatCost(calls.cost_usd)}</span> },
        ...(calls.unpriced > 0
          ? [{ key: 'unpriced', value: <span className="tabular-nums">{calls.unpriced}</span> }]
          : []),
        ...(summary.observations.length > 0
          ? [
              {
                key: 'calls',
                value: (
                  <span className="font-mono">
                    {summary.observations
                      .map(
                        (o) =>
                          `${o.tool} (${o.status}${o.attempts > 1 ? `, ${o.attempts} attempts` : ''})`,
                      )
                      .join(', ')}
                  </span>
                ),
              },
            ]
          : []),
      ]}
    />
  )
}