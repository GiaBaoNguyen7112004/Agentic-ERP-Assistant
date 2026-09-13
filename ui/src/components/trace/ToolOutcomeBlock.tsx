import { ToneBadge } from '@/components/shared/ToneBadge'
import { TextBlock } from '@/components/shared/TextBlock'
import type { Tone } from '@/lib/routeBadge'
import type { ToolOutcomeOut } from '../../protocol'

const STATUS_TONE: Record<string, Tone> = {
  ok: 'success',
  invalid_arguments: 'destructive',
  approval_required: 'warning',
  denied: 'warning',
  rate_limited: 'warning',
  transient_failure: 'destructive',
  failed: 'destructive',
}

/** What one tool call returned -- `state/tool_outcome.py::ToolOutcome`, in
 * full: not just the status a chat chip shows, but the summary, what it
 * read or changed, and why it failed when it did. */
function ToolOutcomeItem({ outcome }: { outcome: ToolOutcomeOut }) {
  return (
    <div className="space-y-1 rounded-md border bg-muted/30 p-2 text-xs">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-mono font-medium">{outcome.tool_name}</span>
        <ToneBadge tone={STATUS_TONE[outcome.status] ?? 'neutral'}>{outcome.status}</ToneBadge>
        {outcome.attempts > 1 && <span className="text-muted-foreground">{outcome.attempts} attempts</span>}
        {outcome.retry_after_seconds !== null && (
          <span className="text-muted-foreground">retry in {outcome.retry_after_seconds.toFixed(1)}s</span>
        )}
      </div>
      <p className="font-mono text-muted-foreground">{outcome.arguments_summary}</p>
      {outcome.status === 'ok' ? (
        <>
          <TextBlock text={outcome.summary.text} />
          {outcome.source_ids.length > 0 && (
            <p className="text-muted-foreground">source_ids: {outcome.source_ids.join(', ')}</p>
          )}
        </>
      ) : (
        outcome.error && <p className="text-destructive">{outcome.error}</p>
      )}
    </div>
  )
}

/** What this step's own node added to the turn's observations -- never the
 * whole accumulated list (see `StepEvent.observations`'s own doc). */
export function ToolOutcomeBlock({ observations }: { observations: ToolOutcomeOut[] }) {
  if (observations.length === 0) return null
  return <div className="space-y-1.5">{observations.map((outcome, index) => <ToolOutcomeItem key={index} outcome={outcome} />)}</div>
}
