import { CornerDownRight } from 'lucide-react'
import { traceTone } from '@/lib/traceTone'
import { cn } from '@/lib/utils'
import type { TraceRow } from '../../protocol'

const toneTextClass: Record<string, string> = {
  neutral: 'text-foreground',
  info: 'text-info',
  success: 'text-success',
  warning: 'text-warning',
  destructive: 'text-destructive',
  memory: 'text-memory',
}

/**
 * One trace row, rendered as a single line inside a node card -- the same
 * kind/detail/tone reading `EventRow` gives the raw table, just without the
 * table chrome. Gateway rows (no `seq`) keep the same live-only marker.
 */
export function RowLine({ row }: { row: TraceRow }) {
  const gateway = row.source === 'tool_gateway'
  const tone = traceTone(row.kind)

  return (
    <div
      className={cn('flex flex-wrap items-baseline gap-1.5 text-xs', gateway && 'text-muted-foreground italic')}
      data-source={row.source}
      title={gateway ? 'live only — tool gateway hook, not persisted' : undefined}
    >
      {gateway && <CornerDownRight className="size-3 shrink-0" aria-hidden />}
      <span className={cn('font-mono font-semibold', toneTextClass[tone])}>{row.kind}</span>
      <span className="min-w-0 break-words text-muted-foreground">{row.detail}</span>
    </div>
  )
}
