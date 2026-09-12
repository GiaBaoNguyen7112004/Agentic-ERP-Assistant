import { CornerDownRight } from 'lucide-react'
import { TableCell, TableRow } from '@/components/ui/table'
import { traceTone } from '@/lib/traceTone'
import { cn } from '@/lib/utils'
import type { TraceRow } from '../protocol'

const toneTextClass: Record<string, string> = {
  neutral: 'text-foreground',
  info: 'text-info',
  success: 'text-success',
  warning: 'text-warning',
  destructive: 'text-destructive',
  memory: 'text-memory',
}

export function EventRow({ event }: { event: TraceRow }) {
  const gateway = event.source === 'tool_gateway'
  const tone = traceTone(event.kind)

  return (
    <TableRow
      data-source={event.source}
      className={cn('text-xs', gateway && 'text-muted-foreground italic')}
      title={gateway ? 'live only — tool gateway hook, not persisted' : undefined}
    >
      <TableCell className="w-8 py-1 pl-2 tabular-nums text-muted-foreground">
        {event.seq ?? '·'}
      </TableCell>
      <TableCell className="py-1 whitespace-nowrap">
        {gateway && <CornerDownRight className="mr-1 inline size-3" aria-hidden />}
        {event.node}
      </TableCell>
      <TableCell className={cn('py-1 font-mono font-semibold', toneTextClass[tone])}>
        {event.kind}
      </TableCell>
      <TableCell className="min-w-40 py-1 whitespace-normal break-words">
        {event.detail}
      </TableCell>
    </TableRow>
  )
}