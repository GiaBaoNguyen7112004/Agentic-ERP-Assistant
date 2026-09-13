import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import type { TraceRow } from '../../protocol'
import { RowLine } from './RowLine'

/**
 * A section of rows that sit outside the numbered node executions -- the
 * orchestrator's own work before the graph ran (recall, the reply
 * contract) and after it finished (consolidation). Collapsed by default:
 * unlike a node card, there is usually nothing here to look at, and the
 * common case (no memory layer, no session) is an empty section.
 */
export function PhaseCard({
  title,
  rows,
  defaultOpen = false,
  extra,
}: {
  title: string
  rows: TraceRow[]
  defaultOpen?: boolean
  extra?: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  if (rows.length === 0 && !extra) return null

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-lg border border-dashed">
      <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-1.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring/50">
        {open ? (
          <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
        ) : (
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
        )}
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">{title}</span>
        {rows.length > 0 && (
          <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-muted-foreground">
            {rows.length}
          </span>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-1.5 border-t px-3 py-2">
        {extra}
        {rows.map((row, index) => (
          <RowLine key={`${row.seq ?? 'gw'}-${index}`} row={row} />
        ))}
      </CollapsibleContent>
    </Collapsible>
  )
}
