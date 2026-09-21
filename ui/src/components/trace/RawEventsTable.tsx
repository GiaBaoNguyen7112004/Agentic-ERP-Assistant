import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Table, TableBody, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { EventRow } from '../EventRow'
import type { TraceRow } from '../../protocol'

/**
 * The flat event log, preserved verbatim (same `EventRow`, same columns) for
 * anyone who wants the underlying rows rather than the grouped execution
 * view -- and for `docs/manual-test.md` E1/E2, which check every row against
 * `trace_events` directly. Collapsed by default: the node cards above are
 * the same information, organized by what produced it.
 */
export function RawEventsTable({ events }: { events: TraceRow[] }) {
  const [open, setOpen] = useState(false)
  if (events.length === 0) return null

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="space-y-2">
      <CollapsibleTrigger className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50">
        {open ? (
          <ChevronDown className="size-3" aria-hidden />
        ) : (
          <ChevronRight className="size-3" aria-hidden />
        )}
        Raw events
        <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-muted-foreground">
          {events.length}
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="rounded-lg border">
          <Table className="text-xs [&_th]:py-1.5 [&_td]:py-1">
            <TableHeader>
              <TableRow>
                <TableHead className="w-8 pl-2">#</TableHead>
                <TableHead>node</TableHead>
                <TableHead>kind</TableHead>
                <TableHead>detail</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {events.map((event, index) => (
                <EventRow key={`${event.seq ?? 'gw'}-${index}`} event={event} />
              ))}
            </TableBody>
          </Table>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
