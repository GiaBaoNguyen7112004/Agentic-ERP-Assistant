import { Activity, ExternalLink } from 'lucide-react'
import { DecisionList } from './DecisionList'
import { EventRow } from './EventRow'
import { TurnSummary } from './TurnSummary'
import { EmptyState } from '@/components/shared/EmptyState'
import { MonoId } from '@/components/shared/MonoId'
import { RouteBadge } from '@/components/shared/RouteBadge'
import { SectionHeading } from '@/components/shared/SectionHeading'
import { Table, TableBody, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { Message } from '../appState'

export function TurnTrace({ message }: { message: Extract<Message, { role: 'assistant' }> }) {
  const { turn, traceId } = message
  const runUrl = traceId ? `/api/runs/${traceId}` : null

  // A turn restored from session history carries no live trace events --
  // say so instead of showing an empty table (SessionTurn has no events).
  const historyOnly = turn.events.length === 0 && turn.status !== 'starting'

  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <SectionHeading>Trace</SectionHeading>
        <div className="flex flex-wrap items-center gap-2">
          <RouteBadge turn={turn} />
          {traceId ? (
            <MonoId value={traceId} href={runUrl ?? undefined} copy />
          ) : (
            <span className="font-mono text-xs text-muted-foreground">(starting…)</span>
          )}
        </div>
      </div>

      {turn.decisions.length > 0 && (
        <div className="space-y-2">
          <SectionHeading count={turn.decisions.length}>Decisions</SectionHeading>
          <DecisionList decisions={turn.decisions} />
        </div>
      )}

      {turn.events.length > 0 && (
        <div className="space-y-2">
          <SectionHeading count={turn.events.length}>Events</SectionHeading>
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
                {turn.events.map((event, index) => (
                  <EventRow key={`${event.seq ?? 'gw'}-${index}`} event={event} />
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}

      {historyOnly && (
        <EmptyState
          icon={Activity}
          title="No live trace for a turn loaded from history"
          hint={
            runUrl && (
              <a
                href={runUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 underline outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
              >
                open the run
                <ExternalLink className="size-3" aria-hidden />
              </a>
            )
          }
        />
      )}

      {turn.summary && <TurnSummary summary={turn.summary} />}
    </section>
  )
}