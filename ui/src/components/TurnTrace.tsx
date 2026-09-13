import { Activity, ExternalLink } from 'lucide-react'
import { buildExecutionTree } from '@/lib/executionTree'
import { EmptyState } from '@/components/shared/EmptyState'
import { MonoId } from '@/components/shared/MonoId'
import { RouteBadge } from '@/components/shared/RouteBadge'
import { ContextBlock } from './trace/ContextBlock'
import { MemoryAuditBlock } from './trace/MemoryAuditBlock'
import { NodeCard } from './trace/NodeCard'
import { PhaseCard } from './trace/PhaseCard'
import { RawEventsTable } from './trace/RawEventsTable'
import { TurnSummary } from './TurnSummary'
import type { Message } from '../appState'

/**
 * The execution inspector: a run's node executions, in order, with what each
 * one received, did, and produced -- built from the same stream the old flat
 * event table read, grouped by `lib/executionTree.ts`. See
 * `docs/trace-inspector-plan.md` for the design this implements and the
 * later phases that widen what a node card can show.
 */
export function TurnTrace({ message }: { message: Extract<Message, { role: 'assistant' }> }) {
  const { turn, traceId } = message
  const runUrl = traceId ? `/api/runs/${traceId}` : null

  // A turn restored from session history carries no live trace events --
  // say so instead of showing an empty tree (SessionTurn has no events).
  const historyOnly = turn.events.length === 0 && turn.status !== 'starting'
  const tree = buildExecutionTree(turn)

  // The rich ContextEvent, when the stream carried one, replaces the flat
  // prelude rows entirely (it says what was recalled, not just that
  // something was) -- see ContextEvent's own doc for why both still exist
  // on the wire. Consolidation is split the other way: memory decisions get
  // the rich MemoryAuditBlock once the run has filed (turn_finished carries
  // it), but history_promoted has no rich payload of its own yet, so its
  // flat row is always kept alongside.
  const memoryAudit = turn.summary?.memory_audit ?? null
  const consolidationRows = memoryAudit
    ? tree.consolidation.filter((row) => row.kind !== 'memory_written' && row.kind !== 'memory_rejected')
    : tree.consolidation

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <RouteBadge turn={turn} />
        {traceId ? (
          <MonoId value={traceId} copy />
        ) : (
          <span className="font-mono text-xs text-muted-foreground">(starting…)</span>
        )}
      </div>

      {turn.summary && <TurnSummary summary={turn.summary} />}

      {!historyOnly && (tree.prelude.length > 0 || tree.entries.length > 0 || tree.consolidation.length > 0 || turn.context || turn.events.length > 0) && (
        <div className="space-y-2">
          <PhaseCard
            title="Context"
            rows={turn.context ? [] : tree.prelude}
            extra={turn.context ? <ContextBlock context={turn.context} /> : undefined}
          />
          {tree.entries.map((entry, index) => (
            <NodeCard key={index} entry={entry} />
          ))}
          <PhaseCard
            title="Consolidation"
            rows={consolidationRows}
            extra={memoryAudit && memoryAudit.length > 0 ? <MemoryAuditBlock rows={memoryAudit} /> : undefined}
          />
          <RawEventsTable events={turn.events} />
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
    </section>
  )
}
