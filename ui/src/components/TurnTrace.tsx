import { Activity, ExternalLink, History } from 'lucide-react'
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
import type { ToolOutcomeOut } from '../protocol'
import type { TurnView } from '../turnReducer'
import type { Message } from '../appState'

/**
 * The execution inspector: a run's node executions, in order, with what each
 * one received, did, and produced -- built from the same stream the old flat
 * event table read, grouped by `lib/executionTree.ts`. See
 * `docs/trace-inspector-plan.md` for the design this implements.
 *
 * A view may also be *hydrated* from a filed run (`hydrate.ts`) -- a turn
 * loaded from history, or a queue-resumed decision's fresh bubble -- in which
 * case `turn` carries the reconstructed view and `reconstructed` makes the
 * panel say so: its attribution is best-effort, not the stream's word.
 */
export function TurnTrace({
  message,
  turn: hydrated,
  reconstructed = false,
  unattributed = [],
}: {
  message: Extract<Message, { role: 'assistant' }>
  /** A reconstructed view from the filed run, when the panel hydrated one
   * (Phase 4); `undefined` inspects the message's own streamed view. */
  turn?: TurnView
  reconstructed?: boolean
  /** Outcomes the reconstruction could not place on any span. */
  unattributed?: ToolOutcomeOut[]
}) {
  const view = hydrated ?? message.turn
  const { turn, traceId } = { turn: view, traceId: view.traceId }
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

      {reconstructed && (
        <p className="flex items-start gap-1.5 text-xs text-muted-foreground">
          <History className="mt-0.5 size-3 shrink-0" aria-hidden />
          <span>
            Reconstructed from the filed run -- node attribution is best-effort, never
            the stream's word (docs/trace-inspector-plan.md §6.3).
            {unattributed.length > 0 &&
              ` ${unattributed.length} tool result${unattributed.length === 1 ? '' : 's'} could not be attributed to a node.`}
          </span>
        </p>
      )}

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
