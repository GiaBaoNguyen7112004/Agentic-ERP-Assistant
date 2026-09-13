import { ChevronDown, ChevronRight, LoaderCircle } from 'lucide-react'
import { useState } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { JsonBlock } from '@/components/shared/JsonBlock'
import { ToneBadge } from '@/components/shared/ToneBadge'
import { nodeLabel } from '@/lib/nodeLabel'
import { traceTone } from '@/lib/traceTone'
import type { Tone } from '@/lib/routeBadge'
import { entryToolCall, type ExecutionEntry } from '@/lib/executionTree'
import { EvidenceBlock } from './EvidenceBlock'
import { RowLine } from './RowLine'
import { ToolOutcomeBlock } from './ToolOutcomeBlock'
import { TransitionRow } from './TransitionRow'

const TONE_RANK: Record<Tone, number> = {
  neutral: 0,
  info: 1,
  success: 1,
  memory: 1,
  warning: 2,
  destructive: 3,
}

/** The worst (most attention-worthy) tone among a span's own rows -- what
 * colors the card's header badge. */
function worstTone(rows: ExecutionEntry['rows']): Tone {
  let worst: Tone = 'neutral'
  for (const row of rows) {
    const tone = traceTone(row.kind)
    if (TONE_RANK[tone] > TONE_RANK[worst]) worst = tone
  }
  return worst
}

/**
 * One node execution, or one bare engine-level step -- a card the developer
 * reads top to bottom: what ran, what it did, and why the graph moved on.
 *
 * Open by default (the point of the panel is what happened, not that
 * something happened), and numbered only for real node executions --
 * engine-level cards (an approval being recorded, a denial's refusal) sit
 * between numbered nodes without claiming an ordinal of their own.
 */
export function NodeCard({ entry }: { entry: ExecutionEntry }) {
  const [open, setOpen] = useState(true)
  const label = nodeLabel(entry.nodeName)
  const streaming = entry.step === null
  const tone = worstTone(entry.rows)
  const toolCall = entryToolCall(entry)

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-lg border">
      <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring/50">
        {open ? (
          <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
        ) : (
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
        )}
        {entry.kind === 'node' && (
          <span className="shrink-0 rounded-md bg-muted px-1.5 py-0.5 font-mono text-[10px] font-semibold tabular-nums text-muted-foreground">
            {entry.ordinal}
          </span>
        )}
        <span className="shrink-0 text-sm font-medium">{label}</span>
        {entry.nodeName && (
          <span className="shrink-0 font-mono text-[11px] text-muted-foreground">({entry.nodeName})</span>
        )}
        {toolCall && (
          <span className="min-w-0 truncate font-mono text-xs text-foreground">→ {toolCall.name}</span>
        )}
        <span className="ml-auto flex shrink-0 items-center gap-1.5">
          {streaming && <LoaderCircle className="size-3.5 animate-spin text-muted-foreground" aria-hidden />}
          {tone !== 'neutral' && !streaming && (
            <ToneBadge tone={tone} className="text-[10px]">
              {entry.rows.length} row{entry.rows.length === 1 ? '' : 's'}
            </ToneBadge>
          )}
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-2 border-t px-3 py-2">
        {entry.rows.length > 0 ? (
          <div className="space-y-1">
            {entry.rows.map((row, index) => (
              <RowLine key={`${row.seq ?? 'gw'}-${index}`} row={row} />
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">no new trace rows</p>
        )}
        {entry.step?.evidence && (
          <div className="space-y-1">
            <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Evidence</p>
            <EvidenceBlock evidence={entry.step.evidence} />
          </div>
        )}
        {entry.step && entry.step.observations.length > 0 && (
          <div className="space-y-1">
            <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Result</p>
            <ToolOutcomeBlock observations={entry.step.observations} />
          </div>
        )}
        {toolCall?.arguments && (
          <div className="space-y-1">
            <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Call</p>
            <JsonBlock value={toolCall.arguments} />
          </div>
        )}
        <TransitionRow entry={entry} />
      </CollapsibleContent>
    </Collapsible>
  )
}
