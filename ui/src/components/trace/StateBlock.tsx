import { ChevronDown, ChevronRight, Copy } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { JsonBlock } from '@/components/shared/JsonBlock'
import { SectionHeading } from '@/components/shared/SectionHeading'
import { TextBlock } from '@/components/shared/TextBlock'
import { ToneBadge } from '@/components/shared/ToneBadge'
import { changedFields, diffState, STATE_FIELD_GROUPS, type FieldChange } from '@/lib/stateDiff'
import type { AgentStateSnapshot } from '../../protocol'

const CHIP_LIMIT = 6

function renderScalar(value: unknown) {
  if (value === null || value === undefined) {
    return <span className="text-muted-foreground/70">null</span>
  }
  if (typeof value === 'boolean' || typeof value === 'number') {
    return <span className="font-mono">{String(value)}</span>
  }
  return <span className="font-mono break-words">{String(value)}</span>
}

/** Whether `value` should render as a one-line scalar (`before → after`)
 * rather than as two stacked JSON blocks. */
function isInlineScalar(value: unknown): boolean {
  if (value === null) return true
  const type = typeof value
  return type === 'string' || type === 'number' || type === 'boolean'
}

function FieldValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) return renderScalar(value)
  if (typeof value === 'boolean' || typeof value === 'number') return renderScalar(value)
  if (typeof value === 'string') {
    if (value.length === 0) return <span className="text-muted-foreground/70">""</span>
    if (value.length <= 120 && !value.includes('\n')) return renderScalar(value)
    return <TextBlock text={value} />
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="font-mono text-muted-foreground/70">[]</span>
    return <JsonBlock value={value} />
  }
  return <JsonBlock value={value} />
}

function FieldRow({ change }: { change: FieldChange }) {
  return (
    <div className="space-y-1 border-t py-1.5 first:border-t-0 first:pt-0" data-field={change.field} data-change={change.kind}>
      <div className="flex items-center gap-1.5">
        <span className="font-mono text-[11px] text-muted-foreground">{change.field}</span>
        {change.kind === 'appended' && (
          <span className="text-[10px] text-info">+{change.appended?.length ?? 0} appended</span>
        )}
        {change.kind === 'changed' && change.before !== null && (
          <span className="text-[10px] text-warning">changed</span>
        )}
      </div>

      {change.kind === 'appended' ? (
        <AppendedValue change={change} />
      ) : change.kind === 'changed' && isInlineScalar(change.before) && isInlineScalar(change.after) ? (
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-muted-foreground line-through decoration-muted-foreground/50">
            {renderScalar(change.before)}
          </span>
          <span className="text-muted-foreground">→</span>
          {renderScalar(change.after)}
        </div>
      ) : change.kind === 'changed' ? (
        <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
          <div className="space-y-0.5">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">before</p>
            <FieldValue value={change.before} />
          </div>
          <div className="space-y-0.5">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">after</p>
            <FieldValue value={change.after} />
          </div>
        </div>
      ) : (
        <FieldValue value={change.after} />
      )}
    </div>
  )
}

function AppendedValue({ change }: { change: FieldChange }) {
  const [showAll, setShowAll] = useState(false)
  const appended = change.appended ?? []
  const after = Array.isArray(change.after) ? change.after : []

  return (
    <div className="space-y-1">
      <JsonBlock value={showAll ? after : appended} />
      {after.length > appended.length && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-1.5 text-[11px] text-muted-foreground"
          onClick={() => setShowAll((value) => !value)}
        >
          {showAll ? `show only the ${appended.length} new` : `show all ${after.length}`}
        </Button>
      )}
    </div>
  )
}

/**
 * The whole `AgentState` a node returned (or the state the engine starts
 * from, on the Context card), diffed against the state before it.
 * Collapsed by default -- a full state is 20-30 KB of JSON -- with a
 * trigger line that already answers "what did this node touch" without
 * expanding anything (docs/agent-state-inspector-plan.md D6).
 */
export function StateBlock({
  state,
  previous,
  title = 'State',
}: {
  state: AgentStateSnapshot | null
  previous: AgentStateSnapshot | null
  title?: string
}) {
  const isInitial = previous === null
  const [open, setOpen] = useState(false)
  const [showAllFields, setShowAllFields] = useState(isInitial)

  if (state === null) {
    return (
      <p className="text-xs text-muted-foreground">
        State not carried -- a reconstructed run files only its final state
        (<span className="font-mono">runs.state</span>); see the last node.
      </p>
    )
  }

  const changes = diffState(previous, state)
  const changed = changedFields(changes)
  const visible = showAllFields ? changes : changed

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-md border bg-muted/20">
      <CollapsibleTrigger className="flex w-full flex-wrap items-center gap-1.5 px-2 py-1.5 text-left text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50">
        {open ? <ChevronDown className="size-3 shrink-0" aria-hidden /> : <ChevronRight className="size-3 shrink-0" aria-hidden />}
        <span className="font-medium">{title}</span>
        <ToneBadge tone={isInitial ? 'neutral' : changed.length > 0 ? 'info' : 'neutral'} className="text-[10px]">
          {isInitial ? 'initial' : `${changed.length} changed`}
        </ToneBadge>
        {!isInitial &&
          changed.slice(0, CHIP_LIMIT).map((change) => (
            <span key={change.field} className="font-mono text-[10px] text-muted-foreground">
              {change.field}
            </span>
          ))}
        {!isInitial && changed.length > CHIP_LIMIT && (
          <span className="text-[10px] text-muted-foreground">+{changed.length - CHIP_LIMIT}</span>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-2 border-t px-2 py-2">
        <div className="flex items-center justify-between gap-2">
          {isInitial ? (
            <span />
          ) : (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-1.5 text-[11px] text-muted-foreground"
              onClick={() => setShowAllFields((value) => !value)}
            >
              {showAllFields ? 'changed only' : 'all fields'}
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="h-6 gap-1 px-1.5 text-[11px] text-muted-foreground"
            onClick={() => {
              void navigator.clipboard?.writeText(JSON.stringify(state, null, 2))
            }}
          >
            <Copy className="size-3" aria-hidden />
            copy JSON
          </Button>
        </div>
        {STATE_FIELD_GROUPS.map((group) => {
          const rows = visible.filter((change) => group.fields.includes(change.field))
          if (rows.length === 0) return null
          return (
            <div key={group.title} className="space-y-1">
              <SectionHeading>{group.title}</SectionHeading>
              <div>
                {rows.map((change) => (
                  <FieldRow key={change.field} change={change} />
                ))}
              </div>
            </div>
          )
        })}
      </CollapsibleContent>
    </Collapsible>
  )
}
