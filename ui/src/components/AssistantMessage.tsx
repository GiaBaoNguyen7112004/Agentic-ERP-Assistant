import { Bot, Database } from 'lucide-react'
import { RouteBadge } from '@/components/shared/RouteBadge'
import { StreamingCaret } from '@/components/shared/StreamingCaret'
import { ApprovalCard } from './ApprovalCard'
import { CitationChips } from './CitationChips'
import { FailureBlock } from './FailureBlock'
import { cn } from '@/lib/utils'
import type { TurnView } from '../turnReducer'

export function AssistantMessage({
  turn,
  actor,
  canApprove,
  deciding,
  onDecide,
  onInspect,
}: {
  turn: TurnView
  actor: string
  canApprove: boolean
  deciding: boolean
  onDecide: (approved: boolean) => void
  /** Clicking the bubble inspects this turn in the trace panel -- a plain
   * convenience, not an interactive control: buttons and links inside the
   * bubble keep their own behavior, and the panel still shows the newest
   * turn by default without it. */
  onInspect?: () => void
}) {
  // D4: the authoritative answer replaces the streamed preview the instant
  // it arrives; before that, the preview is all there is to show.
  const text = turn.answer ? turn.answer.text : turn.streamed
  const observations = turn.summary?.observations ?? []

  return (
    <div
      className={cn('flex gap-2.5', onInspect && 'cursor-pointer')}
      onClick={
        onInspect
          ? (event) => {
              // The bubble carries real controls (approval actions, citation
              // chips); only a click on the bubble itself selects it.
              if ((event.target as HTMLElement).closest('button, a')) return
              onInspect()
            }
          : undefined
      }
      title={onInspect ? 'Inspect this turn in the trace panel' : undefined}
    >
      <div className="mt-1 flex size-7 shrink-0 items-center justify-center rounded-lg border bg-card">
        <Bot className="size-4 text-muted-foreground" aria-hidden />
      </div>
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <RouteBadge turn={turn} />
          {observations.map((observation) => (
            <span
              key={observation.tool}
              className="inline-flex items-center gap-1 rounded-md border bg-muted/50 px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground"
            >
              <Database className="size-3" aria-hidden />
              {observation.tool}
            </span>
          ))}
        </div>
        <p className={cn('whitespace-pre-wrap text-sm leading-relaxed')}>
          {text || <StreamingCaret />}
          {text && !turn.answer && <StreamingCaret />}
        </p>
        {turn.answer && <CitationChips citations={turn.answer.citations} actor={actor} />}
        {turn.answer && (
          <FailureBlock failure={turn.answer.failure} errorDetail={turn.answer.error_detail} />
        )}
        {turn.approval && (
          <ApprovalCard
            approval={turn.approval}
            canApprove={canApprove}
            deciding={deciding}
            onDecide={onDecide}
          />
        )}
        {turn.status === 'error' && turn.error && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive" role="alert">
            {turn.error}
          </div>
        )}
      </div>
    </div>
  )
}