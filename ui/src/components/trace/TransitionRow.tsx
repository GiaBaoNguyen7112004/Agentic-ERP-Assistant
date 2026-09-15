import { ArrowRight } from 'lucide-react'
import { ToneBadge } from '@/components/shared/ToneBadge'
import { routeTone } from '@/lib/routeBadge'
import { transitionReason } from '@/lib/transitionReason'
import type { ExecutionEntry } from '@/lib/executionTree'

/** Why the graph moved on from this node -- the route it landed on, and the
 * row that best explains the choice (see `transitionReason`'s priority). */
export function TransitionRow({ entry }: { entry: ExecutionEntry }) {
  const route = entry.step?.route ?? null
  if (route === null && entry.step === null) return null // still streaming; nothing to report yet

  const reason = transitionReason(entry.rows)

  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
      <ArrowRight className="size-3 shrink-0" aria-hidden />
      <ToneBadge tone={routeTone(route)}>{route ?? '(none)'}</ToneBadge>
      {reason && <span className="min-w-0 break-words">{reason}</span>}
    </div>
  )
}
