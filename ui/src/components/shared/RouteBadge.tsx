import { FileText, Wrench, MessageCircleQuestion, ShieldBan, CircleAlert, Hourglass, LoaderCircle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { ToneBadge } from './ToneBadge'
import { routeBadge as routeBadgeView } from '@/lib/routeBadge'
import type { Tone } from '@/lib/routeBadge'
import type { TurnView } from '@/turnReducer'

const toneIcons: Record<Tone, LucideIcon | null> = {
  info: FileText,
  neutral: null,
  success: null,
  warning: ShieldBan,
  destructive: CircleAlert,
  memory: null,
}

const statusIcons: Record<string, LucideIcon | null> = {
  starting: LoaderCircle,
  running: LoaderCircle,
  'waiting for approval': Hourglass,
  refused: ShieldBan,
  failed: CircleAlert,
  failed_by_failure: CircleAlert,
  clarify: MessageCircleQuestion,
  tool: Wrench,
  documents: FileText,
}

/**
 * The route badge for a turn: label byte-identical to the pre-redesign
 * strings, tone and icon from the §1 semantic map.
 */
export function RouteBadge({ turn, className }: { turn: TurnView; className?: string }) {
  const view = routeBadgeView(turn)
  const Icon = statusIcons[view.label] ?? toneIcons[view.tone]
  const spinning = view.label === 'starting' || view.label === 'running'

  return (
    <ToneBadge tone={view.tone} className={className} data-testid="route-badge">
      {Icon && <Icon className={spinning ? 'animate-spin' : undefined} aria-hidden />}
      {view.label}
    </ToneBadge>
  )
}