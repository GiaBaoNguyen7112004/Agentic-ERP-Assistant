import { MessagesSquare, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { EmptyState } from '@/components/shared/EmptyState'
import { SectionHeading } from '@/components/shared/SectionHeading'
import { formatRelativeTime } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { SessionSummary } from '../protocol'

export function SessionList({
  sessions,
  activeSessionId,
  onSelect,
  onNewChat,
}: {
  sessions: SessionSummary[]
  activeSessionId: string | null
  onSelect: (sessionId: string) => void
  onNewChat: () => void
}) {
  return (
    <section className="space-y-2">
      <SectionHeading count={sessions.length} action={
        <Button type="button" variant="outline" size="sm" onClick={onNewChat}>
          <Plus aria-hidden />
          New chat
        </Button>
      }>
        Sessions
      </SectionHeading>
      <ul className="space-y-0.5">
        {sessions.map((session) => {
          const active = session.session_id === activeSessionId
          return (
            <li key={session.session_id}>
              <button
                type="button"
                onClick={() => onSelect(session.session_id)}
                aria-current={active ? 'true' : undefined}
                title={session.first_request}
                className={cn(
                  'w-full rounded-md px-2 py-1.5 text-left text-sm outline-none transition-colors',
                  'hover:bg-accent hover:text-accent-foreground focus-visible:ring-2 focus-visible:ring-ring/50',
                  active ? 'bg-accent text-accent-foreground' : 'text-foreground',
                )}
              >
                <span className="block truncate">{session.first_request}</span>
                <span className="block text-xs text-muted-foreground">
                  {formatRelativeTime(session.last_started_at)}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      {sessions.length === 0 && (
        <EmptyState icon={MessagesSquare} title="No sessions yet" hint="Start a chat to see it here." />
      )}
    </section>
  )
}