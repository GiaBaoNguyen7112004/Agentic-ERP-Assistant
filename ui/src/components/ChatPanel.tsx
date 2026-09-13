import type { Message } from '../appState'
import { Composer } from './Composer'
import { MessageList } from './MessageList'
import { MonoId } from '@/components/shared/MonoId'

export function ChatPanel({
  messages,
  actor,
  sessionId,
  canApprove,
  running,
  decidingId,
  onSend,
  onStop,
  onDecide,
  onInspect,
}: {
  messages: Message[]
  actor: string
  /** The live session's id, once the server names one -- the header strip. */
  sessionId: string | null
  canApprove: boolean
  running: boolean
  decidingId: string | null
  onSend: (message: string) => void
  onStop: () => void
  onDecide: (id: string, approved: boolean) => void
  onInspect?: (id: string) => void
}) {
  return (
    <main className="flex h-full min-h-0 min-w-0 flex-col">
      <header className="flex items-center gap-3 border-b bg-sidebar px-4 py-2 text-xs text-muted-foreground">
        <span className="font-medium text-foreground">{actor}</span>
        {sessionId ? <MonoId value={sessionId} truncate className="min-w-0 flex-1" /> : (
          <span className="font-mono">New chat</span>
        )}
        {running && (
          <span className="flex items-center gap-1.5">
            <span className="size-2 animate-pulse rounded-full bg-success" aria-hidden />
            running
          </span>
        )}
      </header>
      <MessageList
        messages={messages}
        actor={actor}
        canApprove={canApprove}
        decidingId={decidingId}
        onDecide={onDecide}
        onInspect={onInspect}
      />
      <Composer running={running} onSend={onSend} onStop={onStop} />
    </main>
  )
}