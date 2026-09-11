import type { Message } from '../appState'
import { Composer } from './Composer'
import { MessageList } from './MessageList'

export function ChatPanel({
  messages,
  actor,
  canApprove,
  running,
  decidingId,
  onSend,
  onStop,
  onDecide,
}: {
  messages: Message[]
  actor: string
  canApprove: boolean
  running: boolean
  decidingId: string | null
  onSend: (message: string) => void
  onStop: () => void
  onDecide: (id: string, approved: boolean) => void
}) {
  return (
    <main className="chat-panel">
      <MessageList
        messages={messages}
        actor={actor}
        canApprove={canApprove}
        decidingId={decidingId}
        onDecide={onDecide}
      />
      <Composer running={running} onSend={onSend} onStop={onStop} />
    </main>
  )
}
