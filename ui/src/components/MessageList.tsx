import type { Message } from '../appState'
import { AssistantMessage } from './AssistantMessage'

export function MessageList({
  messages,
  actor,
  canApprove,
  decidingId,
  onDecide,
}: {
  messages: Message[]
  actor: string
  canApprove: boolean
  decidingId: string | null
  onDecide: (id: string, approved: boolean) => void
}) {
  return (
    <ol className="message-list" aria-live="polite">
      {messages.map((message) =>
        message.role === 'user' ? (
          <li key={message.id} className="message user">
            {message.text}
          </li>
        ) : (
          <li key={message.id}>
            <AssistantMessage
              turn={message.turn}
              actor={actor}
              canApprove={canApprove}
              deciding={decidingId === message.id}
              onDecide={(approved) => onDecide(message.id, approved)}
            />
          </li>
        ),
      )}
    </ol>
  )
}
