import { useEffect, useRef } from 'react'
import { Bot } from 'lucide-react'
import { AssistantMessage } from './AssistantMessage'
import { EmptyState } from '@/components/shared/EmptyState'
import { cn } from '@/lib/utils'
import type { Message } from '../appState'

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
  const endRef = useRef<HTMLDivElement>(null)

  // Keep the newest message in view; the guard keeps jsdom and exotic
  // browsers (no scrollIntoView) from throwing.
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: 'end' })
  }, [messages])

  return (
    <ol
      className="flex min-h-0 flex-1 list-none flex-col gap-4 overflow-y-auto px-4 py-4"
      aria-live="polite"
    >
      {messages.map((message) =>
        message.role === 'user' ? (
          <li key={message.id} className="flex max-w-[80%] flex-col items-end self-end">
            <div className="rounded-2xl rounded-br-md bg-primary px-3.5 py-2 text-sm text-primary-foreground whitespace-pre-wrap">
              {message.text}
            </div>
          </li>
        ) : (
          <li key={message.id} className="min-w-0">
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
      {messages.length === 0 && (
        <EmptyState
          icon={Bot}
          title="Ask about a document, a milestone, a budget, a risk…"
          hint="Answers carry citations; write actions wait for an approval."
        />
      )}
      {/* the bottom sentinel auto-scroll keeps an eye on */}
      <div ref={endRef} className={cn(messages.length === 0 && 'hidden')} />
    </ol>
  )
}