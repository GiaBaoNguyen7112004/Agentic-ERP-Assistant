import { Activity } from 'lucide-react'
import { TurnTrace } from './TurnTrace'
import { EmptyState } from '@/components/shared/EmptyState'
import type { Message } from '../appState'

export function TracePanel({ messages }: { messages: Message[] }) {
  const latestAssistant = [...messages].reverse().find((message) => message.role === 'assistant') as
    | Extract<Message, { role: 'assistant' }>
    | undefined

  return (
    <section className="space-y-5">
      {latestAssistant ? (
        <TurnTrace message={latestAssistant} />
      ) : (
        <EmptyState
          icon={Activity}
          title="Ask something to see its trace here."
          hint="The trace panel shows the newest assistant turn."
        />
      )}
    </section>
  )
}