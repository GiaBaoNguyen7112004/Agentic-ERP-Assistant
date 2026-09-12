import type { Message } from '../appState'
import { TurnTrace } from './TurnTrace'

export function TracePanel({ messages }: { messages: Message[] }) {
  const latestAssistant = [...messages].reverse().find((message) => message.role === 'assistant') as
    | Extract<Message, { role: 'assistant' }>
    | undefined

  return (
    <section className="space-y-5">
      <h2 className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        Trace
      </h2>
      {latestAssistant ? (
        <TurnTrace message={latestAssistant} />
      ) : (
        <p className="text-sm text-muted-foreground">Ask something to see its trace here.</p>
      )}
    </section>
  )
}
