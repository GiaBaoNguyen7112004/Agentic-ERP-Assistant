import type { Message } from '../appState'
import { TurnTrace } from './TurnTrace'

export function TracePanel({ messages }: { messages: Message[] }) {
  const latestAssistant = [...messages].reverse().find((message) => message.role === 'assistant') as
    | Extract<Message, { role: 'assistant' }>
    | undefined

  return (
    <aside className="trace-panel">
      <h2>Trace</h2>
      {latestAssistant ? (
        <TurnTrace message={latestAssistant} />
      ) : (
        <p>Ask something to see its trace here.</p>
      )}
    </aside>
  )
}
