import type { Message } from '../appState'
import { DecisionList } from './DecisionList'
import { EventRow } from './EventRow'
import { TurnSummary } from './TurnSummary'

export function TurnTrace({ message }: { message: Extract<Message, { role: 'assistant' }> }) {
  const { turn, traceId } = message

  return (
    <section>
      <h3>
        {traceId ? (
          <a href={`/api/runs/${traceId}`} target="_blank" rel="noreferrer">
            {traceId}
          </a>
        ) : (
          '(starting…)'
        )}
      </h3>
      <DecisionList decisions={turn.decisions} />
      <div>
        {turn.events.map((event, index) => (
          <EventRow key={`${event.seq ?? 'gw'}-${index}`} event={event} />
        ))}
      </div>
      {turn.summary && <TurnSummary summary={turn.summary} />}
    </section>
  )
}
