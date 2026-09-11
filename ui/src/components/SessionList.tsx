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
    <section>
      <h2>Sessions</h2>
      <button type="button" onClick={onNewChat}>
        New chat
      </button>
      <ul className="session-list">
        {sessions.map((session) => (
          <li
            key={session.session_id}
            onClick={() => onSelect(session.session_id)}
            style={{
              fontWeight: session.session_id === activeSessionId ? 700 : 400,
            }}
          >
            {session.first_request}
          </li>
        ))}
        {sessions.length === 0 && <li>No sessions yet</li>}
      </ul>
    </section>
  )
}
