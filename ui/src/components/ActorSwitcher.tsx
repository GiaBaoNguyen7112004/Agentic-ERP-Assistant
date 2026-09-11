import type { UserSummary } from '../protocol'

export function ActorSwitcher({
  users,
  actor,
  onSelect,
}: {
  users: UserSummary[]
  actor: string | null
  onSelect: (actor: string) => void
}) {
  const current = users.find((user) => user.actor === actor)

  return (
    <section>
      <h2>Acting as</h2>
      <select
        aria-label="Actor"
        value={actor ?? ''}
        onChange={(event) => onSelect(event.target.value)}
      >
        {users.map((user) => (
          <option key={user.actor} value={user.actor}>
            {user.display_name} — {user.role}
          </option>
        ))}
      </select>
      {current && (
        <div style={{ marginTop: 8 }}>
          {current.can_approve && <span className="scope-chip">can approve</span>}
          {current.scopes.map((scope) => (
            <span key={scope} className="scope-chip">
              {scope}
            </span>
          ))}
        </div>
      )}
    </section>
  )
}
