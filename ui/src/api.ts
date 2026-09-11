import type { PendingApproval, SessionSummary, SessionTurn, UserSummary } from './protocol'

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`${url} -> ${response.status}`)
  }
  return (await response.json()) as T
}

export function getUsers(): Promise<UserSummary[]> {
  return getJson('/api/users')
}

export async function createSession(actor: string): Promise<string> {
  const response = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ actor }),
  })
  if (!response.ok) {
    throw new Error(`create session -> ${response.status}`)
  }
  const body = (await response.json()) as { session_id: string }
  return body.session_id
}

export function listSessions(actor: string): Promise<SessionSummary[]> {
  return getJson(`/api/sessions?actor=${encodeURIComponent(actor)}`)
}

export function listTurns(sessionId: string, actor: string): Promise<SessionTurn[]> {
  return getJson(
    `/api/sessions/${encodeURIComponent(sessionId)}/turns?actor=${encodeURIComponent(actor)}`,
  )
}

export function listApprovals(actor?: string): Promise<PendingApproval[]> {
  const query = actor ? `?actor=${encodeURIComponent(actor)}` : ''
  return getJson(`/api/approvals${query}`)
}

export function getRun(traceId: string): Promise<unknown> {
  return getJson(`/api/runs/${encodeURIComponent(traceId)}`)
}

/** Where a citation chip opens a document -- never fetched by this module
 * itself, since the browser navigating there is what applies the actor's
 * own cookie/session state in a deployment that has one; here it just
 * carries the actor as a query param, matching GET /api/documents/{id}. */
export function documentUrl(documentId: string, actor: string): string {
  return `/api/documents/${encodeURIComponent(documentId)}?actor=${encodeURIComponent(actor)}`
}
