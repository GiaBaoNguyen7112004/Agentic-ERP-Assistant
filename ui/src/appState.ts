import { useReducer } from 'react'
import type { PendingApproval, ServerEvent, SessionSummary, UserSummary } from './protocol'
import { initialTurn, turnReducer, type TurnView } from './turnReducer'

export type Message =
  | { role: 'user'; id: string; text: string }
  | { role: 'assistant'; id: string; traceId: string | null; turn: TurnView }

export interface AppState {
  actor: string | null
  users: UserSummary[]
  sessionId: string | null
  sessions: SessionSummary[]
  messages: Message[]
  approvals: PendingApproval[]
}

export const initialAppState: AppState = {
  actor: null,
  users: [],
  sessionId: null,
  sessions: [],
  messages: [],
  approvals: [],
}

export type AppAction =
  | { type: 'users_loaded'; users: UserSummary[] }
  | { type: 'actor_selected'; actor: string }
  | { type: 'session_selected'; sessionId: string }
  | { type: 'session_id_captured'; sessionId: string }
  | { type: 'new_chat' }
  | { type: 'sessions_loaded'; sessions: SessionSummary[] }
  | { type: 'approvals_loaded'; approvals: PendingApproval[] }
  | { type: 'history_loaded'; messages: Message[] }
  | { type: 'user_message_sent'; id: string; text: string }
  | { type: 'assistant_message_started'; id: string }
  | { type: 'turn_event'; id: string; event: ServerEvent }

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'users_loaded':
      return { ...state, users: action.users }

    case 'actor_selected':
      // Switching actor starts a new session (G5: ActorSwitcher).
      return {
        ...state,
        actor: action.actor,
        sessionId: null,
        messages: [],
        approvals: [],
      }

    case 'session_selected':
      // A user picked a different (or the same) session from the list --
      // its history replaces whatever is showing.
      return { ...state, sessionId: action.sessionId, messages: [] }

    case 'session_id_captured':
      // The server generated a session id for the chat already on screen
      // (a fresh "New chat" had none yet). Record it without touching
      // messages -- unlike session_selected, this is not a navigation.
      return { ...state, sessionId: action.sessionId }

    case 'new_chat':
      return { ...state, sessionId: null, messages: [] }

    case 'sessions_loaded':
      return { ...state, sessions: action.sessions }

    case 'approvals_loaded':
      return { ...state, approvals: action.approvals }

    case 'history_loaded':
      return { ...state, messages: action.messages }

    case 'user_message_sent':
      return {
        ...state,
        messages: [...state.messages, { role: 'user', id: action.id, text: action.text }],
      }

    case 'assistant_message_started':
      return {
        ...state,
        messages: [
          ...state.messages,
          { role: 'assistant', id: action.id, traceId: null, turn: initialTurn },
        ],
      }

    case 'turn_event':
      return {
        ...state,
        messages: state.messages.map((message) =>
          message.role === 'assistant' && message.id === action.id
            ? {
                ...message,
                traceId: message.traceId ?? (action.event as { trace_id?: string }).trace_id ?? null,
                turn: turnReducer(message.turn, action.event),
              }
            : message,
        ),
      }

    default:
      return state
  }
}

export function useAppState() {
  return useReducer(appReducer, initialAppState)
}

// -- localStorage: per-viewer convenience only, never a source of truth ----

const ACTOR_KEY = 'agentic-erp-assistant:actor'
const SESSION_KEY_PREFIX = 'agentic-erp-assistant:session:'

export function loadRememberedActor(): string | null {
  try {
    return window.localStorage.getItem(ACTOR_KEY)
  } catch {
    return null
  }
}

export function rememberActor(actor: string): void {
  try {
    window.localStorage.setItem(ACTOR_KEY, actor)
  } catch {
    // a private window or blocked storage -- the switcher still works
  }
}

export function loadRememberedSession(actor: string): string | null {
  try {
    return window.localStorage.getItem(SESSION_KEY_PREFIX + actor)
  } catch {
    return null
  }
}

export function rememberSession(actor: string, sessionId: string): void {
  try {
    window.localStorage.setItem(SESSION_KEY_PREFIX + actor, sessionId)
  } catch {
    // ignore
  }
}
