import { useEffect, useState } from 'react'
import { getUsers, listApprovals, listSessions, listTurns } from './api'
import {
  loadRememberedActor,
  rememberActor,
  useAppState,
  type Message,
} from './appState'
import { ActorSwitcher } from './components/ActorSwitcher'
import { ApprovalQueue } from './components/ApprovalQueue'
import { ChatPanel } from './components/ChatPanel'
import { SessionList } from './components/SessionList'
import { TracePanel } from './components/TracePanel'
import { AppShell } from './components/layout/AppShell'
import { initialTurn } from './turnReducer'
import { useTurnStream } from './useTurnStream'

function newId(): string {
  return crypto.randomUUID()
}

function traceIdOfMessage(messages: Message[], id: string | null): string | null {
  if (!id) return null
  const message = messages.find((candidate) => candidate.id === id)
  return message && message.role === 'assistant' ? message.traceId : null
}

export default function App() {
  const [state, dispatch] = useAppState()
  const [decidingId, setDecidingId] = useState<string | null>(null)
  const { running, start, stop } = useTurnStream()

  // Load the actor directory once, then pick a remembered or first actor.
  useEffect(() => {
    getUsers()
      .then((users) => {
        dispatch({ type: 'users_loaded', users })
        const remembered = loadRememberedActor()
        const actor =
          remembered && users.some((user) => user.actor === remembered)
            ? remembered
            : users[0]?.actor
        if (actor) dispatch({ type: 'actor_selected', actor })
      })
      .catch((error: unknown) => {
        console.error('failed to load users', error)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const refreshApprovals = () => {
    listApprovals()
      .then((approvals) => dispatch({ type: 'approvals_loaded', approvals }))
      .catch((error: unknown) => console.error('failed to load approvals', error))
  }

  useEffect(() => {
    if (!state.actor) return
    listSessions(state.actor)
      .then((sessions) => dispatch({ type: 'sessions_loaded', sessions }))
      .catch((error: unknown) => console.error('failed to load sessions', error))
    refreshApprovals()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.actor])

  const selectActor = (actor: string) => {
    rememberActor(actor)
    dispatch({ type: 'actor_selected', actor })
  }

  const selectSession = (sessionId: string) => {
    if (!state.actor) return
    dispatch({ type: 'session_selected', sessionId })
    listTurns(sessionId, state.actor)
      .then((turns) => {
        const messages: Message[] = turns.flatMap((turn) => [
          { role: 'user', id: `${turn.trace_id}-req`, text: turn.request },
          {
            role: 'assistant',
            id: `${turn.trace_id}-res`,
            traceId: turn.trace_id,
            turn: {
              ...initialTurn,
              status: turn.approval === 'pending' ? 'paused' : 'done',
              answer:
                turn.approval === 'pending'
                  ? null
                  : {
                      type: 'answer',
                      text: turn.response ?? '',
                      route: turn.route,
                      failure: turn.failure,
                      error_detail: null,
                      citations: [],
                    },
            },
          },
        ])
        dispatch({ type: 'history_loaded', messages })
      })
      .catch((error: unknown) => console.error('failed to load turns', error))
  }

  const sendMessage = async (message: string) => {
    if (!state.actor) return
    const userId = newId()
    const assistantId = newId()
    dispatch({ type: 'user_message_sent', id: userId, text: message })
    dispatch({ type: 'assistant_message_started', id: assistantId })

    let capturedSessionId: string | null = null
    await start(
      '/api/chat',
      { actor: state.actor, session_id: state.sessionId, message },
      (event) => {
        dispatch({ type: 'turn_event', id: assistantId, event })
        if (event.type === 'turn_started') {
          capturedSessionId = event.session_id
        }
      },
    )
    if (capturedSessionId && !state.sessionId) {
      dispatch({ type: 'session_id_captured', sessionId: capturedSessionId })
    }
    refreshApprovals()
  }

  const decideFromMessage = async (id: string, approved: boolean) => {
    if (!state.actor) return
    const message = state.messages.find((candidate) => candidate.id === id)
    if (!message || message.role !== 'assistant' || !message.traceId) return

    setDecidingId(id)
    try {
      await start(
        `/api/approvals/${message.traceId}`,
        { actor: state.actor, approved },
        (event) => dispatch({ type: 'turn_event', id, event }),
      )
    } finally {
      setDecidingId(null)
      refreshApprovals()
    }
  }

  const decideFromQueue = async (traceId: string, approved: boolean) => {
    if (!state.actor) return
    const assistantId = newId()
    dispatch({ type: 'assistant_message_started', id: assistantId })
    setDecidingId(assistantId)
    try {
      await start(
        `/api/approvals/${traceId}`,
        { actor: state.actor, approved },
        (event) => dispatch({ type: 'turn_event', id: assistantId, event }),
      )
    } finally {
      setDecidingId(null)
      refreshApprovals()
    }
  }

  const currentUser = state.users.find((user) => user.actor === state.actor)
  const canApprove = currentUser?.can_approve ?? false

  if (!state.actor) {
    return <AppShell loading />
  }

  return (
    <AppShell
      sidebar={
        <>
          <ActorSwitcher users={state.users} actor={state.actor} onSelect={selectActor} />
          <SessionList
            sessions={state.sessions}
            activeSessionId={state.sessionId}
            onSelect={selectSession}
            onNewChat={() => dispatch({ type: 'new_chat' })}
          />
          <ApprovalQueue
            approvals={state.approvals}
            canApprove={canApprove}
            decidingTraceId={traceIdOfMessage(state.messages, decidingId)}
            onDecide={decideFromQueue}
          />
        </>
      }
      main={
        <ChatPanel
          messages={state.messages}
          actor={state.actor}
          sessionId={state.sessionId}
          canApprove={canApprove}
          running={running}
          decidingId={decidingId}
          onSend={(message) => void sendMessage(message)}
          onStop={stop}
          onDecide={(id, approved) => void decideFromMessage(id, approved)}
          onInspect={(id) => dispatch({ type: 'turn_selected', id })}
        />
      }
      aside={
        <TracePanel
          messages={state.messages}
          inspectedId={state.inspectedId}
          onInspect={(id) => dispatch({ type: 'turn_selected', id })}
        />
      }
    />
  )
}
