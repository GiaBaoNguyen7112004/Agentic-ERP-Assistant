import { useEffect, useState } from 'react'
import { Activity, ExternalLink, History } from 'lucide-react'
import { getRun } from '../api'
import { hydrateTurn, type HydratedTurn } from '../hydrate'
import { TurnTrace } from './TurnTrace'
import { EmptyState } from '@/components/shared/EmptyState'
import type { Message } from '../appState'

/**
 * The trace panel: the newest assistant turn by default, or the one the user
 * clicked (`turn_selected`), hydrating a filed run when the selected turn
 * never streamed -- a session loaded from history, or a queue-resumed
 * decision whose fresh bubble started mid-turn (Phase 4,
 * docs/trace-inspector-plan.md). A failed read falls back to the empty
 * state with the raw-run link, the same view Phase 1 showed.
 */
export function TracePanel({
  messages,
  inspectedId,
  onInspect,
}: {
  messages: Message[]
  inspectedId: string | null
  onInspect: (id: string | null) => void
}) {
  const latestAssistant = [...messages].reverse().find((message) => message.role === 'assistant') as
    | Extract<Message, { role: 'assistant' }>
    | undefined
  const selected =
    (inspectedId
      ? (messages.find(
          (message) => message.role === 'assistant' && message.id === inspectedId,
        ) as Extract<Message, { role: 'assistant' }> | undefined)
      : undefined) ?? latestAssistant

  const [hydrated, setHydrated] = useState<Record<string, HydratedTurn | null>>({})
  const traceId = selected?.traceId ?? null
  const turn = selected?.turn
  // Only a settled turn with no events at all is a filed run to read back;
  // a live turn still streaming starts at zero events too.
  const needsHydration =
    !!traceId &&
    !!turn &&
    turn.events.length === 0 &&
    (turn.status === 'done' || turn.status === 'paused')
  // `undefined` = not read yet, `null` = the read failed, a value = hydrated.
  const hydratedView = traceId && traceId in hydrated ? hydrated[traceId] : undefined

  useEffect(() => {
    if (!needsHydration || !traceId || traceId in hydrated) return
    let cancelled = false
    getRun(traceId)
      .then((report) => {
        if (!cancelled) setHydrated((prev) => ({ ...prev, [traceId]: hydrateTurn(report) }))
      })
      .catch((error: unknown) => {
        console.error('failed to load the filed run', error)
        if (!cancelled) setHydrated((prev) => ({ ...prev, [traceId]: null }))
      })
    return () => {
      cancelled = true
    }
  }, [needsHydration, traceId, hydrated])

  if (!selected) {
    return (
      <section className="space-y-5">
        <EmptyState
          icon={Activity}
          title="Ask something to see its trace here."
          hint="The trace panel shows the newest assistant turn."
        />
      </section>
    )
  }

  if (needsHydration) {
    if (hydratedView === null) {
      return (
        <section className="space-y-5">
          <EmptyState
            icon={Activity}
            title="No live trace for a turn loaded from history"
            hint={
              traceId && (
                <a
                  href={`/api/runs/${traceId}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 underline outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                >
                  open the run
                  <ExternalLink className="size-3" aria-hidden />
                </a>
              )
            }
          />
        </section>
      )
    }
    if (hydratedView === undefined) {
      return (
        <section className="space-y-5">
          <EmptyState
            icon={Activity}
            title="Loading the filed run…"
            hint="This turn was not streamed here; its trace is being read back."
          />
        </section>
      )
    }
    return (
      <section className="space-y-5">
        <TurnTrace message={selected} turn={hydratedView.turn} reconstructed unattributed={hydratedView.unattributed} />
      </section>
    )
  }

  return (
    <section className="space-y-5">
      {inspectedId && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <History className="size-3" aria-hidden />
          inspecting this bubble —{' '}
          <button
            type="button"
            onClick={() => onInspect(null)}
            className="underline outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            back to the newest turn
          </button>
        </p>
      )}
      <TurnTrace message={selected} />
    </section>
  )
}