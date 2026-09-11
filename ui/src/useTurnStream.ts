import { useCallback, useRef, useState } from 'react'
import { postSse, readSse } from './sse'
import type { ServerEvent } from './protocol'

/**
 * The fetch + SSE lifecycle, decoupled from where the resulting view lives.
 *
 * `start` is handed an `onEvent` callback rather than owning a `TurnView`
 * itself -- the same hook drives a fresh chat turn (dispatching into a new
 * message's view) and a resumed approval decision (dispatching into the
 * *same* message's view the pause card came from), and the caller is the
 * one that knows which.
 */
export function useTurnStream() {
  const [running, setRunning] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)

  const start = useCallback(
    async (url: string, body: unknown, onEvent: (event: ServerEvent) => void) => {
      controllerRef.current?.abort()
      const controller = new AbortController()
      controllerRef.current = controller
      setRunning(true)

      try {
        const response = await postSse(url, body, controller.signal)
        if (!response.ok) {
          let detail = response.statusText
          try {
            const problem = (await response.json()) as { detail?: string }
            if (problem.detail) detail = problem.detail
          } catch {
            // no JSON body -- the status text stands
          }
          onEvent({ type: 'error', message: detail })
          return
        }
        for await (const event of readSse(response, controller.signal)) {
          onEvent(event)
        }
      } catch (error) {
        if (controller.signal.aborted) return // D11: stopping is not a failure
        onEvent({ type: 'error', message: String(error) })
      } finally {
        setRunning(false)
      }
    },
    [],
  )

  /** D11: the server keeps running the turn to completion and files the
   * trace regardless -- this only stops the client from listening. */
  const stop = useCallback(() => {
    controllerRef.current?.abort()
  }, [])

  return { running, start, stop }
}
