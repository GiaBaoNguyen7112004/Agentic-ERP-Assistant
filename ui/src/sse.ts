import type { ServerEvent } from './protocol'

/**
 * Read a `text/event-stream` response body as typed events, in order.
 *
 * Frames are separated by a blank line; a `data:` line carries the JSON
 * payload (`event:` lines are read but not relied on -- the payload's own
 * `type` field is authoritative, the same way the union in protocol.ts is
 * discriminated). Tolerates a frame split across two chunk boundaries: bytes
 * are buffered until a full `\n\n` separator is seen, never parsed early.
 */
export async function* readSse(
  response: Response,
  signal: AbortSignal,
): AsyncGenerator<ServerEvent> {
  if (!response.body) return

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const onAbort = () => {
    void reader.cancel().catch(() => undefined)
  }
  signal.addEventListener('abort', onAbort)

  try {
    while (true) {
      if (signal.aborted) return

      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let separatorIndex = buffer.indexOf('\n\n')
      while (separatorIndex !== -1) {
        const frame = buffer.slice(0, separatorIndex)
        buffer = buffer.slice(separatorIndex + 2)
        const event = parseFrame(frame)
        if (event) yield event
        if (signal.aborted) return
        separatorIndex = buffer.indexOf('\n\n')
      }
    }

    // A final frame with no trailing blank line -- the stream ended right
    // after the server's last write.
    const event = parseFrame(buffer)
    if (event) yield event
  } finally {
    signal.removeEventListener('abort', onAbort)
  }
}

function parseFrame(frame: string): ServerEvent | null {
  const dataLines: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith(':')) continue // an SSE comment line
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).replace(/^ /, ''))
    }
  }
  if (dataLines.length === 0) return null

  try {
    return JSON.parse(dataLines.join('\n')) as ServerEvent
  } catch {
    return null
  }
}

/** POST JSON and return the raw response, for readSse to stream from. */
export function postSse(url: string, body: unknown, signal: AbortSignal): Promise<Response> {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}
