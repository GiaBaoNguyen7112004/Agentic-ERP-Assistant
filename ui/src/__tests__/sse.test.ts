import { describe, expect, it } from 'vitest'
import { readSse } from '../sse'
import type { ServerEvent } from '../protocol'

function responseFromChunks(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
  return new Response(stream)
}

async function collect(response: Response, signal: AbortSignal): Promise<ServerEvent[]> {
  const events: ServerEvent[] = []
  for await (const event of readSse(response, signal)) {
    events.push(event)
  }
  return events
}

const FRAME_A = 'event: turn_started\ndata: {"type":"turn_started","trace_id":"run-1","session_id":"s1","actor":"priya","resumed":false}\n\n'
const FRAME_B = 'event: token\ndata: {"type":"token","text":"Hi"}\n\n'

describe('readSse', () => {
  it('parses a whole frame delivered in one chunk', async () => {
    const response = responseFromChunks([FRAME_A])
    const events = await collect(response, new AbortController().signal)
    expect(events).toEqual([
      { type: 'turn_started', trace_id: 'run-1', session_id: 's1', actor: 'priya', resumed: false },
    ])
  })

  it('parses a frame split across two chunk boundaries, at every split point', async () => {
    for (let splitAt = 1; splitAt < FRAME_A.length; splitAt++) {
      const response = responseFromChunks([FRAME_A.slice(0, splitAt), FRAME_A.slice(splitAt)])
      const events = await collect(response, new AbortController().signal)
      expect(events, `split at ${splitAt}`).toEqual([
        { type: 'turn_started', trace_id: 'run-1', session_id: 's1', actor: 'priya', resumed: false },
      ])
    }
  })

  it('yields multiple frames in order, however they are chunked', async () => {
    const response = responseFromChunks([FRAME_A.slice(0, 10), FRAME_A.slice(10) + FRAME_B.slice(0, 5), FRAME_B.slice(5)])
    const events = await collect(response, new AbortController().signal)
    expect(events.map((e) => e.type)).toEqual(['turn_started', 'token'])
  })

  it('ignores SSE comment lines', async () => {
    const response = responseFromChunks([': keep-alive\n\n', FRAME_B])
    const events = await collect(response, new AbortController().signal)
    expect(events).toEqual([{ type: 'token', text: 'Hi' }])
  })

  it('reads a final frame with no trailing blank line', async () => {
    const response = responseFromChunks([FRAME_B.trimEnd()])
    const events = await collect(response, new AbortController().signal)
    expect(events).toEqual([{ type: 'token', text: 'Hi' }])
  })

  it('aborting mid-stream ends the generator without throwing', async () => {
    const controller = new AbortController()
    const encoder = new TextEncoder()
    let cancelled = false

    const stream = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(encoder.encode(FRAME_A))
        // Deliberately never closes -- the only way this test ends is abort.
      },
      cancel() {
        cancelled = true
      },
    })
    const response = new Response(stream)

    const events: ServerEvent[] = []
    const iterator = readSse(response, controller.signal)

    const first = await iterator.next()
    expect(first.done).toBe(false)
    if (!first.done) events.push(first.value)

    controller.abort()
    const second = await iterator.next()
    expect(second.done).toBe(true)
    expect(events.map((e) => e.type)).toEqual(['turn_started'])
    expect(cancelled).toBe(true)
  })
})
