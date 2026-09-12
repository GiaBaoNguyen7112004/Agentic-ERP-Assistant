import { useState, type KeyboardEvent } from 'react'
import { SendHorizontal, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/shared/Kbd'

export function Composer({
  running,
  onSend,
  onStop,
}: {
  running: boolean
  onSend: (message: string) => void
  onStop: () => void
}) {
  const [value, setValue] = useState('')

  const send = () => {
    const message = value.trim()
    if (!message) return
    onSend(message)
    setValue('')
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      send()
    }
  }

  return (
    <div className="border-t px-4 pb-3 pt-3">
      <div className="rounded-lg border bg-card px-3 py-2 transition-shadow focus-within:ring-2 focus-within:ring-ring/40">
        <textarea
          aria-label="Message"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about a document, a milestone, a budget, a risk…"
          className="block min-h-[2.5rem] w-full resize-y border-0 bg-transparent p-0 text-sm outline-none placeholder:text-muted-foreground"
        />
      </div>
      <div className="mt-2 flex items-center justify-between gap-2">
        <span className="text-xs text-muted-foreground">
          <Kbd>Enter ↵</Kbd> send · <Kbd>Shift+Enter</Kbd> newline
        </span>
        {running ? (
          <Button type="button" variant="outline" onClick={onStop}>
            <Square aria-hidden />
            Stop
          </Button>
        ) : (
          <Button type="button" onClick={send} disabled={!value.trim()}>
            Send
            <SendHorizontal aria-hidden />
          </Button>
        )}
      </div>
    </div>
  )
}