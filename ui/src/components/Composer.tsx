import { useState, type KeyboardEvent } from 'react'

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
    <div className="composer">
      <textarea
        aria-label="Message"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Ask about a document, a milestone, a budget, a risk…"
      />
      {running ? (
        <button type="button" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button type="button" onClick={send} disabled={!value.trim()}>
          Send
        </button>
      )}
    </div>
  )
}
