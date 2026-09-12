/** The blinking caret shown after streamed text while the turn is running. */
export function StreamingCaret() {
  return (
    <span
      aria-hidden
      className="ml-0.5 inline-block h-3.5 w-[2px] translate-y-0.5 animate-pulse bg-foreground"
    />
  )
}