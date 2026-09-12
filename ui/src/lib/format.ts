/** Formatting helpers shared across the redesign -- all pure. */

/** `5s ago`-style relative time, deterministic in `now` so tests can pin it. */
export function formatRelativeTime(iso: string, now: number = Date.now()): string {
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const seconds = Math.max(0, Math.round((now - then) / 1000))
  if (seconds < 60) return seconds === 1 ? '1s ago' : `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return minutes === 1 ? '1m ago' : `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return hours === 1 ? '1h ago' : `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 30) return days === 1 ? '1d ago' : `${days}d ago`
  return new Date(iso).toLocaleDateString()
}

/** Cost, exactly the old `unknown` wording for the null case. */
export function formatCost(costUsd: number | null): string {
  return costUsd === null ? 'unknown' : `$${costUsd.toFixed(4)}`
}

/**
 * A tool argument for display: objects and arrays as JSON (the old card
 * rendered `[object Object]` for these), anything scalar as its String.
 */
export function formatArgument(value: unknown): string {
  if (value !== null && typeof value === 'object') return JSON.stringify(value)
  return String(value)
}