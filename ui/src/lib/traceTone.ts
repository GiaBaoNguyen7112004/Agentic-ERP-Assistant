import type { Tone } from './routeBadge'

/**
 * The tone of a trace row's kind, using the same substring rules the old
 * `kindClass()` used -- only the output changed, from a CSS class to a token.
 */
export function traceTone(kind: string): Tone {
  if (kind.includes('route_selected')) return 'info'
  if (kind.includes('tool_called')) return 'success'
  if (kind.includes('approval')) return 'warning'
  if (kind.includes('memory')) return 'memory'
  if (kind.includes('failed') || kind === 'run_failed') return 'destructive'
  return 'neutral'
}