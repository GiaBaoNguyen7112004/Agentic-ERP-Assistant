import type { TraceRow } from '../protocol'

function kindClass(kind: string): string {
  if (kind.includes('route_selected')) return 'kind-decision'
  if (kind.includes('tool_called')) return 'kind-tool'
  if (kind.includes('approval')) return 'kind-approval'
  if (kind.includes('memory')) return 'kind-memory'
  if (kind.includes('failed') || kind === 'run_failed') return 'kind-failure'
  return ''
}

export function EventRow({ event }: { event: TraceRow }) {
  const gateway = event.source === 'tool_gateway'

  return (
    <div className={`trace-row${gateway ? ' gateway' : ''}`}>
      <span>{event.seq ?? '·'}</span>
      <span>{event.node}</span>
      <span className={`kind ${kindClass(event.kind)}`}>{event.kind}</span>
      <span>{event.detail}</span>
    </div>
  )
}
