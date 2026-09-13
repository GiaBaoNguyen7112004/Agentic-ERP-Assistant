import { ToneBadge } from '@/components/shared/ToneBadge'
import type { Tone } from '@/lib/routeBadge'
import type { MemoryAuditOut } from '../../protocol'

const DECISION_TONE: Record<string, Tone> = {
  write: 'success',
  update: 'success',
  reject: 'warning',
  forget: 'neutral',
}

/** One decision consolidation made about a piece of would-be memory,
 * rejections included -- the audit's whole point (see `memory/audit.py`):
 * a run that refused four proposals and kept one is a run where the policy
 * worked, and it must not read the same as one that stored five. */
function MemoryAuditRow({ row }: { row: MemoryAuditOut }) {
  return (
    <li className="space-y-0.5 rounded-md border bg-muted/30 p-2 text-xs">
      <div className="flex flex-wrap items-center gap-1.5">
        <ToneBadge tone={DECISION_TONE[row.decision] ?? 'neutral'}>{row.decision}</ToneBadge>
        <span className="text-muted-foreground">{row.kind}</span>
        {row.rejection && <span className="font-mono text-warning">{row.rejection}</span>}
      </div>
      <p className="whitespace-pre-wrap break-words">{row.statement_summary}</p>
      {row.reason && <p className="text-muted-foreground">{row.reason}</p>}
    </li>
  )
}

/** Every memory decision consolidation made while filing this run --
 * `TurnFinishedEvent.memory_audit`, the same rows `memory_audit` holds. */
export function MemoryAuditBlock({ rows }: { rows: MemoryAuditOut[] }) {
  if (rows.length === 0) return null
  return <ul className="space-y-1.5">{rows.map((row, index) => <MemoryAuditRow key={index} row={row} />)}</ul>
}
