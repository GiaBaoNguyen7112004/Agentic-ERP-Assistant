import { ToneBadge } from '@/components/shared/ToneBadge'
import type { ContractOut } from '../../protocol'

/** What the planner declared this turn's reply must rest on (ADR 0021), or
 * that the turn was never checked at all -- `null` here is not "nothing
 * needed", it is "no declaration ran" (a replay, a hand-built state, a
 * declarer that raised outright); `needs: []` is the real "nothing needed"
 * answer. */
export function ContractBlock({ contract }: { contract: ContractOut | null }) {
  if (contract === null) {
    return <p className="text-xs text-muted-foreground">unchecked -- no reply contract was declared</p>
  }
  if (contract.needs.length === 0) {
    return <p className="text-xs text-muted-foreground">declared: nothing needed (a write, a refusal, or a clarification)</p>
  }

  return (
    <div className="space-y-1 text-xs">
      <div className="flex flex-wrap gap-1.5">
        {contract.needs.map((need) => (
          <ToneBadge key={need} tone="info">
            {need}
          </ToneBadge>
        ))}
      </div>
      {contract.document_query && (
        <p className="text-muted-foreground">
          document query: <span className="font-mono text-foreground">{contract.document_query}</span>
        </p>
      )}
    </div>
  )
}
