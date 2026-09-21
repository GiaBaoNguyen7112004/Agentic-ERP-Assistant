import { MonoId } from '@/components/shared/MonoId'
import { ToneBadge } from '@/components/shared/ToneBadge'
import { routeTone } from '@/lib/routeBadge'
import type { HistoryTurnOut } from '../../protocol'

/** One prior turn of the session, as this turn was shown it -- already
 * clipped and citation-stripped by the time it reaches the wire. */
function HistoryTurnRow({ turn, index }: { turn: HistoryTurnOut; index: number }) {
  const reply = turn.response ?? (turn.approval === 'pending' ? `(waiting for approval to run ${turn.tool_name})` : `(${turn.failure})`)

  return (
    <li className="space-y-1 rounded-md border bg-muted/30 p-2 text-xs">
      <div className="flex flex-wrap items-center gap-1.5 text-muted-foreground">
        <span className="font-mono">#{index + 1}</span>
        {turn.route && <ToneBadge tone={routeTone(turn.route)}>{turn.route}</ToneBadge>}
        <MonoId value={turn.trace_id} truncate className="max-w-32" />
      </div>
      <p className="whitespace-pre-wrap break-words">
        <span className="font-medium text-foreground">User: </span>
        {turn.request}
      </p>
      <p className="whitespace-pre-wrap break-words text-muted-foreground">
        <span className="font-medium">Assistant: </span>
        {reply}
      </p>
    </li>
  )
}

/** The session's recent turns this turn was shown, in the `history` role --
 * verbatim, never judged by the memory policy (see `state/conversation.py`). */
export function HistoryBlock({ turns }: { turns: HistoryTurnOut[] }) {
  if (turns.length === 0) return <p className="text-xs text-muted-foreground">no prior turns in this session</p>

  return <ol className="space-y-1.5">{turns.map((turn, index) => <HistoryTurnRow key={turn.trace_id} turn={turn} index={index} />)}</ol>
}
