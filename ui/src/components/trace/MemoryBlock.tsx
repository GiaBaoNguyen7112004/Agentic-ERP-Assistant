import { ToneBadge } from '@/components/shared/ToneBadge'
import { formatRelativeTime } from '@/lib/format'
import type { MemoryOut } from '../../protocol'

const KIND_TONE: Record<string, 'memory' | 'neutral'> = {
  preference: 'memory',
  decision: 'memory',
  fact: 'memory',
  intent: 'memory',
  session_summary: 'memory',
}

/** One durable memory recall selected for this turn -- background the model
 * reads in its own role, never a citation (see `state/memory.py`: it has no
 * locator, and the grounding check rejects anything that isn't one). */
function MemoryRow({ memory }: { memory: MemoryOut }) {
  return (
    <li className="space-y-1 rounded-md border bg-muted/30 p-2 text-xs">
      <div className="flex flex-wrap items-center gap-1.5 text-muted-foreground">
        <ToneBadge tone={KIND_TONE[memory.kind] ?? 'neutral'}>{memory.kind}</ToneBadge>
        <span className="font-mono">{memory.key}</span>
        <span className="ml-auto tabular-nums">{formatRelativeTime(memory.recorded_at)}</span>
        <span className="tabular-nums">conf {memory.confidence.toFixed(2)}</span>
      </div>
      <p className="whitespace-pre-wrap break-words">{memory.statement}</p>
    </li>
  )
}

/** What recall selected for this turn, in the order it will be rendered
 * to the model -- see `memory/service.py::SessionMemory.recall`. */
export function MemoryBlock({ memories }: { memories: MemoryOut[] }) {
  if (memories.length === 0) {
    return <p className="text-xs text-muted-foreground">nothing remembered that bears on this</p>
  }

  return <ul className="space-y-1.5">{memories.map((memory) => <MemoryRow key={memory.memory_id} memory={memory} />)}</ul>
}
