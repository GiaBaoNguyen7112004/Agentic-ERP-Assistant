import { SectionHeading } from '@/components/shared/SectionHeading'
import type { ContextEvent } from '../../protocol'
import { ContractBlock } from './ContractBlock'
import { HistoryBlock } from './HistoryBlock'
import { MemoryBlock } from './MemoryBlock'
import { ModelCallBlock } from './ModelCallBlock'

/** What the orchestrator attached to the state before the engine ever ran:
 * the session's recent turns, recalled memory, and the declared reply
 * contract -- see `web/protocol.py::ContextEvent`'s own doc for why this
 * exists beside the flat `history_recalled`/`memory_recalled`/
 * `contract_declared` rows the raw table already shows. */
export function ContextBlock({ context }: { context: ContextEvent }) {
  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <SectionHeading count={context.history.length}>History</SectionHeading>
        <HistoryBlock turns={context.history} />
      </div>
      <div className="space-y-1.5">
        <SectionHeading count={context.memories.length}>Memory</SectionHeading>
        <MemoryBlock memories={context.memories} />
      </div>
      <div className="space-y-1.5">
        <SectionHeading>Contract</SectionHeading>
        <ContractBlock contract={context.contract} />
      </div>
      {context.model_calls.length > 0 && (
        <div className="space-y-1.5">
          <SectionHeading count={context.model_calls.length}>Declaration call</SectionHeading>
          <div className="space-y-1.5">
            {context.model_calls.map((call, index) => (
              <ModelCallBlock key={index} call={call} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
