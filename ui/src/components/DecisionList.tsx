import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { JsonBlock } from '@/components/shared/JsonBlock'
import { ToneBadge } from '@/components/shared/ToneBadge'
import { routeTone } from '@/lib/routeBadge'
import type { StepEvent } from '../protocol'

function DecisionItem({ decision }: { decision: StepEvent }): ReactNode {
  // Arguments start open: the point of the decision list is to show what the
  // planner was about to do before the tool runs (handbook T1).
  const [open, setOpen] = useState(true)

  return (
    <li className="space-y-1">
      <div className="flex flex-wrap items-center gap-1.5">
        <ToneBadge tone={routeTone(decision.route)}>
          {decision.route ?? '(none)'}
        </ToneBadge>
        {decision.tool_name && (
          <span className="font-mono text-xs text-foreground">→ {decision.tool_name}</span>
        )}
      </div>
      {decision.tool_arguments && (
        <Collapsible open={open} onOpenChange={setOpen}>
          <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50">
            {open ? (
              <ChevronDown className="size-3" aria-hidden />
            ) : (
              <ChevronRight className="size-3" aria-hidden />
            )}
            arguments
          </CollapsibleTrigger>
          <CollapsibleContent>
            <JsonBlock value={decision.tool_arguments} />
          </CollapsibleContent>
        </Collapsible>
      )}
    </li>
  )
}

export function DecisionList({ decisions }: { decisions: StepEvent[] }) {
  if (decisions.length === 0) return null

  return (
    <ol className="space-y-2.5">
      {decisions.map((decision, index) => (
        <DecisionItem key={index} decision={decision} />
      ))}
    </ol>
  )
}