import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { JsonBlock } from '@/components/shared/JsonBlock'
import { ToneBadge } from '@/components/shared/ToneBadge'
import { TextBlock } from '@/components/shared/TextBlock'
import { formatCost } from '@/lib/format'
import type { Tone } from '@/lib/routeBadge'
import type { ModelCallOut } from '../../protocol'

const OUTCOME_TONE: Record<string, Tone> = {
  answered: 'success',
  routed: 'success',
  invalid_schema: 'destructive',
  budget_exceeded: 'destructive',
  provider_failure: 'destructive',
}

/** One model call's cost line, and -- only when `DEV_TRACE_MODEL_IO` was on
 * for this turn -- its actual prompt and reply, collapsed behind their own
 * triggers so a seven-block prompt does not dominate the card by default. */
export function ModelCallBlock({ call }: { call: ModelCallOut }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="space-y-1 rounded-md border bg-muted/30 p-2 text-xs">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-mono">{call.model}</span>
        <ToneBadge tone={OUTCOME_TONE[call.outcome] ?? 'neutral'}>{call.outcome}</ToneBadge>
        <span className="tabular-nums text-muted-foreground">
          {call.input_tokens}/{call.output_tokens} tok
        </span>
        <span className="font-mono text-muted-foreground">{formatCost(call.cost_usd)}</span>
        <span className="tabular-nums text-muted-foreground">{(call.latency_seconds * 1000).toFixed(0)}ms</span>
        {call.attempts > 1 && <span className="text-muted-foreground">{call.attempts} attempts</span>}
      </div>
      {call.detail && <p className="text-muted-foreground">{call.detail}</p>}
      {call.request || call.response ? (
        <Collapsible open={open} onOpenChange={setOpen}>
          <CollapsibleTrigger className="flex items-center gap-1 text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50">
            {open ? <ChevronDown className="size-3" aria-hidden /> : <ChevronRight className="size-3" aria-hidden />}
            request &amp; reply
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-1 space-y-2">
            {call.request && (
              <div className="space-y-1">
                <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                  Request
                  {call.request.tools.length > 0 && (
                    <span className="ml-1.5 font-mono normal-case tracking-normal">
                      tools: {call.request.tools.join(', ')}
                      {call.request.tool_choice ? ` (${call.request.tool_choice})` : ''}
                    </span>
                  )}
                </p>
                <div className="space-y-1">
                  {call.request.messages.map((message, index) => (
                    <div key={index} className="space-y-0.5">
                      <span className="font-mono text-[11px] font-semibold text-muted-foreground">{message.role}</span>
                      <TextBlock text={message.content.text} />
                    </div>
                  ))}
                </div>
              </div>
            )}
            {call.response && (
              <div className="space-y-1">
                <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Reply</p>
                {call.response.tool_name ? (
                  <>
                    <p className="font-mono">→ {call.response.tool_name}</p>
                    {call.response.arguments && <JsonBlock value={call.response.arguments} />}
                  </>
                ) : (
                  call.response.content && <TextBlock text={call.response.content.text} />
                )}
              </div>
            )}
          </CollapsibleContent>
        </Collapsible>
      ) : (
        <p className="text-muted-foreground/70">
          set <span className="font-mono">DEV_TRACE_MODEL_IO=1</span> to capture the prompt and reply
        </p>
      )}
    </div>
  )
}
