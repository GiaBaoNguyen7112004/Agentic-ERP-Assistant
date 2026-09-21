import { ClipboardCheck } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { ApprovalActions } from '@/components/shared/ApprovalActions'
import { EmptyState } from '@/components/shared/EmptyState'
import { SectionHeading } from '@/components/shared/SectionHeading'
import { formatRelativeTime } from '@/lib/format'
import type { PendingApproval } from '../protocol'

export function ApprovalQueue({
  approvals,
  canApprove,
  decidingTraceId,
  onDecide,
}: {
  approvals: PendingApproval[]
  canApprove: boolean
  decidingTraceId: string | null
  onDecide: (traceId: string, approved: boolean) => void
}) {
  return (
    <section className="space-y-2">
      <SectionHeading count={approvals.length}>Pending approvals</SectionHeading>
      <ul className="space-y-2">
        {approvals.map((approval) => {
          const deciding = decidingTraceId === approval.trace_id
          return (
            <li key={approval.trace_id}>
              <Card className="gap-2 border-warning/50 py-2.5">
                <CardHeader className="gap-1 px-2.5">
                  <CardTitle className="font-mono text-sm font-semibold">
                    {approval.tool_name}
                  </CardTitle>
                  <p className="text-xs text-muted-foreground">
                    requested by {approval.actor} · {formatRelativeTime(approval.created_at)}
                  </p>
                </CardHeader>
                <CardContent className="space-y-2 px-2.5">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <p className="truncate font-mono text-xs text-muted-foreground">
                        {approval.arguments_summary}
                      </p>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-72 break-words font-mono">
                      {approval.arguments_summary}
                    </TooltipContent>
                  </Tooltip>
                  <ApprovalActions
                    canApprove={canApprove}
                    deciding={deciding}
                    onDecide={(approved) => onDecide(approval.trace_id, approved)}
                  />
                  {!canApprove && (
                    <p className="text-xs text-muted-foreground">switch to an approver</p>
                  )}
                </CardContent>
              </Card>
            </li>
          )
        })}
      </ul>
      {approvals.length === 0 && (
        <EmptyState
          icon={ClipboardCheck}
          title="Nothing pending"
          hint="Write actions that need approval appear here."
        />
      )}
    </section>
  )
}