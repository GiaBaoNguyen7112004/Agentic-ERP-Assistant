import { Hourglass } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ApprovalActions } from '@/components/shared/ApprovalActions'
import { KeyValueList } from '@/components/shared/KeyValueList'
import { formatArgument } from '@/lib/format'
import type { ApprovalRequiredEvent } from '../protocol'

export function ApprovalCard({
  approval,
  canApprove,
  deciding,
  onDecide,
}: {
  approval: ApprovalRequiredEvent
  canApprove: boolean
  deciding: boolean
  onDecide: (approved: boolean) => void
}) {
  return (
    <Card
      className="border-warning/60 py-3"
      aria-label="Approval needed"
      data-testid="approval-card"
    >
      <CardHeader className="gap-1 px-3">
        <CardTitle className="flex items-center gap-1.5 text-sm">
          <Hourglass className="size-4 text-warning" aria-hidden />
          Approval needed: <span className="font-mono">{approval.tool_name}</span>
        </CardTitle>
        <p className="text-xs text-muted-foreground">{approval.summary}</p>
      </CardHeader>
      <CardContent className="space-y-2.5 px-3">
        <KeyValueList
          className="text-xs"
          items={Object.entries(approval.arguments).map(([key, value]) => ({
            key,
            value: <span className="font-mono">{formatArgument(value)}</span>,
          }))}
        />
        <ApprovalActions canApprove={canApprove} deciding={deciding} onDecide={onDecide} />
        {!canApprove && <p className="text-xs text-muted-foreground">switch to an approver to decide this</p>}
      </CardContent>
    </Card>
  )
}