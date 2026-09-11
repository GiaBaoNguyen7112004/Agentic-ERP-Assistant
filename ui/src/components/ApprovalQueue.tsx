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
    <section>
      <h2>Pending approvals</h2>
      <ul className="approval-queue">
        {approvals.map((approval) => (
          <li key={approval.trace_id}>
            <div>
              <strong>{approval.tool_name}</strong> — {approval.actor}
            </div>
            <div>{approval.arguments_summary}</div>
            <div>
              <button
                type="button"
                disabled={!canApprove || decidingTraceId === approval.trace_id}
                onClick={() => onDecide(approval.trace_id, true)}
              >
                Approve
              </button>
              <button
                type="button"
                disabled={!canApprove || decidingTraceId === approval.trace_id}
                onClick={() => onDecide(approval.trace_id, false)}
              >
                Deny
              </button>
            </div>
            {!canApprove && <small>switch to an approver</small>}
          </li>
        ))}
        {approvals.length === 0 && <li>Nothing pending</li>}
      </ul>
    </section>
  )
}
