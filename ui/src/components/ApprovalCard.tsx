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
    <section className="approval-card" aria-label="Approval needed">
      <strong>Approval needed: {approval.tool_name}</strong>
      <p>{approval.summary}</p>
      <dl>
        {Object.entries(approval.arguments).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{String(value)}</dd>
          </div>
        ))}
      </dl>
      <div className="approval-actions">
        <button
          type="button"
          className="approve"
          disabled={!canApprove || deciding}
          onClick={() => onDecide(true)}
        >
          Approve
        </button>
        <button
          type="button"
          className="deny"
          disabled={!canApprove || deciding}
          onClick={() => onDecide(false)}
        >
          Deny
        </button>
      </div>
      {!canApprove && <p>switch to an approver to decide this</p>}
    </section>
  )
}
