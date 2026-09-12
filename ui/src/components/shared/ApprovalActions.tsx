import { Check, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

/**
 * The Approve/Deny pair, shared by the approval card and the queue so both
 * render (and behave) identically. Button text stays exactly `Approve` and
 * `Deny` -- the handbook and the component tests pin it.
 */
export function ApprovalActions({
  canApprove,
  deciding,
  onDecide,
  size = 'sm',
  className,
}: {
  canApprove: boolean
  deciding: boolean
  onDecide: (approved: boolean) => void
  size?: 'sm' | 'default'
  className?: string
}) {
  return (
    <div className={className ?? 'flex gap-2'}>
      <Button
        type="button"
        size={size}
        disabled={!canApprove || deciding}
        onClick={() => onDecide(true)}
      >
        <Check aria-hidden />
        Approve
      </Button>
      <Button
        type="button"
        variant="outline"
        size={size}
        disabled={!canApprove || deciding}
        onClick={() => onDecide(false)}
        className="text-destructive"
      >
        <X aria-hidden />
        Deny
      </Button>
    </div>
  )
}