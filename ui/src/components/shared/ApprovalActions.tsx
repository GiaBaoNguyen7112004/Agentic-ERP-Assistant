import type { ReactNode } from 'react'
import { Check, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

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
  const disabled = !canApprove || deciding
  const hint = deciding
    ? 'A decision is already being recorded.'
    : 'This actor cannot decide approvals -- switch to an approver.'

  return (
    <div className={className ?? 'flex gap-2'}>
      <WithDisabledHint disabled={disabled} hint={hint}>
        <Button
          type="button"
          size={size}
          disabled={disabled}
          onClick={() => onDecide(true)}
        >
          <Check aria-hidden />
          Approve
        </Button>
      </WithDisabledHint>
      <WithDisabledHint disabled={disabled} hint={hint}>
        <Button
          type="button"
          variant="outline"
          size={size}
          disabled={disabled}
          onClick={() => onDecide(false)}
          className="text-destructive"
        >
          <X aria-hidden />
          Deny
        </Button>
      </WithDisabledHint>
    </div>
  )
}

/** Disabled buttons eat pointer events, so the tooltip wraps them in a span. */
function WithDisabledHint({
  disabled,
  hint,
  children,
}: {
  disabled: boolean
  hint: string
  children: ReactNode
}) {
  if (!disabled) return <>{children}</>
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex">{children}</span>
      </TooltipTrigger>
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  )
}