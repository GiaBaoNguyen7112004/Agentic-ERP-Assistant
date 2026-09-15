import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/**
 * The old global `h2` rule, made into a component: an 11px uppercase label,
 * optionally with a count and an action (e.g. a "New chat" button).
 */
export function SectionHeading({
  children,
  count,
  action,
  className,
}: {
  children: ReactNode
  count?: number
  action?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex items-center justify-between gap-2', className)}>
      <h2 className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {children}
        {typeof count === 'number' && (
          <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-muted-foreground">
            {count}
          </span>
        )}
      </h2>
      {action}
    </div>
  )
}