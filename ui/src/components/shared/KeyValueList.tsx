import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/**
 * A definition list of key/value rows -- the shared rendering for approval
 * arguments and the turn summary.
 */
export function KeyValueList({
  items,
  className,
}: {
  items: { key: string; value: ReactNode }[]
  className?: string
}) {
  if (items.length === 0) return null

  return (
    <dl
      className={cn('grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-sm', className)}
    >
      {items.map((item) => (
        <div key={item.key} className="contents">
          <dt className="font-medium text-muted-foreground">{item.key}</dt>
          <dd className="min-w-0 break-words">{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}