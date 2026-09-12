import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

/** Icon + one line + optional hint, for every empty panel state. */
export function EmptyState({
  icon: Icon,
  title,
  hint,
  className,
}: {
  icon: LucideIcon
  title: string
  hint?: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center gap-1.5 rounded-lg px-4 py-6 text-center text-muted-foreground',
        className,
      )}
    >
      <Icon className="size-5 shrink-0 opacity-60" aria-hidden />
      <p className="text-sm">{title}</p>
      {hint && <p className="text-xs text-muted-foreground/80">{hint}</p>}
    </div>
  )
}