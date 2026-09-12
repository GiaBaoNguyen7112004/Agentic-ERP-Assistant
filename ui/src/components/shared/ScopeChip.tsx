import { cn } from '@/lib/utils'

/** A mono outline badge for one authorization scope. */
export function ScopeChip({ scope, className }: { scope: string; className?: string }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-md border border-border bg-background px-1.5 py-0.5 font-mono text-xs text-muted-foreground',
        className,
      )}
    >
      {scope}
    </span>
  )
}