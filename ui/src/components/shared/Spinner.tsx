import { LoaderCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

/** The one spinner, for loading gates and "deciding" states. */
export function Spinner({ className }: { className?: string }) {
  return (
    <LoaderCircle
      aria-hidden
      className={cn('size-4 shrink-0 animate-spin text-muted-foreground', className)}
    />
  )
}