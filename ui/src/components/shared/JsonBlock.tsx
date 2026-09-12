import { cn } from '@/lib/utils'

/** Pretty-printed JSON in a mono block -- the decision list's arguments. */
export function JsonBlock({ value, className }: { value: unknown; className?: string }) {
  return (
    <pre
      className={cn(
        'overflow-x-auto rounded-md bg-muted px-2 py-1.5 font-mono text-xs whitespace-pre-wrap break-words',
        className,
      )}
    >
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}