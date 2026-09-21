import { Copy, ExternalLink } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

/**
 * A monospace id, optionally a link, with an optional copy button -- the one
 * rendering for trace ids, session ids and any other opaque identifier.
 */
export function MonoId({
  value,
  href,
  copy = false,
  truncate = false,
  className,
}: {
  value: string
  href?: string
  copy?: boolean
  truncate?: boolean
  className?: string
}) {
  const text = (
    <span
      className={cn(
        'font-mono text-xs text-muted-foreground',
        truncate && 'block truncate',
        !truncate && 'break-all',
        className,
      )}
    >
      {value}
    </span>
  )

  return (
    <span className="inline-flex min-w-0 items-center gap-1">
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
        >
          {text}
          <ExternalLink className="size-3 shrink-0 text-muted-foreground" aria-hidden />
        </a>
      ) : (
        text
      )}
      {copy && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={`Copy ${value}`}
              onClick={() => {
                void navigator.clipboard?.writeText(value)
              }}
            >
              <Copy aria-hidden />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Copy id</TooltipContent>
        </Tooltip>
      )}
    </span>
  )
}