import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { cn } from '@/lib/utils'

const COLLAPSE_LINE_COUNT = 12

/**
 * Bounded rendering for a block of free text -- an evidence passage, a tool
 * summary, a prompt block. Collapses beyond `COLLAPSE_LINE_COUNT` lines with
 * a "show more" toggle, rather than truncating: the server-side cap (a wire
 * `truncated`/`chars` pair) is a later phase's concern, this is purely a
 * client-side render bound so one long payload cannot push a whole node card
 * off-screen.
 */
export function TextBlock({ text, className }: { text: string; className?: string }) {
  const [open, setOpen] = useState(false)
  const lines = text.split('\n')
  const long = lines.length > COLLAPSE_LINE_COUNT

  if (!long) {
    return (
      <pre
        className={cn(
          'overflow-x-auto rounded-md bg-muted px-2 py-1.5 font-mono text-xs whitespace-pre-wrap break-words',
          className,
        )}
      >
        {text}
      </pre>
    )
  }

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <pre
        className={cn(
          'overflow-x-auto rounded-md bg-muted px-2 py-1.5 font-mono text-xs whitespace-pre-wrap break-words',
          !open && 'max-h-48 overflow-y-hidden',
          className,
        )}
      >
        {text}
      </pre>
      <CollapsibleTrigger asChild>
        <Button variant="ghost" size="sm" className="mt-1 h-6 gap-1 px-1.5 text-xs text-muted-foreground">
          {open ? <ChevronDown className="size-3" aria-hidden /> : <ChevronRight className="size-3" aria-hidden />}
          {open ? 'show less' : `show all ${lines.length} lines`}
        </Button>
      </CollapsibleTrigger>
      {/* CollapsibleContent is a no-op here (the <pre> above already toggles
          via line-clamp) -- kept for Radix's own aria-expanded wiring. */}
      <CollapsibleContent />
    </Collapsible>
  )
}
