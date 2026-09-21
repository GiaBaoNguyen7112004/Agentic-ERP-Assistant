import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { TextBlock } from '@/components/shared/TextBlock'
import type { EvidenceOut } from '../../protocol'

/** One retrieved passage, expandable to the text it was cited from --
 * `tag` is the exact token the model was told to cite, `[doc#locator]`. */
function EvidenceItem({ snippet }: { snippet: EvidenceOut }) {
  const [open, setOpen] = useState(false)

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-md border bg-muted/30">
      <CollapsibleTrigger className="flex w-full items-center gap-1.5 px-2 py-1 text-left text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50">
        {open ? <ChevronDown className="size-3 shrink-0" aria-hidden /> : <ChevronRight className="size-3 shrink-0" aria-hidden />}
        <span className="font-mono">{snippet.tag}</span>
      </CollapsibleTrigger>
      <CollapsibleContent className="px-2 pb-2">
        <TextBlock text={snippet.text.text} />
        {snippet.text.truncated && (
          <p className="mt-1 text-[11px] text-muted-foreground">
            showing {snippet.text.text.length.toLocaleString()} of {snippet.text.chars.toLocaleString()} characters
          </p>
        )}
      </CollapsibleContent>
    </Collapsible>
  )
}

/** The passages retrieval supplied this turn -- assigned exactly once, so
 * a node card shows this only on the step that set it. */
export function EvidenceBlock({ evidence }: { evidence: EvidenceOut[] }) {
  if (evidence.length === 0) {
    return <p className="text-xs text-muted-foreground">no passages retrieved</p>
  }

  return <div className="space-y-1">{evidence.map((snippet) => <EvidenceItem key={snippet.tag} snippet={snippet} />)}</div>
}
