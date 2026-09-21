import { ExternalLink, Database } from 'lucide-react'
import { ToneBadge } from '@/components/shared/ToneBadge'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { Citation } from '../protocol'
import { documentUrl } from '../api'

export function CitationChips({ citations, actor }: { citations: Citation[]; actor: string }) {
  if (citations.length === 0) return null

  return (
    <div className="flex flex-wrap gap-1.5">
      {citations.map((citation) => {
        if (citation.kind === 'document') {
          return (
            <Tooltip key={citation.tag}>
              <TooltipTrigger asChild>
                <ToneBadge
                  tone="info"
                  asChild
                  className="cursor-pointer font-mono hover:bg-info/20"
                >
                  <a
                    href={documentUrl(citation.source_id, actor)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <ExternalLink aria-hidden />
                    {citation.tag}
                  </a>
                </ToneBadge>
              </TooltipTrigger>
              <TooltipContent className="font-mono">
                {citation.source_id} · {citation.locator ?? '—'}
              </TooltipContent>
            </Tooltip>
          )
        }
        return (
          <Tooltip key={citation.tag}>
            <TooltipTrigger asChild>
              <ToneBadge tone="neutral" className="font-mono">
                <Database aria-hidden />
                {citation.tag}
              </ToneBadge>
            </TooltipTrigger>
            <TooltipContent className="font-mono">
              {citation.source_id} · {citation.locator ?? '—'}
            </TooltipContent>
          </Tooltip>
        )
      })}
    </div>
  )
}