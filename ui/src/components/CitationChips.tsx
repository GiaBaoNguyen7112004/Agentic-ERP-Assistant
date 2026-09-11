import type { Citation } from '../protocol'
import { documentUrl } from '../api'

export function CitationChips({ citations, actor }: { citations: Citation[]; actor: string }) {
  if (citations.length === 0) return null

  return (
    <div>
      {citations.map((citation) => {
        if (citation.kind === 'document') {
          return (
            <a
              key={citation.tag}
              className="citation-chip document"
              href={documentUrl(citation.source_id, actor)}
              target="_blank"
              rel="noreferrer"
            >
              {citation.tag}
            </a>
          )
        }
        return (
          <span key={citation.tag} className="citation-chip">
            {citation.tag}
          </span>
        )
      })}
    </div>
  )
}
