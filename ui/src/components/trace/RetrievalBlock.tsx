import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { RetrievalOut } from '../../protocol'

/** One search's diagnostics -- what the dense and lexical halves actually
 * saw, and why a chunk did or didn't make the cut. Never available from the
 * port itself (`DocumentRetrieverPort.search` returns no scores,
 * deliberately); this is `InspectedRetriever`'s own live-only report. */
export function RetrievalBlock({ retrieval }: { retrieval: RetrievalOut }) {
  return (
    <div className="space-y-1.5 text-xs">
      <p className="text-muted-foreground">
        query <span className="font-mono text-foreground">{retrieval.query}</span> · limit {retrieval.limit} ·{' '}
        {retrieval.dense_candidates} dense / {retrieval.lexical_candidates} lexical candidate(s)
      </p>
      {retrieval.gated ? (
        <p className="text-warning">
          closest chunk scored{' '}
          {retrieval.best_similarity !== null ? retrieval.best_similarity.toFixed(3) : 'nothing'}, under the{' '}
          {retrieval.minimum_similarity.toFixed(2)} floor -- refused before synthesis
        </p>
      ) : (
        retrieval.hits.length > 0 && (
          <div className="overflow-x-auto rounded-md border">
            <Table className="text-xs [&_th]:py-1 [&_td]:py-1">
              <TableHeader>
                <TableRow>
                  <TableHead className="pl-2">chunk</TableHead>
                  <TableHead>fused</TableHead>
                  <TableHead>vector</TableHead>
                  <TableHead>lexical</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {retrieval.hits.map((hit) => (
                  <TableRow key={hit.chunk_id}>
                    <TableCell className="pl-2 font-mono">{hit.chunk_id}</TableCell>
                    <TableCell className="tabular-nums">{hit.score.toFixed(3)}</TableCell>
                    <TableCell className="tabular-nums text-muted-foreground">
                      {hit.ranks.vector !== undefined
                        ? `#${hit.ranks.vector} (${hit.scores.vector?.toFixed(3)})`
                        : '—'}
                    </TableCell>
                    <TableCell className="tabular-nums text-muted-foreground">
                      {hit.ranks.lexical !== undefined
                        ? `#${hit.ranks.lexical} (${hit.scores.lexical?.toFixed(3)})`
                        : '—'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )
      )}
    </div>
  )
}
