import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { RetrievalBlock } from '../components/trace/RetrievalBlock'
import type { RetrievalOut } from '../protocol'

function retrieval(overrides: Partial<RetrievalOut> = {}): RetrievalOut {
  return {
    query: 'why is M2 late',
    limit: 4,
    hits: [],
    best_similarity: 0.8,
    minimum_similarity: 0.3,
    dense_candidates: 1,
    lexical_candidates: 1,
    gated: false,
    ...overrides,
  }
}

describe('RetrievalBlock', () => {
  it('renders the query and candidate counts', () => {
    render(<RetrievalBlock retrieval={retrieval()} />)
    expect(screen.getByText('why is M2 late')).toBeInTheDocument()
    expect(screen.getByText(/1 dense \/ 1 lexical candidate/)).toBeInTheDocument()
  })

  it('renders each hit with its fused score and per-list ranks', () => {
    render(
      <RetrievalBlock
        retrieval={retrieval({
          hits: [
            {
              chunk_id: 'risk-register#row R-2',
              document_id: 'risk-register',
              locator: 'row R-2',
              title: 'Risk Register',
              score: 1.5,
              ranks: { vector: 1, lexical: 2 },
              scores: { vector: 0.91, lexical: 4.2 },
            },
          ],
        })}
      />,
    )
    expect(screen.getByText('risk-register#row R-2')).toBeInTheDocument()
    expect(screen.getByText('1.500')).toBeInTheDocument()
    expect(screen.getByText('#1 (0.910)')).toBeInTheDocument()
    expect(screen.getByText('#2 (4.200)')).toBeInTheDocument()
  })

  it('renders a warning line instead of a table when the search was gated', () => {
    render(
      <RetrievalBlock
        retrieval={retrieval({ gated: true, hits: [], best_similarity: 0.29, dense_candidates: 0 })}
      />,
    )
    expect(screen.getByText(/under the 0.30 floor/)).toBeInTheDocument()
  })

  it('renders "nothing" when a gated search saw no dense candidates at all', () => {
    render(
      <RetrievalBlock retrieval={retrieval({ gated: true, hits: [], best_similarity: null, dense_candidates: 0 })} />,
    )
    expect(screen.getByText(/closest chunk scored nothing/)).toBeInTheDocument()
  })
})
