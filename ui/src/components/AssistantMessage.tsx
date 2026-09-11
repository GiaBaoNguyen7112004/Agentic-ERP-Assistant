import type { TurnView } from '../turnReducer'
import { ApprovalCard } from './ApprovalCard'
import { CitationChips } from './CitationChips'
import { FailureBlock } from './FailureBlock'

function routeBadge(turn: TurnView): string {
  if (turn.status === 'paused') return 'waiting for approval'
  if (!turn.answer) return turn.status
  const { route, failure } = turn.answer
  if (route === 'refuse') return 'refused'
  if (route === 'fail' || failure !== 'none') return 'failed'
  if (route === 'clarify') return 'clarify'
  if (route === 'answer') {
    const usedTool = (turn.summary?.observations.length ?? 0) > 0
    return usedTool ? 'tool' : 'documents'
  }
  return route ?? 'unknown'
}

export function AssistantMessage({
  turn,
  actor,
  canApprove,
  deciding,
  onDecide,
}: {
  turn: TurnView
  actor: string
  canApprove: boolean
  deciding: boolean
  onDecide: (approved: boolean) => void
}) {
  // D4: the authoritative answer replaces the streamed preview the instant
  // it arrives; before that, the preview is all there is to show.
  const text = turn.answer ? turn.answer.text : turn.streamed

  return (
    <div className="message assistant">
      <span className="route-badge">{routeBadge(turn)}</span>
      <p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{text || '…'}</p>
      {turn.answer && <CitationChips citations={turn.answer.citations} actor={actor} />}
      {turn.answer && (
        <FailureBlock failure={turn.answer.failure} errorDetail={turn.answer.error_detail} />
      )}
      {turn.approval && (
        <ApprovalCard
          approval={turn.approval}
          canApprove={canApprove}
          deciding={deciding}
          onDecide={onDecide}
        />
      )}
      {turn.status === 'error' && turn.error && (
        <div className="failure-block" role="alert">
          {turn.error}
        </div>
      )}
    </div>
  )
}
