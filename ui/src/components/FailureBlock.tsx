import { CircleAlert, TriangleAlert } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'

/**
 * Failure modes that still delivered a reply. ADR 0021's `incomplete_reply`
 * is the planner's own words handed over after a redirected search could
 * not ground them (nothing found, or passages that did not support the
 * reply) -- an answer with a caveat, not a turn that failed. It reads as a
 * warning under the text; every other failure keeps the red block.
 */
const DELIVERED_WITH_CAVEAT: Record<string, string> = {
  incomplete_reply: 'not confirmed by project documents',
}

export function FailureBlock({
  failure,
  errorDetail,
}: {
  failure: string
  errorDetail: string | null
}) {
  if (failure === 'none') return null

  const caveat = DELIVERED_WITH_CAVEAT[failure]
  if (caveat !== undefined) {
    return (
      <Alert variant="warning">
        <TriangleAlert aria-hidden />
        <AlertTitle>
          {caveat} <span className="font-mono text-xs opacity-80">({failure})</span>
        </AlertTitle>
        {errorDetail && <AlertDescription className="break-words">{errorDetail}</AlertDescription>}
      </Alert>
    )
  }

  return (
    <Alert variant="destructive">
      <CircleAlert aria-hidden />
      <AlertTitle className="font-mono">{failure}</AlertTitle>
      {errorDetail && <AlertDescription className="break-words">{errorDetail}</AlertDescription>}
    </Alert>
  )
}
