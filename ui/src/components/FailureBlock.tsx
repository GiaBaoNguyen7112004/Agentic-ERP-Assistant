import { CircleAlert } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'

export function FailureBlock({
  failure,
  errorDetail,
}: {
  failure: string
  errorDetail: string | null
}) {
  if (failure === 'none') return null

  return (
    <Alert variant="destructive">
      <CircleAlert aria-hidden />
      <AlertTitle className="font-mono">{failure}</AlertTitle>
      {errorDetail && <AlertDescription className="break-words">{errorDetail}</AlertDescription>}
    </Alert>
  )
}