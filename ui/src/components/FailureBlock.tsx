export function FailureBlock({
  failure,
  errorDetail,
}: {
  failure: string
  errorDetail: string | null
}) {
  if (failure === 'none') return null

  return (
    <div className="failure-block" role="alert">
      <strong>{failure}</strong>
      {errorDetail ? `: ${errorDetail}` : null}
    </div>
  )
}
