import type { StepEvent } from '../protocol'

export function DecisionList({ decisions }: { decisions: StepEvent[] }) {
  if (decisions.length === 0) return null

  return (
    <ol className="decision-list">
      {decisions.map((decision, index) => (
        <li key={index}>
          <strong>{decision.route ?? '(none)'}</strong>
          {decision.tool_name && ` -> ${decision.tool_name}`}
          {decision.tool_arguments && (
            <pre style={{ margin: '4px 0', whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(decision.tool_arguments, null, 2)}
            </pre>
          )}
        </li>
      ))}
    </ol>
  )
}
