/**
 * A human label for a node execution, from the engine's own name for it
 * (`node_entered.node` -- see `engine/workflow.py`'s `state.route or "start"`,
 * and `engine/nodes.py`'s per-node event names for the rows *inside* a span).
 *
 * Two different vocabularies meet here on purpose. The engine's own
 * `node_entered`/`node_exited` pair is named by *route* (`start`, `think`,
 * `retrieve_project_documents`, `call_tool`); the rows a node emits about its
 * own work are named by *function* (`think`, `retrieve`, `execute_tool`).
 * This map only ever sees the first kind -- a span is identified by its
 * `node_entered` row -- so `retrieve` and `execute_tool` are not members
 * here; they are inner row labels, rendered as-is in mono next to this one.
 */
export function nodeLabel(nodeName: string | null): string {
  switch (nodeName) {
    case 'start':
    case 'think':
      return 'Planner'
    case 'retrieve_project_documents':
      return 'Retrieve & compose'
    case 'call_tool':
      return 'Execute tool'
    case 'approval':
      return 'Approval'
    case 'engine':
      return 'Loop guard'
    case 'history':
    case 'memory':
    case 'contract':
      return 'Context'
    default:
      return nodeName ?? 'Turn ended'
  }
}
