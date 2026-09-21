import type { AgentStateSnapshot } from '../protocol'

export type StateField = keyof AgentStateSnapshot

/**
 * The four sections `state/agent_state.py` groups its fields into, in the
 * model's own order -- so the State block reads like the class it mirrors
 * rather than an alphabetized dump. `stateDiff.test.ts` asserts every key of
 * `AgentStateSnapshot` appears here exactly once, so a field added to the
 * Python model and forgotten here fails a test instead of silently missing
 * from the panel.
 */
export const STATE_FIELD_GROUPS: readonly { title: string; fields: readonly StateField[] }[] = [
  { title: 'Turn', fields: ['request', 'actor', 'project_code', 'scopes', 'trace_id', 'session_id'] },
  {
    title: 'Decided and gathered',
    fields: [
      'route',
      'evidence',
      'memories',
      'history',
      'contract',
      'redirected_needs',
      'draft',
      'observations',
      'tool_name',
      'tool_arguments',
      'tool_mutating',
      'approval',
    ],
  },
  { title: 'How it ended', fields: ['response', 'failure', 'error_detail', 'terminal'] },
  { title: 'Record', fields: ['step_count', 'retry_count', 'events', 'state_version'] },
]

/** Fields the server serializes from a Python `frozenset` -- order carries
 * no meaning on the wire, so these are compared as sorted arrays rather
 * than by position. */
const SET_FIELDS: ReadonlySet<StateField> = new Set(['scopes', 'redirected_needs'])

export type ChangeKind = 'unchanged' | 'changed' | 'appended'

export interface FieldChange {
  field: StateField
  kind: ChangeKind
  /** `null` when there is no previous state (the initial state itself). */
  before: unknown
  after: unknown
  /** For `kind === 'appended'`: the elements `after` has beyond `before`.
   * `null` for every other kind. */
  appended: unknown[] | null
}

/** Structural equality, key-order-insensitive for plain objects and arrays.
 * Small and local on purpose -- this compares wire-JSON-shaped values only
 * (strings, numbers, booleans, null, arrays, plain objects), never a class
 * instance or a Map/Set/Date. */
export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (a === null || b === null) return false
  if (typeof a !== 'object' || typeof b !== 'object') return false

  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) return false
    if (a.length !== b.length) return false
    return a.every((item, index) => deepEqual(item, b[index]))
  }

  const aKeys = Object.keys(a as Record<string, unknown>)
  const bKeys = Object.keys(b as Record<string, unknown>)
  if (aKeys.length !== bKeys.length) return false
  return aKeys.every((key) =>
    deepEqual((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]),
  )
}

function sortedForCompare(field: StateField, value: unknown): unknown {
  if (!SET_FIELDS.has(field) || !Array.isArray(value)) return value
  return [...value].sort()
}

/**
 * One entry per field of `next`, in `STATE_FIELD_GROUPS` order. With no
 * `previous` every field is `changed` with `before: null` -- the initial
 * state: everything is new. An array field whose previous value is an
 * exact prefix of the next one is `appended`, never `changed`: that is what
 * "a node added an observation" looks like, and the block renders only the
 * tail rather than repeating what the reader already saw on an earlier
 * card.
 */
export function diffState(
  previous: AgentStateSnapshot | null,
  next: AgentStateSnapshot,
): FieldChange[] {
  const fields = STATE_FIELD_GROUPS.flatMap((group) => group.fields)

  return fields.map((field): FieldChange => {
    const after = next[field]

    if (previous === null) {
      return { field, kind: 'changed', before: null, after, appended: null }
    }

    const before = previous[field]
    const compareBefore = sortedForCompare(field, before)
    const compareAfter = sortedForCompare(field, after)

    if (deepEqual(compareBefore, compareAfter)) {
      return { field, kind: 'unchanged', before, after, appended: null }
    }

    if (
      Array.isArray(before) &&
      Array.isArray(after) &&
      after.length > before.length &&
      before.every((item, index) => deepEqual(item, after[index]))
    ) {
      return { field, kind: 'appended', before, after, appended: after.slice(before.length) }
    }

    return { field, kind: 'changed', before, after, appended: null }
  })
}

/** The `changed`/`appended` entries only -- what the block's trigger line
 * lists and what "changed only" mode renders. */
export function changedFields(changes: FieldChange[]): FieldChange[] {
  return changes.filter((change) => change.kind !== 'unchanged')
}
