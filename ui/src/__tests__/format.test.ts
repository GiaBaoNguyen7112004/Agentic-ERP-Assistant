import { describe, expect, it } from 'vitest'
import { formatArgument, formatCost, formatRelativeTime } from '../lib/format'

describe('formatArgument', () => {
  it('renders objects as JSON, not [object Object]', () => {
    expect(formatArgument({ a: 1 })).toBe('{"a":1}')
    expect(formatArgument({ nested: { x: [1, 2] } })).toBe('{"nested":{"x":[1,2]}}')
    expect(formatArgument([])).toBe('[]')
  })

  it('renders scalars as their String', () => {
    expect(formatArgument('atlas')).toBe('atlas')
    expect(formatArgument(3)).toBe('3')
    expect(formatArgument(true)).toBe('true')
    expect(formatArgument(null)).toBe('null')
    expect(formatArgument(undefined)).toBe('undefined')
  })
})

describe('formatCost', () => {
  it("'unknown' exactly as before for null", () => {
    expect(formatCost(null)).toBe('unknown')
  })

  it('four decimals with a dollar sign, as the old summary did', () => {
    expect(formatCost(0.012345)).toBe('$0.0123')
    expect(formatCost(0)).toBe('$0.0000')
  })
})

describe('formatRelativeTime', () => {
  const now = Date.UTC(2026, 8, 12, 12, 0, 0)

  it('seconds under a minute', () => {
    expect(formatRelativeTime(new Date(now - 5_000).toISOString(), now)).toBe('5s ago')
  })

  it('minutes under an hour', () => {
    expect(formatRelativeTime(new Date(now - 3 * 60_000).toISOString(), now)).toBe('3m ago')
  })

  it('hours under a day', () => {
    expect(formatRelativeTime(new Date(now - 2 * 3_600_000).toISOString(), now)).toBe('2h ago')
  })

  it('days under a month', () => {
    expect(formatRelativeTime(new Date(now - 3 * 86_400_000).toISOString(), now)).toBe('3d ago')
  })

  it('older than a month falls back to the date', () => {
    expect(formatRelativeTime(new Date(now - 45 * 86_400_000).toISOString(), now)).toBe(
      new Date(now - 45 * 86_400_000).toLocaleDateString(),
    )
  })

  it('a future timestamp clamps to zero rather than going negative', () => {
    expect(formatRelativeTime(new Date(now + 10_000).toISOString(), now)).toBe('0s ago')
  })
})