import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { StateBlock } from '../components/trace/StateBlock'
import { agentState } from './fixtures'

afterEach(() => {
  Reflect.deleteProperty(navigator, 'clipboard')
})

describe('StateBlock', () => {
  it('says the state was not carried, with no trigger, when state is null', () => {
    render(<StateBlock state={null} previous={null} />)
    expect(screen.getByText(/State not carried/)).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('shows "initial" and opens in all-fields mode with no previous state', async () => {
    const user = userEvent.setup()
    render(<StateBlock state={agentState()} previous={null} />)

    expect(screen.getByText('initial')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /State/ }))
    // "actor" never changes shape but is present in all-fields mode from the start.
    expect(screen.getByText('actor')).toBeInTheDocument()
  })

  it('counts changed fields in the trigger and lists their names as chips', () => {
    const previous = agentState({ route: null, terminal: false })
    const next = agentState({ route: 'call_tool', terminal: false, step_count: 1 })
    render(<StateBlock state={next} previous={previous} />)

    expect(screen.getByText('2 changed')).toBeInTheDocument()
    expect(screen.getAllByText('route').length).toBeGreaterThan(0)
    expect(screen.getAllByText('step_count').length).toBeGreaterThan(0)
  })

  it('shows only changed rows by default, and reveals unchanged rows in "all fields" mode', async () => {
    const user = userEvent.setup()
    const previous = agentState({ route: null })
    const next = agentState({ route: 'call_tool' })
    render(<StateBlock state={next} previous={previous} />)

    await user.click(screen.getByRole('button', { name: /State/ }))
    expect(document.querySelector('[data-field="route"][data-change="changed"]')).not.toBeNull()
    expect(document.querySelector('[data-field="actor"]')).toBeNull()

    await user.click(screen.getByRole('button', { name: 'all fields' }))
    expect(document.querySelector('[data-field="actor"][data-change="unchanged"]')).not.toBeNull()
  })

  it('marks a grown array field as appended, with a count', async () => {
    const user = userEvent.setup()
    const observation = {
      tool_name: 'get_project_status',
      arguments_summary: 'milestone_id=M2',
      status: 'ok',
      summary: 'M2 is on track',
      source_ids: ['m2'],
      error: null,
      attempts: 1,
      retry_after_seconds: null,
    }
    const previous = agentState({ observations: [] })
    const next = agentState({ observations: [observation] })
    render(<StateBlock state={next} previous={previous} />)

    await user.click(screen.getByRole('button', { name: /State/ }))
    expect(screen.getByText('+1 appended')).toBeInTheDocument()
  })

  it('copies the state JSON to the clipboard', async () => {
    const user = userEvent.setup()
    const writeText = vi.fn().mockResolvedValue(undefined)
    // user-event installs its own Clipboard stub on `setup()`; ours must
    // land after that or it gets clobbered.
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    const state = agentState()
    render(<StateBlock state={state} previous={null} />)

    await user.click(screen.getByRole('button', { name: /State/ }))
    await user.click(screen.getByRole('button', { name: /copy JSON/ }))

    expect(writeText).toHaveBeenCalledWith(JSON.stringify(state, null, 2))
  })
})
