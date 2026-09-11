"""A graph is only replayable if a transition cannot touch the state it came from."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.state.agent_state import (
    AgentState,
    ERROR_DETAIL_MAX_CHARS,
    STATE_VERSION,
)
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.events import TraceEvent
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.state.tool_outcome import ToolOutcome


def initial() -> AgentState:
    """The smallest state a turn can start from: the three required fields."""
    return AgentState(request="How is M2 tracking?", actor="bao", trace_id="run-1")


# --------------------------------------------------------------------------
# The acceptance criterion: evolve never touches the original
# --------------------------------------------------------------------------


def test_evolve_returns_a_different_object() -> None:
    state = initial()

    assert state.evolve(step_count=1) is not state


def test_the_original_is_byte_for_byte_unchanged_after_evolve() -> None:
    """Not just 'the field we changed' -- the whole state, so a transition that
    reached in sideways would show up here."""
    state = initial()
    before = state.model_dump()

    state.evolve(
        route="retrieve_project_documents",
        step_count=7,
        retry_count=2,
        response="tracking to plan",
        terminal=True,
    )

    assert state.model_dump() == before


def test_the_original_keeps_its_own_field_values() -> None:
    state = initial()

    evolved = state.evolve(route="answer", step_count=3)

    assert state.route is None
    assert state.step_count == 0
    assert evolved.route == "answer"
    assert evolved.step_count == 3


def test_fields_left_out_of_evolve_are_carried_over() -> None:
    state = initial().evolve(tool_name="get_project_status", step_count=4)

    evolved = state.evolve(step_count=5)

    assert evolved.tool_name == "get_project_status"
    assert evolved.request == state.request
    assert evolved.trace_id == state.trace_id


def test_a_chain_of_transitions_leaves_every_earlier_state_intact() -> None:
    """What replay and retry depend on: each link is still readable afterwards."""
    first = initial()
    second = first.evolve(step_count=1, route="retrieve_project_documents")
    third = second.evolve(step_count=2, route="answer")

    assert (first.step_count, second.step_count, third.step_count) == (0, 1, 2)
    assert first.route is None
    assert second.route == "retrieve_project_documents"


def test_a_state_cannot_be_assigned_to_in_place() -> None:
    """The reason the trace can claim a node saw the state it was handed."""
    state = initial()

    with pytest.raises(ValidationError):
        state.step_count = 1  # type: ignore[misc]


# --------------------------------------------------------------------------
# evolve re-validates: the invariants hold for transitions, not just for
# freshly constructed states
# --------------------------------------------------------------------------


def test_evolve_rejects_a_value_the_constructor_would_have_rejected() -> None:
    """model_copy(update=...) would have written this through unchecked, and
    every state after the first is reached through evolve."""
    with pytest.raises(ValidationError, match="step_count"):
        initial().evolve(step_count=-1)


def test_evolve_rejects_a_combination_that_only_the_validator_catches() -> None:
    """A cross-field rule, not a field constraint -- proving the whole
    validator runs on a transition and not merely the per-field checks."""
    with pytest.raises(ValidationError, match="approval"):
        initial().evolve(approval="approved")


def test_evolve_rejects_a_field_that_does_not_exist() -> None:
    """Guarded by extra='forbid', the same mechanism that guards the
    constructor, so there is no second rule to keep in step."""
    with pytest.raises(ValidationError):
        initial().evolve(setp_count=1)


def test_evolve_with_no_changes_produces_an_equal_but_separate_state() -> None:
    state = initial()

    same = state.evolve()

    assert same == state
    assert same is not state


# --------------------------------------------------------------------------
# The required fields: a turn that cannot be traced must not exist
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["request", "actor", "trace_id"])
def test_the_identifying_fields_have_no_default(field: str) -> None:
    kwargs = {"request": "q", "actor": "bao", "trace_id": "run-1"}
    del kwargs[field]

    with pytest.raises(ValidationError, match=field):
        AgentState(**kwargs)


@pytest.mark.parametrize("field", ["request", "actor", "trace_id"])
def test_the_identifying_fields_cannot_be_empty(field: str) -> None:
    kwargs = {"request": "q", "actor": "bao", "trace_id": "run-1"}
    kwargs[field] = ""

    with pytest.raises(ValidationError, match=field):
        AgentState(**kwargs)


# --------------------------------------------------------------------------
# Route: the decision layer's set, and 'undecided' kept apart from 'answer'
# --------------------------------------------------------------------------


def test_a_new_state_has_not_routed_yet() -> None:
    assert initial().route is None


def test_deciding_to_reply_is_a_route_not_the_absence_of_one() -> None:
    """'answer' and None must stay different values: the first is a node to
    dispatch, the second is a turn that never routed."""
    assert initial().evolve(route="answer").route == "answer"


def test_a_label_the_decision_layer_does_not_know_is_rejected() -> None:
    with pytest.raises(ValidationError, match="route"):
        initial().evolve(route="do_something_clever")


# --------------------------------------------------------------------------
# Approval: constraint 4 lives in the type, not in a caller's discipline
# --------------------------------------------------------------------------


def test_a_fresh_state_needs_no_approval() -> None:
    """'not_required' is a member, so a gate has to match a branch rather than
    writing `if state.approval:` and treating four outcomes as two."""
    state = initial()

    assert state.approval == "not_required"
    assert state.approval is not None


@pytest.mark.parametrize("decision", ["pending", "approved", "denied"])
def test_an_approval_must_name_the_tool_it_is_about(decision: str) -> None:
    with pytest.raises(ValidationError, match="approval"):
        initial().evolve(approval=decision)


@pytest.mark.parametrize("decision", ["pending", "approved", "denied"])
def test_an_approval_alongside_its_tool_is_valid(decision: str) -> None:
    state = initial().evolve(tool_name="close_milestone", approval=decision)

    assert state.approval == decision
    assert state.tool_name == "close_milestone"


def test_an_unknown_approval_word_is_rejected() -> None:
    with pytest.raises(ValidationError, match="approval"):
        initial().evolve(tool_name="close_milestone", approval="probably")


def test_a_blank_tool_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="tool_name"):
        initial().evolve(tool_name="   ")


# --------------------------------------------------------------------------
# Ending a turn
# --------------------------------------------------------------------------


def test_a_turn_that_ends_with_neither_answer_nor_failure_cannot_exist() -> None:
    """It would leave the caller nothing and the trace no reason."""
    with pytest.raises(ValidationError, match="terminal"):
        initial().evolve(terminal=True)


def test_a_turn_may_end_with_a_response() -> None:
    state = initial().evolve(terminal=True, response="M2 is two days behind.")

    assert state.terminal is True


def test_a_turn_may_end_with_a_failure_and_no_response() -> None:
    state = initial().evolve(terminal=True, failure="insufficient_evidence")

    assert state.terminal is True
    assert state.response is None


def test_a_clean_turn_reports_the_string_none_for_failure() -> None:
    assert initial().failure == "none"


def test_error_detail_is_capped_like_a_rationale() -> None:
    """Otherwise it becomes where a stack trace or a whole prompt is stored,
    and then read out of an exported trace as if it had been reviewed."""
    state = initial().evolve(
        failure="provider_failure", error_detail="x" * ERROR_DETAIL_MAX_CHARS
    )

    assert len(state.error_detail or "") == ERROR_DETAIL_MAX_CHARS

    with pytest.raises(ValidationError, match="error_detail"):
        initial().evolve(
            failure="provider_failure",
            error_detail="x" * (ERROR_DETAIL_MAX_CHARS + 1),
        )


# --------------------------------------------------------------------------
# The sequences are tuples, so 'frozen' is not merely cosmetic
# --------------------------------------------------------------------------


def test_evidence_is_a_tuple_that_cannot_be_appended_to() -> None:
    snippet = EvidenceSnippet(
        source_id="sprint-12-report.md", locator="3.2", text="M2 slipped."
    )
    state = initial().evolve(evidence=(snippet,))

    assert isinstance(state.evidence, tuple)
    with pytest.raises(AttributeError):
        state.evidence.append(snippet)  # type: ignore[attr-defined]


def test_a_list_of_evidence_is_coerced_to_a_tuple() -> None:
    """A caller passing a list must not end up holding a mutable handle into a
    state that claims to be frozen."""
    snippet = EvidenceSnippet(
        source_id="budget-q3.md", locator="p.4", text="Spend at 61%."
    )
    supplied = [snippet]

    state = initial().evolve(evidence=supplied)
    supplied.append(snippet)

    assert state.evidence == (snippet,)


def test_the_event_log_is_a_tuple_and_evolve_carries_it_forward() -> None:
    entered = TraceEvent(node="retrieve", kind="node_entered")
    state = initial().evolve(events=(entered,))

    later = state.evolve(events=state.events + (TraceEvent(node="retrieve", kind="node_exited"),))

    assert state.events == (entered,)
    assert [event.kind for event in later.events] == ["node_entered", "node_exited"]


# --------------------------------------------------------------------------
# Versioning and serialization
# --------------------------------------------------------------------------


def test_a_state_records_the_shape_it_was_written_in() -> None:
    assert initial().state_version == STATE_VERSION


def test_a_state_from_another_version_is_not_loaded_silently() -> None:
    """It needs a migration; a quiet load would reinterpret fields whose
    meaning changed."""
    with pytest.raises(ValidationError, match="state_version"):
        AgentState(
            request="q", actor="bao", trace_id="run-1", state_version=STATE_VERSION + 1
        )


def test_a_state_survives_a_round_trip_through_plain_data() -> None:
    """Serializable is a requirement, not a convenience: a trace store has to
    write these down and read them back."""
    snippet = EvidenceSnippet(
        source_id="budget-q3.md", locator="p.4", text="Spend at 61%."
    )
    state = initial().evolve(
        route="answer",
        evidence=(snippet,),
        events=(TraceEvent(node="answer", kind="route_selected", detail="answer"),),
        response="Spend is at 61%.",
        terminal=True,
    )

    assert AgentState.model_validate(state.model_dump()) == state


def test_an_unmodelled_field_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        AgentState(request="q", actor="bao", trace_id="run-1", escalate=True)


# --------------------------------------------------------------------------
# What a paused turn has to survive on
# --------------------------------------------------------------------------


def test_an_actor_with_no_declared_scopes_is_entitled_to_nothing() -> None:
    """The direction an omission has to fail in."""
    assert initial().scopes == frozenset()


def test_arguments_are_carried_beside_the_tool_they_belong_to() -> None:
    state = initial().evolve(
        tool_name="create_risk",
        tool_arguments={"project_id": "PRJ-1", "title": "Vendor slip"},
        tool_mutating=True,
    )

    assert state.tool_arguments == {"project_id": "PRJ-1", "title": "Vendor slip"}


def test_arguments_without_a_tool_name_cannot_exist() -> None:
    with pytest.raises(ValidationError, match="tool_arguments"):
        initial().evolve(tool_arguments={"project_id": "PRJ-1"})


def test_the_mutating_flag_without_a_tool_name_cannot_exist() -> None:
    with pytest.raises(ValidationError, match="tool_mutating"):
        initial().evolve(tool_mutating=True)


def test_the_caller_cannot_edit_arguments_an_approver_has_read() -> None:
    """The pause is the reason: a handle kept on the dict would outlive the
    approval decision made about it."""
    arguments = {"project_id": "PRJ-1", "severity": "high"}
    state = initial().evolve(tool_name="create_risk", tool_arguments=arguments)

    arguments["severity"] = "low"

    assert state.tool_arguments["severity"] == "high"
    with pytest.raises(TypeError):
        state.tool_arguments["severity"] = "low"


def test_observations_accumulate_rather_than_replace() -> None:
    first = ToolOutcome(
        tool_name="list_risks", status="ok", summary="2 open", source_ids=("PRJ-1",)
    )
    second = ToolOutcome(
        tool_name="get_budget_summary",
        status="ok",
        summary="61%",
        source_ids=("PRJ-1",),
    )
    state = initial().evolve(observations=(first,))

    state = state.evolve(observations=state.observations + (second,))

    assert state.observations == (first, second)


def test_a_paused_write_survives_a_round_trip_through_plain_data() -> None:
    """The pause can outlive the process; what resumes it reads this back."""
    state = initial().evolve(
        route="request_approval",
        tool_name="create_risk",
        tool_arguments={"project_id": "PRJ-1", "title": "Vendor slip"},
        tool_mutating=True,
        approval="pending",
        scopes=frozenset({"project.risk.write"}),
        observations=(ToolOutcome(
                tool_name="list_risks",
                status="ok",
                summary="2",
                source_ids=("PRJ-1",),
            ),),
    )

    assert AgentState.model_validate(state.model_dump()) == state


# --------------------------------------------------------------------------
# Memory: carried like evidence, and absent by default
# --------------------------------------------------------------------------


def memory() -> MemoryRecord:
    return MemoryRecord(
        memory_id="mem-1",
        kind="preference",
        key="reply_language",
        statement="Prefers replies in Vietnamese.",
        project_code="atlas",
        required_scope="project.docs.read",
        actor="bao",
        session_id="sess-1",
        recorded_in_run="run-1",
        recorded_at=datetime(2026, 9, 8, tzinfo=UTC),
        confidence=0.9,
    )


def test_a_turn_starts_with_no_memory_and_no_session() -> None:
    """A caller that says nothing gets a turn that remembers nothing, which is
    the direction an omission should fail in."""
    state = initial()

    assert state.memories == ()
    assert state.session_id is None


def test_memory_survives_evolve_like_any_other_field() -> None:
    state = initial().evolve(session_id="sess-1", memories=(memory(),))

    carried = state.evolve(step_count=1)

    assert carried.memories == (memory(),)
    assert carried.session_id == "sess-1"


def test_a_state_carrying_memory_round_trips_through_plain_data() -> None:
    """It is stored as one jsonb document, so the nested record has to survive."""
    state = initial().evolve(session_id="sess-1", memories=(memory(),))

    assert AgentState.model_validate(state.model_dump()) == state


def test_a_state_written_before_memory_existed_still_loads() -> None:
    """Why adding these two fields needed no STATE_VERSION bump: the stored
    document predates them and the defaults are the truthful reading of it."""
    stored = initial().model_dump()
    del stored["memories"]
    del stored["session_id"]

    loaded = AgentState.model_validate(stored)

    assert loaded.memories == ()
    assert loaded.session_id is None
    assert loaded.state_version == STATE_VERSION


def turn() -> ConversationTurn:
    return ConversationTurn(
        trace_id="run-0",
        session_id="sess-1",
        actor="bao",
        request="How is M2 tracking?",
        response="On track.",
        route="answer",
        started_at=datetime(2026, 9, 8, tzinfo=UTC),
        finished_at=datetime(2026, 9, 8, tzinfo=UTC),
    )


def test_a_turn_starts_with_no_history() -> None:
    state = initial()

    assert state.history == ()


def test_a_stored_state_without_history_loads_with_none() -> None:
    stored = initial().model_dump()
    del stored["history"]

    loaded = AgentState.model_validate(stored)

    assert loaded.history == ()
    assert loaded.state_version == STATE_VERSION


def test_evolve_carries_history() -> None:
    state = initial().evolve(session_id="sess-1", history=(turn(),))

    carried = state.evolve(step_count=1)

    assert carried.history == (turn(),)
    assert carried.session_id == "sess-1"


def test_a_state_carrying_history_round_trips_through_plain_data() -> None:
    state = initial().evolve(session_id="sess-1", history=(turn(),))

    assert AgentState.model_validate(state.model_dump()) == state
