"""A task can be continued, switched and closed -- and a switch cannot leak a
slot, because there is no call that produces the new task without closing the old
one."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.memory.intent import (
    INTENT_KEY_PREFIX,
    MAX_SLOTS,
    SLOT_VALUE_MAX_CHARS,
    IntentState,
)

OPENED = datetime(2026, 9, 8, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 8, 10, 0, tzinfo=UTC)
LATEST = datetime(2026, 9, 8, 11, 0, tzinfo=UTC)


def intent(**overrides: object) -> IntentState:
    fields: dict[str, object] = {
        "intent_id": "int-1",
        "goal": "Draft the Q4 risk register",
        "project_code": "atlas",
        "required_scope": "project.docs.read",
        "actor": "priya",
        "session_id": "sess-1",
        "unresolved_slots": ("quarter", "severity_threshold"),
        "opened_at": OPENED,
        "updated_at": OPENED,
    }
    fields.update(overrides)
    return IntentState(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Continue
# --------------------------------------------------------------------------


def test_continuing_confirms_a_slot_and_stops_asking_for_it() -> None:
    task = intent().continue_with({"quarter": "Q4"}, at=LATER)

    assert task.confirmed_slots["quarter"] == "Q4"
    assert task.unresolved_slots == ("severity_threshold",)


def test_continuing_keeps_the_same_task() -> None:
    """Continuing and switching are different methods precisely so that "is this
    the same task?" is decided once, in the open."""
    task = intent().continue_with({"quarter": "Q4"}, at=LATER)

    assert task.intent_id == "int-1"
    assert task.goal == "Draft the Q4 risk register"
    assert task.live is True


def test_continuing_leaves_the_previous_task_untouched() -> None:
    before = intent()

    before.continue_with({"quarter": "Q4"}, at=LATER)

    assert before.confirmed_slots == {}
    assert before.unresolved_slots == ("quarter", "severity_threshold")


def test_restating_a_slot_overwrites_it() -> None:
    """A user restating a value is a user correcting it."""
    task = intent().continue_with({"quarter": "Q4"}, at=LATER)

    corrected = task.continue_with({"quarter": "Q3"}, at=LATEST)

    assert corrected.confirmed_slots["quarter"] == "Q3"


def test_a_task_is_complete_when_nothing_is_outstanding() -> None:
    task = intent().continue_with(
        {"quarter": "Q4", "severity_threshold": "high"}, at=LATER
    )

    assert task.complete is True
    assert task.live is True, "complete is not closed: the work still has to happen"


def test_a_closed_task_cannot_be_continued() -> None:
    """It would reopen a task nobody decided to reopen, and the id would then
    describe two different pieces of work."""
    closed = intent().close(at=LATER)

    with pytest.raises(ValueError, match="closed"):
        closed.continue_with({"quarter": "Q4"}, at=LATEST)


# --------------------------------------------------------------------------
# Switch -- the stale-slot proof
# --------------------------------------------------------------------------


def test_switching_carries_no_confirmed_slot_across() -> None:
    """The failure this whole module is written against: the user moves to a new
    task and the assistant answers confidently about the old one's quarter."""
    task = intent().continue_with(
        {"quarter": "Q4", "severity_threshold": "high"}, at=LATER
    )

    _, fresh = task.switch_to(
        "Check the M2 budget", intent_id="int-2", at=LATEST,
        unresolved_slots=("milestone",),
    )

    assert fresh.confirmed_slots == {}
    assert fresh.unresolved_slots == ("milestone",)


def test_switching_closes_the_task_it_left() -> None:
    """Returned as a pair so there is no call that produces the new task without
    also producing the closed old one -- the version that returns only the new
    task leaves the old one open for whoever remembers."""
    task = intent().continue_with({"quarter": "Q4"}, at=LATER)

    previous, fresh = task.switch_to("Check the M2 budget", intent_id="int-2", at=LATEST)

    assert previous.live is False
    assert previous.closed_at == LATEST
    assert fresh.live is True


def test_switching_gives_the_new_task_its_own_identity() -> None:
    """Without it, "the goal changed" and "the slots were re-elicited for the
    same goal" would be the same event in the audit."""
    _, fresh = intent().switch_to("Check the M2 budget", intent_id="int-2", at=LATER)

    assert fresh.intent_id == "int-2"
    assert fresh.goal == "Check the M2 budget"


def test_switching_keeps_whose_task_it_is() -> None:
    _, fresh = intent().switch_to("Check the M2 budget", intent_id="int-2", at=LATER)

    assert fresh.project_code == "atlas"
    assert fresh.actor == "priya"
    assert fresh.session_id == "sess-1"
    assert fresh.required_scope == "project.docs.read"


def test_the_old_and_new_tasks_meet_at_one_instant() -> None:
    """What makes the sequence readable afterwards without a third status."""
    previous, fresh = intent().switch_to(
        "Check the M2 budget", intent_id="int-2", at=LATER
    )

    assert previous.closed_at == fresh.opened_at


def test_switching_away_from_an_already_closed_task_closes_nothing_twice() -> None:
    closed = intent().close(at=LATER)

    previous, fresh = closed.switch_to(
        "Check the M2 budget", intent_id="int-2", at=LATEST
    )

    assert previous is closed
    assert previous.closed_at == LATER
    assert fresh.live is True


# --------------------------------------------------------------------------
# Close
# --------------------------------------------------------------------------


def test_closing_records_when_the_work_stopped() -> None:
    closed = intent().close(at=LATER)

    assert closed.status == "closed"
    assert closed.closed_at == LATER


def test_a_closed_task_cannot_be_closed_again() -> None:
    closed = intent().close(at=LATER)

    with pytest.raises(ValueError, match="already closed"):
        closed.close(at=LATEST)


# --------------------------------------------------------------------------
# The projection into a prompt
# --------------------------------------------------------------------------


def test_the_rendering_says_the_goal_what_is_known_and_what_is_missing() -> None:
    line = intent().continue_with({"quarter": "Q4"}, at=LATER).render()

    assert "Working on: Draft the Q4 risk register." in line
    assert "Confirmed: quarter=Q4." in line
    assert "Still needed: severity_threshold." in line


def test_an_empty_task_says_so_rather_than_saying_nothing() -> None:
    line = intent().render()

    assert "Confirmed: nothing yet." in line


def test_the_rendering_is_derived_on_every_projection() -> None:
    """Not stored beside the slots: a rendering that can disagree with the data
    it describes would show the model a task whose slots were filled two turns
    ago, silently."""
    task = intent()

    before = task.render()
    after = task.continue_with({"quarter": "Q4"}, at=LATER).render()

    assert before != after


def test_a_projection_is_an_ordinary_memory_record() -> None:
    """So selection, the token budget and the access rule need no special case
    for the one kind that has a lifecycle."""
    record = intent().as_memory(
        memory_id="mem-1", recorded_in_run="run-1", recorded_at=LATER
    )

    assert record.kind == "intent"
    assert record.key == f"{INTENT_KEY_PREFIX}int-1"
    assert record.project_code == "atlas"
    assert record.required_scope == "project.docs.read"
    assert record.session_id == "sess-1"
    assert record.live is True


def test_a_closed_task_is_not_projected_at_all() -> None:
    """A finished task shown to the model is work that has stopped presented as
    work in progress."""
    closed = intent().close(at=LATER)

    with pytest.raises(ValueError, match="closed"):
        closed.as_memory(memory_id="mem-1", recorded_in_run="run-1", recorded_at=LATEST)


# --------------------------------------------------------------------------
# The bounds
# --------------------------------------------------------------------------


def test_a_task_may_not_carry_more_slots_than_the_cap() -> None:
    """An intent accumulating dozens is not a task, it is a transcript with an
    index."""
    with pytest.raises(ValidationError, match="slots"):
        intent(unresolved_slots=tuple(f"slot{n}" for n in range(MAX_SLOTS + 1)))


def test_a_slot_may_not_be_both_confirmed_and_unresolved() -> None:
    with pytest.raises(ValidationError, match="slots"):
        intent(confirmed_slots={"quarter": "Q4"}, unresolved_slots=("quarter",))


def test_a_confirmed_slot_with_no_value_is_an_unresolved_slot_misfiled() -> None:
    with pytest.raises(ValidationError, match="confirmed_slots"):
        intent(confirmed_slots={"quarter": "  "}, unresolved_slots=())


def test_a_slot_value_longer_than_the_cap_is_refused() -> None:
    with pytest.raises(ValidationError, match="confirmed_slots"):
        intent(
            confirmed_slots={"quarter": "x" * (SLOT_VALUE_MAX_CHARS + 1)},
            unresolved_slots=(),
        )


def test_confirmed_slots_cannot_be_edited_through_the_handle_that_built_them() -> None:
    slots = {"quarter": "Q4"}
    task = intent(confirmed_slots=slots, unresolved_slots=("severity_threshold",))

    slots["quarter"] = "Q1"

    assert task.confirmed_slots["quarter"] == "Q4"
    with pytest.raises(TypeError):
        task.confirmed_slots["quarter"] = "Q1"  # type: ignore[index]


def test_an_open_task_may_not_claim_to_have_closed() -> None:
    with pytest.raises(ValidationError, match="closed_at"):
        intent(status="open", closed_at=LATER)


def test_a_task_round_trips_through_plain_data() -> None:
    task = intent().continue_with({"quarter": "Q4"}, at=LATER)

    assert IntentState.model_validate(task.model_dump(mode="json")) == task
