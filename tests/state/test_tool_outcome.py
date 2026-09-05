"""An outcome a reader has to guess at is not a record of anything."""

from typing import get_args

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.state.tool_outcome import ToolOutcome, ToolStatus

FAILURES = tuple(status for status in get_args(ToolStatus) if status != "ok")


def succeeded(**overrides: object) -> ToolOutcome:
    fields: dict[str, object] = {
        "tool_name": "get_project_status",
        "status": "ok",
        "summary": "M2 is two days behind, budget at 61%.",
        "source_ids": ("milestone:M2",),
    }
    fields.update(overrides)
    return ToolOutcome(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Success has to be self-describing
# --------------------------------------------------------------------------


def test_a_successful_call_reports_what_it_found_and_where_from() -> None:
    outcome = succeeded()

    assert outcome.summary
    assert outcome.source_ids == ("milestone:M2",)
    assert outcome.error is None


def test_a_successful_call_cannot_also_report_an_error() -> None:
    with pytest.raises(ValidationError, match="error"):
        succeeded(error="but also broken")


def test_a_successful_call_must_say_something() -> None:
    with pytest.raises(ValidationError, match="summary"):
        succeeded(summary="   ")


def test_a_successful_call_must_name_what_it_touched() -> None:
    """Stricter than 'a read must cite its sources': a write has to name the
    record it changed too, and that is the case where it matters most."""
    with pytest.raises(ValidationError, match="source_ids"):
        succeeded(source_ids=())


def test_a_blank_source_identifier_does_not_count() -> None:
    with pytest.raises(ValidationError, match="source_ids"):
        succeeded(source_ids=("milestone:M2", ""))


def test_source_ids_cannot_be_appended_to_after_the_fact() -> None:
    outcome = succeeded()

    assert isinstance(outcome.source_ids, tuple)
    with pytest.raises(AttributeError):
        outcome.source_ids.append("milestone:M9")  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Failure is data, and every kind of it is a different next move
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", FAILURES)
def test_every_failure_must_say_why(status: str) -> None:
    """Otherwise the trace records that something went wrong and nothing about
    what."""
    with pytest.raises(ValidationError, match="error"):
        ToolOutcome(tool_name="close_milestone", status=status)


@pytest.mark.parametrize("status", FAILURES)
def test_a_blank_reason_is_no_reason(status: str) -> None:
    with pytest.raises(ValidationError, match="error"):
        ToolOutcome(tool_name="close_milestone", status=status, error="   ")


def test_never_asked_and_said_no_are_different_statuses() -> None:
    """They look alike -- nothing ran either way -- and lead to opposite moves.
    Under one label the graph would either re-ask a human who already refused,
    or abandon a call that was never put to anyone."""
    unasked = ToolOutcome(
        tool_name="create_risk",
        status="approval_required",
        error="a write needs a human decision before it runs",
    )
    refused = ToolOutcome(
        tool_name="create_risk", status="denied", error="approval was denied"
    )

    assert unasked.status != refused.status


def test_a_refusal_and_a_breakage_are_different_statuses() -> None:
    """One is the safety layer working, the other is the system not working.
    A reviewer counting refusals must not be counting outages."""
    denied = ToolOutcome(
        tool_name="close_milestone", status="denied", error="approval was denied"
    )
    broke = ToolOutcome(
        tool_name="close_milestone", status="failed", error="the ERP rejected it"
    )

    assert denied.status != broke.status


def test_a_failure_needs_no_sources() -> None:
    outcome = ToolOutcome(
        tool_name="close_milestone", status="denied", error="approval was denied"
    )

    assert outcome.source_ids == ()


def test_an_unknown_status_word_is_rejected() -> None:
    with pytest.raises(ValidationError, match="status"):
        ToolOutcome(tool_name="close_milestone", status="probably_fine")


def test_the_status_is_not_a_boolean_in_disguise() -> None:
    """No .ok shortcut: that is exactly how five outcomes get collapsed back
    into two and the distinctions above stop being read."""
    assert not hasattr(succeeded(), "ok")


# --------------------------------------------------------------------------
# What the call cost, and what to do next
# --------------------------------------------------------------------------


def test_an_outcome_records_at_least_one_attempt() -> None:
    assert succeeded().attempts == 1


def test_a_retried_call_records_what_it_spent() -> None:
    """So a trace shows a call that succeeded on the third try as different
    from one that succeeded outright."""
    assert succeeded(attempts=3).attempts == 3


def test_an_outcome_cannot_claim_it_was_never_attempted() -> None:
    with pytest.raises(ValidationError, match="attempts"):
        succeeded(attempts=0)


def test_retry_after_is_accepted_on_a_transient_failure() -> None:
    """Defined before anything sets it: settling the contract once is cheaper
    than a schema change once consumers exist."""
    outcome = ToolOutcome(
        tool_name="get_project_status",
        status="transient_failure",
        error="429 from the ERP",
        retry_after_seconds=2.5,
    )

    assert outcome.retry_after_seconds == 2.5


@pytest.mark.parametrize(
    "status", ["ok", "denied", "approval_required", "invalid_arguments", "failed"]
)
def test_retry_after_is_rejected_where_nothing_could_act_on_it(status: str) -> None:
    """Constrained rather than merely optional -- an early field that no branch
    can read is how an early field becomes one nobody trusts."""
    fields: dict[str, object] = {
        "tool_name": "close_milestone",
        "status": status,
        "retry_after_seconds": 2.5,
    }
    if status == "ok":
        fields |= {"summary": "done", "source_ids": ("milestone:M2",)}
    else:
        fields |= {"error": "no"}

    with pytest.raises(ValidationError, match="retry_after_seconds"):
        ToolOutcome(**fields)  # type: ignore[arg-type]


def test_a_negative_wait_is_rejected() -> None:
    with pytest.raises(ValidationError, match="retry_after_seconds"):
        ToolOutcome(
            tool_name="get_project_status",
            status="transient_failure",
            error="429",
            retry_after_seconds=-1.0,
        )


# --------------------------------------------------------------------------
# The record itself
# --------------------------------------------------------------------------


def test_an_outcome_must_name_its_tool() -> None:
    """So a trace or audit entry stands on its own."""
    with pytest.raises(ValidationError, match="tool_name"):
        succeeded(tool_name="")


def test_an_outcome_cannot_be_edited_after_the_tool_ran() -> None:
    outcome = succeeded()

    with pytest.raises(ValidationError):
        outcome.summary = "something else"  # type: ignore[misc]


def test_an_unmodelled_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        succeeded(raw_payload={"secret": "x"})
