"""The execution envelope: nothing may run on facts the request never carried."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.llm.tools import GET_PROJECT_STATUS_TOOL
from agentic_erp_assistant.tools.models import (
    ARGUMENTS_SUMMARY_MAX_CHARS,
    AuditRow,
    ToolError,
    ToolOutcome,
    ToolRequest,
    TransientToolError,
)

WHEN = datetime(2026, 9, 5, 9, 30, tzinfo=timezone.utc)


def request(**overrides: object) -> ToolRequest:
    fields: dict[str, object] = {
        "tool_name": "close_milestone",
        "arguments": {"milestone_id": "M2"},
        "actor": "bao",
        "scopes": frozenset({"erp:write"}),
    }
    fields.update(overrides)
    return ToolRequest(**fields)  # type: ignore[arg-type]


def row(**overrides: object) -> AuditRow:
    fields: dict[str, object] = {
        "occurred_at": WHEN,
        "actor": "bao",
        "tool_name": "close_milestone",
        "arguments_summary": "move M2 to done",
        "approval": "approved",
        "status": "ok",
        "source_ids": ("milestone:M2",),
    }
    fields.update(overrides)
    return AuditRow(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Every request says who is asking and what they are entitled to
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["tool_name", "arguments", "actor", "scopes"])
def test_a_request_cannot_be_built_without_it(field: str) -> None:
    """None of the four has a default. A call missing any of them is one no
    check downstream could make a decision about."""
    fields = {
        "tool_name": "t",
        "arguments": {},
        "actor": "bao",
        "scopes": frozenset(),
    }
    del fields[field]

    with pytest.raises(ValidationError, match=field):
        ToolRequest(**fields)  # type: ignore[arg-type]


def test_an_anonymous_request_is_rejected() -> None:
    """An unattributed call cannot be audited, and the audit row is what the
    approval rule exists to produce."""
    with pytest.raises(ValidationError, match="actor"):
        request(actor="")


def test_a_request_carries_its_scopes_even_when_it_has_none() -> None:
    """Empty is a real answer: 'entitled to nothing'. It is not the same as
    'nobody said', which is why the field has no default."""
    assert request(scopes=frozenset()).scopes == frozenset()


def test_a_blank_scope_is_rejected() -> None:
    with pytest.raises(ValidationError, match="scopes"):
        request(scopes=frozenset({"erp:write", "  "}))


def test_a_request_needs_no_approval_until_something_asks_for_one() -> None:
    """'not_required' is a member, so a gate matches a branch instead of
    testing a value for truth."""
    assert request().approval == "not_required"


def test_an_approval_decision_travels_with_the_request() -> None:
    assert request(approval="approved").approval == "approved"


def test_an_unknown_approval_word_is_rejected() -> None:
    with pytest.raises(ValidationError, match="approval"):
        request(approval="probably")


# --------------------------------------------------------------------------
# Arguments: unreachable to a handler unless the tool declared them
# --------------------------------------------------------------------------


def test_an_argument_the_tool_never_declared_cannot_reach_a_handler() -> None:
    """The acceptance criterion, proved against the real registry entry rather
    than restated. ToolSpec's model forbids extras, so an invented key raises
    on the way in instead of being ignored on the way through."""
    with pytest.raises(ValidationError):
        GET_PROJECT_STATUS_TOOL.validate_arguments(
            {"milestone_id": "M2", "force": True}
        )


def test_the_declared_arguments_do_pass() -> None:
    validated = GET_PROJECT_STATUS_TOOL.validate_arguments({"milestone_id": "M2"})

    assert validated.milestone_id == "M2"  # type: ignore[attr-defined]


def test_arguments_cannot_be_changed_after_a_request_is_built() -> None:
    """A frozen model holding a plain dict is not frozen, it merely looks it --
    and an approver would have read the arguments before the swap."""
    built = request()

    with pytest.raises(TypeError):
        built.arguments["milestone_id"] = "M9"  # type: ignore[index]


def test_the_caller_keeps_no_handle_on_the_arguments() -> None:
    supplied = {"milestone_id": "M2"}

    built = request(arguments=supplied)
    supplied["milestone_id"] = "M9"

    assert built.arguments["milestone_id"] == "M2"


def test_a_request_still_serializes_as_plain_data() -> None:
    """The read-only view is an in-process guard, not part of the wire shape:
    a trace record wants the mapping."""
    dumped = request().model_dump()

    assert dumped["arguments"] == {"milestone_id": "M2"}
    assert ToolRequest.model_validate(dumped) == request()


def test_a_request_cannot_be_edited_after_it_is_authorized() -> None:
    with pytest.raises(ValidationError):
        request().tool_name = "something_else"  # type: ignore[misc]


def test_an_unmodelled_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        request(bypass_approval=True)


# --------------------------------------------------------------------------
# The audit row: who changed what, when, and on whose say-so
# --------------------------------------------------------------------------


def test_an_audit_row_records_the_five_facts_an_auditor_asks_for() -> None:
    recorded = row()

    assert (recorded.actor, recorded.tool_name, recorded.approval) == (
        "bao",
        "close_milestone",
        "approved",
    )
    assert recorded.occurred_at == WHEN
    assert recorded.source_ids == ("milestone:M2",)


def test_an_audit_row_does_not_stamp_itself() -> None:
    """No default_factory reading the clock: a self-stamping model cannot be
    asserted on, and only the caller knows whether the time means the start or
    the end of the attempt."""
    with pytest.raises(ValidationError, match="occurred_at"):
        AuditRow(
            actor="bao",
            tool_name="close_milestone",
            arguments_summary="move M2 to done",
            approval="approved",
            status="ok",
        )


def test_an_audit_row_must_state_what_the_approver_said() -> None:
    """No fallback to 'not_required': the one fact the row exists for would go
    unstated by omission."""
    with pytest.raises(ValidationError, match="approval"):
        AuditRow(
            occurred_at=WHEN,
            actor="bao",
            tool_name="close_milestone",
            arguments_summary="move M2 to done",
            status="ok",
        )


def test_a_denied_call_cannot_be_recorded_as_having_succeeded() -> None:
    """A row carrying both is worse than no row: it is evidence that reads as
    an approved change and evidence that reads as a refusal."""
    with pytest.raises(ValidationError, match="status"):
        row(approval="denied", status="ok")


def test_a_denied_call_is_recorded_with_a_denied_outcome() -> None:
    assert row(approval="denied", status="denied", source_ids=()).status == "denied"


def test_the_summary_is_capped_so_it_cannot_become_the_payload() -> None:
    """This is the line an approver reads and an auditor reads back. Uncapped,
    it will eventually hold a full argument dict -- credentials included --
    rendered to a screen and written to a record that outlives the run."""
    assert len(row(arguments_summary="x" * ARGUMENTS_SUMMARY_MAX_CHARS).arguments_summary) == (
        ARGUMENTS_SUMMARY_MAX_CHARS
    )

    with pytest.raises(ValidationError, match="arguments_summary"):
        row(arguments_summary="x" * (ARGUMENTS_SUMMARY_MAX_CHARS + 1))


def test_an_empty_summary_is_no_summary() -> None:
    with pytest.raises(ValidationError, match="arguments_summary"):
        row(arguments_summary="")


def test_a_blank_source_identifier_is_rejected() -> None:
    with pytest.raises(ValidationError, match="source_ids"):
        row(source_ids=("milestone:M2", " "))


def test_an_audit_row_cannot_be_rewritten() -> None:
    with pytest.raises(ValidationError):
        row().approval = "denied"  # type: ignore[misc]


def test_the_row_and_the_outcome_speak_one_vocabulary() -> None:
    """So a row and the outcome it was written from cannot disagree about what
    happened."""
    outcome = ToolOutcome(
        tool_name="close_milestone",
        status="transient_failure",
        error="the ERP timed out",
        retry_after_seconds=2.0,
    )

    assert row(status=outcome.status, approval="approved", source_ids=()).status == (
        outcome.status
    )


# --------------------------------------------------------------------------
# Exceptions live inside the boundary
# --------------------------------------------------------------------------


def test_a_transient_failure_is_its_own_type() -> None:
    """An except clause is what decides what gets retried, and it cannot read a
    flag it never caught."""
    assert issubclass(TransientToolError, ToolError)


def test_catching_the_base_catches_the_transient_one() -> None:
    with pytest.raises(ToolError):
        raise TransientToolError("the ERP timed out")


def test_no_second_validation_error_is_declared_here() -> None:
    """Arguments are checked against the tool's own pydantic model, which
    raises pydantic's ValidationError. A second name for the same failure would
    make every caller catch two things to catch one problem."""
    from agentic_erp_assistant.tools import models

    assert not hasattr(models, "ValidationError")


# --------------------------------------------------------------------------
# One file to read
# --------------------------------------------------------------------------


def test_the_shared_types_are_re_exported_and_not_redefined() -> None:
    """A reviewer sees the whole envelope in one file, and there is still one
    definition of each piece."""
    from agentic_erp_assistant.state.agent_state import ApprovalDecision as Defined
    from agentic_erp_assistant.state.tool_outcome import ToolOutcome as DefinedOutcome
    from agentic_erp_assistant.tools import models

    assert models.ToolOutcome is DefinedOutcome
    assert models.ApprovalDecision is Defined
