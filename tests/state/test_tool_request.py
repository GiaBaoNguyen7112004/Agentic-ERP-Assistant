"""Nothing may run on facts the request never carried."""

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.llm.tools import GET_PROJECT_STATUS_TOOL
from agentic_erp_assistant.state.tool_request import (
    ARGUMENTS_SUMMARY_MAX_CHARS,
    ToolRequest,
    summarize_tool_call,
)


def request(**overrides: object) -> ToolRequest:
    fields: dict[str, object] = {
        "trace_id": "run-1",
        "tool_name": "close_milestone",
        "arguments": {"milestone_id": "M2"},
        "actor": "bao",
        "project_code": "atlas",
        "scopes": frozenset({"erp:write"}),
    }
    fields.update(overrides)
    return ToolRequest(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Every request says who is asking and what they are entitled to
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["trace_id", "tool_name", "arguments", "actor", "project_code", "scopes"]
)
def test_a_request_cannot_be_built_without_it(field: str) -> None:
    """None of the six has a default. A call missing any of them is one no
    check downstream could make a decision about."""
    fields = {
        "trace_id": "run-1",
        "tool_name": "t",
        "arguments": {},
        "actor": "bao",
        "project_code": "atlas",
        "scopes": frozenset(),
    }
    del fields[field]

    with pytest.raises(ValidationError, match=field):
        ToolRequest(**fields)  # type: ignore[arg-type]


def test_a_request_bound_to_no_project_is_rejected() -> None:
    """The same argument scopes already makes: a call unbindable to a project
    is one the gateway's project check has nothing to compare against."""
    with pytest.raises(ValidationError, match="project_code"):
        request(project_code="")


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


def test_the_tool_layer_re_exports_the_same_class() -> None:
    """One definition, readable from the file that describes the envelope."""
    from agentic_erp_assistant.tools import models

    assert models.ToolRequest is ToolRequest


# --------------------------------------------------------------------------
# summarize_tool_call: the one rendering the gateway, the audit row, and
# the engine's repeated-call guard (ADR 0019) all read
# --------------------------------------------------------------------------


def test_summarize_tool_call_includes_values_not_just_keys() -> None:
    line = summarize_tool_call("create_risk", {"project_id": "orion", "severity": "high"})

    assert line == "create_risk(project_id=orion, severity=high)"


def test_summarize_tool_call_elides_a_long_value() -> None:
    line = summarize_tool_call("create_risk", {"title": "x" * 100})

    assert "x" * 100 not in line
    assert line.endswith("…)")


def test_summarize_tool_call_caps_the_whole_line() -> None:
    arguments = {f"key{i}": "value" * 10 for i in range(20)}

    line = summarize_tool_call("create_risk", arguments)

    assert len(line) <= ARGUMENTS_SUMMARY_MAX_CHARS


def test_summarize_tool_call_with_no_arguments_still_names_the_tool() -> None:
    assert summarize_tool_call("list_risks", {}) == "list_risks()"


def test_the_tool_layer_re_exports_the_same_cap() -> None:
    from agentic_erp_assistant.tools import models

    assert models.ARGUMENTS_SUMMARY_MAX_CHARS is ARGUMENTS_SUMMARY_MAX_CHARS
