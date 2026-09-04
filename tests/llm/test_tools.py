"""The tool declaration is a contract with three readers, so all three are tested.

A ``ToolSpec`` is read by the adapter (which renders it to a vendor schema), by
the router (which reads ``mutating`` to decide whether a human has to approve),
and by the executor (which validates what came back). Nothing here touches a
provider or a network -- the wire rendering is asserted where the request body
is, in the adapter's own tests.

The load-bearing test in this file is the strict-compatibility one: it runs over
every tool in ``DEFAULT_TOOLS``, so the second tool cannot be added with an
optional argument and discover the problem in production, where the symptom is a
model inventing an argument nobody declared.
"""

import pytest
from pydantic import Field, ValidationError

from agentic_erp_assistant.llm.tools import (
    DEFAULT_TOOLS,
    GET_PROJECT_STATUS_TOOL,
    ProjectStatusArguments,
    StrictArguments,
    ToolCallResult,
    ToolSpec,
)

# --------------------------------------------------------------------------
# The declared tool
# --------------------------------------------------------------------------


def test_the_project_status_tool_is_declared_read_only() -> None:
    """The flag the wire format has no room for, and the reason ToolSpec exists."""
    assert GET_PROJECT_STATUS_TOOL.name == "get_project_status"
    assert GET_PROJECT_STATUS_TOOL.mutating is False
    assert GET_PROJECT_STATUS_TOOL.arguments is ProjectStatusArguments


def test_the_description_is_written_for_the_model() -> None:
    """It is prompt text: it is the only thing steering the model to the tool."""
    description = GET_PROJECT_STATUS_TOOL.description
    assert len(description) > 40
    assert "milestone" in description.lower()


def test_the_schema_declares_the_milestone_argument() -> None:
    schema = GET_PROJECT_STATUS_TOOL.schema

    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["milestone_id"]
    assert schema["properties"]["milestone_id"]["type"] == "string"
    assert schema["required"] == ["milestone_id"]


def test_the_schema_is_emitted_not_hand_written() -> None:
    """One artifact: the shape asked for is the shape validated against."""
    assert GET_PROJECT_STATUS_TOOL.schema == ProjectStatusArguments.model_json_schema()


@pytest.mark.parametrize("spec", DEFAULT_TOOLS, ids=lambda spec: spec.name)
def test_every_offered_tool_is_strict_compatible(spec: ToolSpec) -> None:
    """Strict function calling requires exactly this, for every offered tool."""
    schema = spec.schema

    assert schema.get("additionalProperties") is False
    assert sorted(schema.get("properties") or {}) == sorted(schema.get("required") or [])


# --------------------------------------------------------------------------
# The strictness check, which runs at construction
# --------------------------------------------------------------------------


def test_a_tool_with_an_optional_argument_cannot_be_constructed() -> None:
    """Caught at import rather than by a 400 in production."""

    class Loose(StrictArguments):
        needed: str
        optional: str = "sometimes"

    with pytest.raises(ValueError, match="required"):
        ToolSpec(
            name="loose",
            description="A tool with an argument the model may omit.",
            arguments=Loose,
            mutating=False,
        )


def test_a_tool_that_accepts_undeclared_arguments_cannot_be_constructed() -> None:
    from pydantic import BaseModel

    class Open(BaseModel):  # no extra="forbid"
        needed: str

    with pytest.raises(ValueError, match="additionalProperties"):
        ToolSpec(
            name="open",
            description="A tool that would accept an invented argument.",
            arguments=Open,
            mutating=False,
        )


def test_a_nested_object_is_checked_too() -> None:
    """A root-only check would pass a tool whose nested object accepts anything."""
    from pydantic import BaseModel

    class LooseInner(BaseModel):
        field: str

    class Outer(StrictArguments):
        inner: LooseInner

    with pytest.raises(ValueError, match="additionalProperties"):
        ToolSpec(
            name="nested",
            description="A tool whose nested object is not strict.",
            arguments=Outer,
            mutating=False,
        )


def test_a_nameless_or_undescribed_tool_is_rejected() -> None:
    for name, description in (("", "fine"), ("fine", "  ")):
        with pytest.raises(ValueError):
            ToolSpec(
                name=name,
                description=description,
                arguments=ProjectStatusArguments,
                mutating=False,
            )


def test_a_spec_cannot_be_edited_after_declaration() -> None:
    with pytest.raises(Exception):
        GET_PROJECT_STATUS_TOOL.mutating = True  # type: ignore[misc]


# --------------------------------------------------------------------------
# Validating what came back
# --------------------------------------------------------------------------


def test_valid_arguments_validate_into_the_declared_model() -> None:
    arguments = GET_PROJECT_STATUS_TOOL.validate_arguments({"milestone_id": "M2"})

    assert isinstance(arguments, ProjectStatusArguments)
    assert arguments.milestone_id == "M2"


@pytest.mark.parametrize(
    "raw",
    [
        {"milestone_id": "M2", "sneaky": "extra"},
        {},
        {"milestone_id": 12},
        {"milestone_id": ""},
    ],
    ids=["unknown-key", "missing-key", "wrong-type", "empty-string"],
)
def test_bad_arguments_raise_a_validation_error(raw: dict) -> None:
    """Not a TransientProviderError: the contract was broken, and retry.py must
    not spend a budget resampling a call that will fail the same way."""
    with pytest.raises(ValidationError):
        GET_PROJECT_STATUS_TOOL.validate_arguments(raw)


def test_validated_arguments_are_frozen() -> None:
    arguments = GET_PROJECT_STATUS_TOOL.validate_arguments({"milestone_id": "M2"})

    with pytest.raises(ValidationError):
        arguments.milestone_id = "M3"  # type: ignore[misc]


# --------------------------------------------------------------------------
# ToolCallResult: one outcome, never two
# --------------------------------------------------------------------------


def test_a_tool_call_result_carries_the_name_arguments_and_id() -> None:
    result = ToolCallResult.from_tool_call(
        tool_name="get_project_status",
        arguments={"milestone_id": "M2"},
        tool_call_id="call_abc",
    )

    assert result.tool_name == "get_project_status"
    assert result.arguments == {"milestone_id": "M2"}
    assert result.tool_call_id == "call_abc"
    assert result.content is None


def test_a_direct_answer_carries_content_and_no_call() -> None:
    result = ToolCallResult.from_content("No milestone was named.")

    assert result.content == "No milestone was named."
    assert result.tool_name is None
    assert result.arguments is None
    assert result.tool_call_id is None


def test_a_result_cannot_be_both_a_call_and_an_answer() -> None:
    """The acceptance criterion, made unconstructable rather than documented."""
    with pytest.raises(ValidationError):
        ToolCallResult(
            tool_name="get_project_status",
            arguments={"milestone_id": "M2"},
            content="and also this",
        )


def test_a_result_cannot_be_neither() -> None:
    with pytest.raises(ValidationError):
        ToolCallResult()


def test_a_call_without_arguments_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolCallResult(tool_name="get_project_status")


def test_arguments_without_a_call_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolCallResult(arguments={"milestone_id": "M2"}, content="hm")


def test_a_call_id_without_a_call_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolCallResult(tool_call_id="call_abc", content="hm")


def test_a_result_is_frozen_and_forbids_unknown_fields() -> None:
    result = ToolCallResult.from_content("hello")

    with pytest.raises(ValidationError):
        result.content = "changed"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        ToolCallResult(content="hello", finish_reason="stop")


def test_strict_arguments_reject_an_invented_field() -> None:
    class Example(StrictArguments):
        needed: str = Field(min_length=1)

    with pytest.raises(ValidationError):
        Example(needed="ok", invented="nope")
