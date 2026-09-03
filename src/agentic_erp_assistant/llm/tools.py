"""What a tool is, before any provider is involved.

A provider's function-calling API wants a name, a description and a JSON Schema
for the arguments. That is not enough to run a tool safely, so the declaration
kept here carries one more thing the wire format has no slot for: whether the
call *mutates* anything. The router reads that flag to decide whether a call has
to stop and wait for a human, and the flag has to live with the declaration --
a separate list of "the dangerous ones" maintained somewhere else is a list that
will be wrong exactly once.

So a :class:`ToolSpec` is provider-neutral and slightly richer than any single
vendor's schema, and the adapter renders it down to the vendor's shape. The
alternative -- writing OpenAI's ``{"type": "function", ...}`` dict directly here
-- would put the wire format in the core, leave nowhere for ``mutating`` to
live, and make the tool registry unusable by a second provider or by a direct
in-process execution path.

The argument schema is emitted from a pydantic model rather than hand-written,
for the reason ``schemas.py`` gives: the shape asked for and the shape accepted
have to be one artifact. Here it buys something extra -- OpenAI's strict
function calling demands that every property be required and that
``additionalProperties`` be false, and :class:`ToolSpec` refuses to be
constructed with a schema that is not strict-compatible. A tool that cannot be
sent strictly fails at import, not in production, where the failure mode is a
model inventing an argument nobody declared.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DEFAULT_TOOLS",
    "GET_PROJECT_STATUS_TOOL",
    "ProjectStatusArguments",
    "StrictArguments",
    "ToolCallResult",
    "ToolSpec",
]


class StrictArguments(BaseModel):
    """Base for every tool's argument model.

    ``extra="forbid"`` renders as ``additionalProperties: false``, which is both
    half of what strict function calling requires and the check that rejects an
    argument the model invented. ``frozen`` because arguments that have been
    validated and then handed to a router must not be edited on the way -- an
    approver who read "move SPR-14 to done" has to be approving the call that
    actually runs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectStatusArguments(StrictArguments):
    """Arguments for ``get_project_status``.

    One required field, no defaults. That is deliberate rather than minimal:
    strict function calling requires every property to appear in ``required``,
    so an optional argument would have to be modelled as required-and-nullable.
    Worth knowing when the second tool arrives -- and worth contrasting with
    ``GroundedAnswer``, whose defaults are why ``response_format`` goes out
    non-strict.
    """

    milestone_id: str = Field(
        min_length=1,
        description="The milestone to report on, e.g. 'M2'.",
    )


def _object_schemas(schema: Mapping[str, Any]):
    """Yield the root schema and every definition under it.

    Nested models land in ``$defs``, and strict mode judges each of them
    separately, so a check that only looked at the root would pass a tool whose
    nested object silently accepts anything.
    """
    yield schema
    for definition in (schema.get("$defs") or {}).values():
        if isinstance(definition, dict):
            yield definition


def _require_strict_compatible(name: str, schema: Mapping[str, Any]) -> None:
    """Reject a schema OpenAI's strict mode would refuse, at construction time."""
    for index, part in enumerate(_object_schemas(schema)):
        if part.get("type") != "object":
            continue

        where = "arguments" if index == 0 else part.get("title", "a nested object")
        if part.get("additionalProperties") is not False:
            raise ValueError(
                f"tool {name!r}: {where} must set additionalProperties=false "
                f"(inherit from StrictArguments)"
            )

        properties = sorted(part.get("properties") or {})
        required = sorted(part.get("required") or [])
        if properties != required:
            missing = sorted(set(properties) - set(required))
            raise ValueError(
                f"tool {name!r}: strict function calling requires every "
                f"property to be required; {where} leaves {missing} optional. "
                f"Model an optional argument as required-and-nullable instead."
            )


@dataclass(frozen=True)
class ToolSpec:
    """One tool the model may be offered, declared once for every consumer.

    The adapter renders this to a vendor's function schema; the router reads
    ``mutating`` to decide whether the call needs approval; the executor uses
    ``arguments`` to validate what came back. All three read the same object,
    which is the point.
    """

    name: str
    """The identifier the model calls and the trace records."""

    description: str
    """What the tool does, written for the model. This is the only thing
    steering it toward or away from the tool, so it is prompt text, not a
    docstring afterthought."""

    arguments: type[BaseModel]
    """The argument contract. Must be strict-compatible; see the module doc."""

    mutating: bool
    """Whether calling this changes ERP data.

    The flag the wire format has no room for, and the reason this class exists.
    A true value means the router must route the call through an approval and
    record the decision before anything executes.
    """

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a tool needs a name")
        if not self.description.strip():
            raise ValueError(f"tool {self.name!r} needs a description")
        _require_strict_compatible(self.name, self.schema)

    @property
    def schema(self) -> dict[str, Any]:
        """The JSON Schema for the arguments, emitted from the model."""
        return self.arguments.model_json_schema()

    def validate_arguments(self, raw: Mapping[str, Any]) -> BaseModel:
        """Validate parsed arguments against the declaration.

        Deliberately *not* called by the adapter. An adapter that validated
        would have to know the tool registry, and a failure there would be
        indistinguishable from a transport problem. Here the failure is a
        ``ValidationError``: the contract was broken, resampling will not fix
        it, and :func:`~agentic_erp_assistant.llm.retry.retry_with_backoff`
        will not retry it.

        Raises:
            ValidationError: The arguments do not satisfy the declaration --
                an unknown key, a missing one, or the wrong type.
        """
        return self.arguments.model_validate(dict(raw))


GET_PROJECT_STATUS_TOOL = ToolSpec(
    name="get_project_status",
    description=(
        "Look up the delivery status of one milestone: schedule, budget "
        "consumed, and open risks. Read-only. Use it whenever the user asks "
        "about a specific milestone by identifier."
    ),
    arguments=ProjectStatusArguments,
    mutating=False,
)
"""The first real tool. Read-only, so it needs no approval -- which is a fact
recorded in the declaration rather than an assumption made at the call site."""


DEFAULT_TOOLS: tuple[ToolSpec, ...] = (GET_PROJECT_STATUS_TOOL,)
"""What the model is offered when a caller does not say otherwise.

A tuple, and the offering is data: adding a tool is an edit here, not a new
branch in the adapter.
"""


class ToolCallResult(BaseModel):
    """The outcome of one tool-enabled turn: a call, or an answer, never both.

    The model either decided to use a tool or decided to reply directly, and a
    result object that can hold both leaves every caller to invent its own rule
    for which one wins. The validator below makes the ambiguous states
    unconstructable, so the rule is decided once, here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str | None = None
    """The tool the model chose, or ``None`` when it answered directly."""

    arguments: dict[str, Any] | None = None
    """The call's arguments, already parsed out of the provider's JSON string.

    Parsed but *not* validated -- see :meth:`ToolSpec.validate_arguments` for
    where that belongs and why it is not here.
    """

    tool_call_id: str | None = None
    """The provider's id for this call.

    Carried because a real tool round trip has to quote it back in the
    following ``tool`` message. Dropping it here would mean the execution layer
    could run the tool and then have no way to report the result.
    """

    content: str | None = None
    """The direct reply, when no tool applied."""

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> "ToolCallResult":
        called = self.tool_name is not None
        if called and (self.arguments is None):
            raise ValueError("arguments: a tool call must carry parsed arguments")
        if not called and self.arguments is not None:
            raise ValueError("arguments: present without a tool_name")
        if not called and self.tool_call_id is not None:
            raise ValueError("tool_call_id: present without a tool_name")
        if called == (self.content is not None):
            raise ValueError(
                "a result is either a tool call or direct content, never both "
                "and never neither"
            )
        return self

    @classmethod
    def from_tool_call(
        cls,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        tool_call_id: str | None = None,
    ) -> "ToolCallResult":
        """Build the tool-call form."""
        return cls(
            tool_name=tool_name,
            arguments=dict(arguments),
            tool_call_id=tool_call_id,
        )

    @classmethod
    def from_content(cls, content: str) -> "ToolCallResult":
        """Build the direct-answer form."""
        return cls(content=content)
