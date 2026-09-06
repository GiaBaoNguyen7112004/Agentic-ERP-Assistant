"""One tool call, with everything needed to decide whether it may run.

This is the input side of the gateway port, and it lives in ``state`` for the
same reason :class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome` does:
:meth:`~agentic_erp_assistant.engine.ports.ToolGatewayPort.execute` takes one
and returns the other, so both types have to be nameable by the runtime and by
the tool layer without either importing the other. A port that took loose
arguments in and returned a typed object would be asymmetric for no reason, and
the loose half is where a scope quietly stops being passed.

Not just a name and arguments. Who is asking, what they are entitled to, and
where the call stands with an approver all travel with the request, because
each is consulted by a different check and none can be recovered later from the
arguments alone.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from agentic_erp_assistant.state.agent_state import ApprovalDecision

__all__ = ["ToolRequest"]


class ToolRequest(BaseModel):
    """One tool call, and every fact a check downstream will consult.

    Frozen because a request that has been authorized must not be edited on the
    way to the handler -- an approver who read "move SPR-14 to done" has to be
    approving the call that actually runs. ``extra="forbid"`` because an
    unmodelled field on a tool call is an execution input the type system never
    saw.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    """The run this call belongs to.

    Required, with no default, for the same reason
    :attr:`~agentic_erp_assistant.state.agent_state.AgentState.trace_id` is: a
    tool call that could happen without one would be a call whose audit row
    cannot be joined back to the trace that explains it -- "who changed what"
    without "as part of which run" is half the evidence an auditor needs.
    """

    tool_name: str = Field(min_length=1)
    """The tool being called, as the registry knows it."""

    arguments: Mapping[str, Any]
    """The arguments as parsed, before they are checked against the tool.

    Parsed, not validated -- validation is
    :meth:`~agentic_erp_assistant.llm.tools.ToolSpec.validate_arguments`, and it
    happens between here and the handler. That is what keeps an argument the
    model invented from reaching any implementation: the tool's model sets
    ``extra="forbid"``, so an unknown key raises instead of being ignored.

    Stored behind a read-only view, so a request that has been authorized
    cannot have its arguments swapped before it executes.
    """

    actor: str = Field(min_length=1)
    """Who the call is made on behalf of.

    Required, with no default. An anonymous call cannot be audited, and the
    audit row is the artifact the approval rule exists to produce.
    """

    scopes: frozenset[str]
    """What the actor is entitled to do, as granted by whatever authenticated
    them.

    A different axis from approval, and both are needed. Scope is standing
    permission attached to an identity; approval is a decision about one call
    at one moment. Neither substitutes for the other: an actor with the scope
    to close milestones still needs a human to approve closing *this* one, and
    an approval cannot grant an entitlement its holder never had.

    Required rather than defaulted to empty -- a call that forgot to say what
    it was entitled to would otherwise read as a call entitled to nothing, and
    pass or fail for the wrong reason.

    **Nothing reads this yet.** The check belongs to the router, which is not
    written; it is declared now because settling the request contract once is
    cheaper than changing it under callers later. Until that check exists,
    approval is the only gate in front of a write.
    """

    approval: ApprovalDecision = "not_required"
    """Where this call stands with its approver.

    The reason there is no ``approval: X | None`` here: ``"not_required"`` is
    already a member, so the gate matches a branch instead of testing a value
    for truth. See :data:`~agentic_erp_assistant.state.agent_state.ApprovalDecision`.
    """

    @field_validator("arguments")
    @classmethod
    def _freeze(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Take a copy behind a read-only view.

        A frozen model holding a plain dict is not frozen, it merely looks it,
        and the caller who passed the dict in would keep a handle on the
        arguments an approver has already read.
        """
        return MappingProxyType(dict(value))

    @field_serializer("arguments")
    def _unwrap_for_serialization(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Hand a plain dict to the serializer.

        The read-only view is an in-process guard, not part of the wire shape;
        a trace record or a JSON body wants the mapping itself.
        """
        return dict(value)

    @field_validator("scopes")
    @classmethod
    def _scopes_must_name_something(cls, value: frozenset[str]) -> frozenset[str]:
        if any(not scope.strip() for scope in value):
            raise ValueError("must not contain a blank scope")
        return value
