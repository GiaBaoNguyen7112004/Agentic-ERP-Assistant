"""The whole shape of a tool call: what is asked, what comes back, what is kept.

Read this file and you know the execution envelope. A call arrives as a
:class:`ToolRequest`, comes back as a
:class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome`, and -- when it
changed something -- leaves an :class:`AuditRow` behind. Those three, plus the
two exception types below, are all of it. There is no framework in between.

Two of the three are re-exported rather than redefined
------------------------------------------------------

``ToolOutcome`` and ``ApprovalDecision`` are imported here so a reader gets the
full picture in one file, but they are defined in
:mod:`agentic_erp_assistant.state`, because both are carried in the turn's
state long after the tool layer is done with them. A second definition would
be a second thing to keep in step, and the copy that drifts is always the one
nobody is reading.

``ValidationError`` is deliberately not defined here either. Arguments are
checked by :meth:`~agentic_erp_assistant.llm.tools.ToolSpec.validate_arguments`
against the tool's own pydantic model, which raises pydantic's
``ValidationError``. Declaring a second exception for the same failure would
mean every caller has to catch two things to catch one problem.

Exceptions live inside the boundary; statuses cross it
------------------------------------------------------

:class:`TransientToolError` is raised by a handler and caught by the gateway,
which spends the retry budget and then reports a
:class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome` with the status
``"transient_failure"``. Nothing raises past the gateway: a turn has to record
that the attempt happened, and an exception unwinding through the graph would
leave that record unwritten. The rule is the same one
:mod:`agentic_erp_assistant.llm.ports` follows for provider failures.
"""

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from agentic_erp_assistant.state.agent_state import ApprovalDecision
from agentic_erp_assistant.state.tool_outcome import ToolOutcome, ToolStatus

__all__ = [
    "ApprovalDecision",
    "ARGUMENTS_SUMMARY_MAX_CHARS",
    "AuditRow",
    "ToolError",
    "ToolOutcome",
    "ToolRequest",
    "ToolStatus",
    "TransientToolError",
]


ARGUMENTS_SUMMARY_MAX_CHARS = 200
"""How long the human-readable description of a call may be.

A cap, because this is the line an approver reads and an auditor reads back.
The moment it can hold a full argument payload, it will, and then credentials
and customer data are rendered to a screen and written to a record that outlives
the run. The same decision ``ApprovalRequest.arguments_summary`` already makes.
"""


class _Envelope(BaseModel):
    """Frozen, closed configuration shared by the models in this file.

    Frozen because both of these are read after a decision was made on them --
    an approver who read "move SPR-14 to done" must be approving the call that
    actually runs, and an audit row must still say what it said when it was
    written. ``extra="forbid"`` because an unmodelled field on a tool call is an
    execution input the type system never saw.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolRequest(_Envelope):
    """One tool call, with everything needed to decide whether it may run.

    Not just name and arguments. Who is asking, what they are entitled to, and
    where the call stands with an approver all travel with the request, because
    each is consulted by a different check and none of them can be looked up
    later from the arguments alone.
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


class AuditRow(_Envelope):
    """One line of the record of what was actually done, and on whose say-so.

    Distinct from :class:`~agentic_erp_assistant.state.events.TraceEvent`,
    which is a step in one graph run. This is a fact about a tool call that
    outlives the run: it answers "who changed what, when, and who approved it",
    it is written per call rather than per node, and it is read by someone who
    was not debugging anything.

    Which is also why it carries a clock and a ``TraceEvent`` does not. Order
    within a run is given by position in the log; an audit reader is comparing
    rows *across* runs, where position says nothing.
    """

    occurred_at: datetime
    """When the call happened.

    Supplied by the caller, with no ``default_factory`` reading the clock. A
    model that stamps itself cannot be constructed in a test without the
    expected value changing every run, and the caller is the only one that
    knows whether it means the start or the end of the attempt.
    """

    actor: str = Field(min_length=1)
    """Who it was done for."""

    tool_name: str = Field(min_length=1)
    """What was called."""

    arguments_summary: str = Field(min_length=1, max_length=ARGUMENTS_SUMMARY_MAX_CHARS)
    """What it would do, in a line -- e.g. ``"move SPR-14 to done"``.

    A summary, never the argument payload. See
    :data:`ARGUMENTS_SUMMARY_MAX_CHARS`.
    """

    approval: ApprovalDecision
    """What the approver said. No default: an audit row exists to record this,
    so leaving it to fall back to ``"not_required"`` would let the one fact the
    row is for go unstated."""

    status: ToolStatus
    """How the call ended. The same vocabulary the outcome reports, so a row
    and the outcome it came from cannot disagree about what happened."""

    source_ids: tuple[str, ...] = ()
    """What was read or changed, copied from the outcome."""

    @model_validator(mode="after")
    def _the_row_must_not_contradict_itself(self) -> "AuditRow":
        if self.approval == "denied" and self.status == "ok":
            raise ValueError(
                "status: a call whose approval was denied cannot have "
                "succeeded; one of the two fields is wrong, and an audit row "
                "that records both is worse than none"
            )
        if any(not source.strip() for source in self.source_ids):
            raise ValueError("source_ids: identifiers must not be blank")
        return self


class ToolError(Exception):
    """Base for failures raised inside the tool boundary.

    Present so a caller that only wants "the tool call failed" has one thing to
    catch. Nothing raised here reaches the graph: the gateway turns it into a
    :class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome`.
    """


class TransientToolError(ToolError):
    """The call failed in a way a later identical attempt may survive.

    Timeout, connection reset, 429, 5xx from an ERP backend. The only failure
    the retry budget is allowed to spend itself on -- which is why it is a
    named type rather than a flag on a generic error: an ``except`` clause is
    what decides what gets retried, and it cannot read a flag it never caught.
    """
