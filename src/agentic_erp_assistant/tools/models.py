"""The whole shape of a tool call: what is asked, what comes back, what is kept.

Read this file and you know the execution envelope. A call arrives as a
:class:`ToolRequest`, comes back as a
:class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome`, and -- when it
changed something -- leaves an :class:`AuditRow` behind. Those three, plus the
two exception types below, are all of it. There is no framework in between.

Most of it is re-exported rather than redefined
-----------------------------------------------

``ToolRequest``, ``ToolOutcome`` and ``ApprovalDecision`` are imported here so
a reader gets the full picture in one file, but they are defined in
:mod:`agentic_erp_assistant.state`. The first two are the two halves of the
gateway port's signature, so the runtime has to be able to name them without
importing this package; the third is carried in the turn's state long after the
tool layer is done with it. A second definition would
be a second thing to keep in step, and the copy that drifts is always the one
nobody is reading. ``ARGUMENTS_SUMMARY_MAX_CHARS`` -- and the renderer it
bounds, ``summarize_tool_call`` -- live there for the same reason: both
``ToolRequest`` and ``ToolOutcome`` need the cap, and neither may import this
package to get it.

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

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_erp_assistant.state.agent_state import ApprovalDecision
from agentic_erp_assistant.state.tool_outcome import ToolOutcome, ToolStatus
from agentic_erp_assistant.state.tool_request import (
    ARGUMENTS_SUMMARY_MAX_CHARS,
    ToolRequest,
)

__all__ = [
    "ApprovalDecision",
    "ARGUMENTS_SUMMARY_MAX_CHARS",
    "AuditRow",
    "ExecutionContext",
    "ToolError",
    "ToolOutcome",
    "ToolRequest",
    "ToolStatus",
    "TransientToolError",
]


@dataclass(frozen=True)
class ExecutionContext:
    """What a handler is told about the call it is running, beyond the
    already-validated arguments.

    A handler reads no permission and makes no policy decision -- that is the
    gateway's job, in one ordered place -- but it still has to know *whose*
    call this is and *which project* to read through
    (:meth:`~agentic_erp_assistant.erp.mock.MockErp.for_project`). Carrying
    that as a second parameter rather than folding it into the arguments model
    keeps "what the model supplied" and "what the gateway already verified"
    visibly separate: an argument the model invented is rejected by the
    argument model's own schema, and nothing here can be spoofed the same way.
    """

    trace_id: str
    actor: str
    project_code: str


class _Envelope(BaseModel):
    """Frozen, closed configuration shared by the models in this file.

    Frozen because both of these are read after a decision was made on them --
    an approver who read "move SPR-14 to done" must be approving the call that
    actually runs, and an audit row must still say what it said when it was
    written. ``extra="forbid"`` because an unmodelled field on a tool call is an
    execution input the type system never saw.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


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

    trace_id: str = Field(min_length=1)
    """The run this call was part of.

    The join back to the trace: without it, an audit reader can see that a
    write happened but not which sequence of decisions produced it, and
    "who changed what" without "as part of which run" is half an answer. The
    request carried it, so the row gets it from the request rather than from
    a second parameter a caller could forget to keep in step.
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
