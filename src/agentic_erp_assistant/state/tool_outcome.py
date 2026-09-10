"""What running one tool produced, in the shape the turn carries afterwards.

A tool call is the one place this assistant can change something outside
itself, so its result is not a return value that gets read once and dropped. It
is carried in the turn's state, shown to the model, written into the trace and,
for a write, into an audit record. This module is that object.

It lives in ``state`` for the reason
:class:`~agentic_erp_assistant.state.evidence.EvidenceSnippet` does: it crosses
a node boundary, so both the port that declares it
(:class:`~agentic_erp_assistant.engine.ports.ToolGatewayPort`) and the tool
layer that produces it need to name the type. Defining it in either one would
make the other import a layer it has no business depending on -- and would undo
most of what declaring the gateway as a Protocol bought.

Failure is data here, exceptions live inside the boundary
---------------------------------------------------------

A tool that was denied, handed bad arguments or timed out does not raise
through the graph. It comes back as one of these, with a :data:`ToolStatus`
saying which, because the turn has to record the attempt and then route on it
-- and an exception unwinding past the node would leave the state describing
that attempt unwritten. Handlers inside the tool layer still raise
(:class:`~agentic_erp_assistant.tools.models.TransientToolError` among them);
the gateway is where a raise becomes a status.

Why the status is not a boolean
-------------------------------

``ok: bool`` plus an error string cannot tell "a human refused this" from "the
provider fell over" from "the model invented an argument". Those three want
different next moves -- a refusal ends the turn, a timeout may be retried, bad
arguments are a contract failure that resampling will not fix -- and a caller
reading a boolean has to parse the error text to find out which it has. A
closed set makes the reader match a branch.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["ToolOutcome", "ToolStatus"]


ToolStatus = Literal[
    "ok",                  # the tool did what it was asked to do
    "invalid_arguments",   # the call did not satisfy the tool's declaration
    "approval_required",   # nobody has been asked yet; nothing ran
    "denied",              # policy or an approver refused it; nothing ran
    "rate_limited",        # the actor's budget for this tool is spent; nothing ran
    "transient_failure",   # it may work later: timeout, 429, connection reset
    "failed",              # it ran and could not complete, and will not later
]
"""How a tool call ended.

Closed, and every member is a distinct next move.

``denied`` and ``failed`` are kept apart for the reason ``refuse`` and ``fail``
are kept apart in
:data:`~agentic_erp_assistant.reasoning.decision.DecisionRoute`: one is the
safety layer working, the other is the system not working, and a reviewer
counting refusals must not be counting outages.

``approval_required`` and ``denied`` are kept apart for a sharper reason.
"Nobody has been asked yet" and "a human said no" look alike -- nothing ran
either way -- and lead to opposite moves: the first goes to
``request_approval`` and may still succeed, the second ends the turn. Under one
label the graph would either re-ask a human who already refused, or abandon a
call that was never put to anyone.

``denied`` and ``rate_limited`` are kept apart on a third axis: permanence. A
missing scope will still be missing on the next attempt, so ``denied`` ends the
call for good; a spent budget refills, so ``rate_limited`` carries a
:attr:`ToolOutcome.retry_after_seconds` and the call is worth making again
later. Collapsed into one status the graph would either abandon a call that
would succeed in forty seconds, or re-offer one that will be refused forever.

``transient_failure`` is the only status a retry engine is allowed to act on.
``rate_limited`` is deliberately not one of them: the wait is reported upward
for the caller to schedule, because retrying inside the same turn would spend
the attempt budget proving a limit that is, by construction, still in force.
"""


class ToolOutcome(BaseModel):
    """The result of one tool call, success or failure alike.

    Frozen and ``extra="forbid"``: what the trace says the tool returned has to
    be what the model was then shown, and a field invented at one call site is
    a column no report knows how to read.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1)
    """Which tool ran. Repeated here rather than inferred from surrounding
    context, so a trace or audit entry stands on its own."""

    status: ToolStatus
    """How it ended. See :data:`ToolStatus`."""

    summary: str = ""
    """What to show the model, and what a person reads in the trace.

    One field, not a payload plus a separate human summary: two would drift the
    first time someone edited one of them, and the trace would then describe
    something other than what the model saw.

    Deliberately not length-capped. What fits in the prompt is the context
    builder's decision, made against a measured budget; a second, smaller limit
    here would truncate silently before that budget ever saw the text.
    """

    source_ids: tuple[str, ...] = ()
    """What the answer rests on, or what the call touched.

    Required on success. The obvious reading of the requirement is that a read
    must cite its sources; this is stricter on purpose and asks the same of a
    write, which names the record it changed. A successful call that cannot say
    what data it involved is one nobody can audit afterwards, and "it was only
    a write" is exactly the case where that matters most.

    A tuple, not a list: a frozen model holding a mutable sequence is not
    frozen.
    """

    error: str | None = None
    """Why it did not succeed, in one line, for a person reading the trace."""

    attempts: int = Field(default=1, ge=1)
    """How many times this call was actually tried.

    At least one -- an outcome exists because something was attempted. The
    retry engine owns the budget; this records what it spent, so a trace shows
    a call that succeeded on the third try as different from one that
    succeeded outright.
    """

    retry_after_seconds: float | None = Field(default=None, ge=0.0)
    """How long the caller was told to wait, when it was told anything.

    Constrained rather than merely optional -- it is only meaningful on the two
    statuses that describe a call worth making again: ``transient_failure``,
    where the backend may recover, and ``rate_limited``, where a budget refills.
    On any other status it would be a number no branch could act on.

    Required on ``rate_limited``, and only there. A transient failure may
    genuinely not know when to come back; a rate limiter always does, because it
    is the thing holding the window, and a refusal that cannot say when to
    retry leaves the caller guessing at the one fact the refusal was for.
    """

    @model_validator(mode="after")
    def _an_outcome_must_be_readable_without_guessing(self) -> "ToolOutcome":
        succeeded = self.status == "ok"

        if succeeded and self.error is not None:
            raise ValueError(
                "error: present on a successful call -- a reader cannot tell "
                "whether the call worked"
            )
        if not succeeded and not (self.error or "").strip():
            raise ValueError(
                f"error: status {self.status!r} must say why, or the trace "
                f"records that something went wrong and nothing about what"
            )

        if succeeded and not self.summary.strip():
            raise ValueError(
                "summary: a successful call must report what it found or did"
            )
        if succeeded and not self.source_ids:
            raise ValueError(
                "source_ids: a successful call must name what it read or "
                "changed, or its result cannot be cited or audited"
            )
        if any(not source.strip() for source in self.source_ids):
            raise ValueError("source_ids: identifiers must not be blank")

        waitable = self.status in ("transient_failure", "rate_limited")
        if self.retry_after_seconds is not None and not waitable:
            raise ValueError(
                f"retry_after_seconds: meaningless on status {self.status!r} -- "
                f"only a call worth repeating is worth waiting to repeat"
            )
        if self.status == "rate_limited" and self.retry_after_seconds is None:
            raise ValueError(
                "retry_after_seconds: a rate-limited call must say when to try "
                "again -- the limiter holds the window, and no one else can "
                "recover the answer"
            )

        return self
