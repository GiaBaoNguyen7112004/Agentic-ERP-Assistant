"""Every memory decision, written down -- including the ones that stored nothing.

An audit that only records writes answers "what does the assistant believe?" and
leaves the more interesting question unanswered: *what did it refuse to believe,
and why?* Those are the rows that show the policy working. A run whose model
proposed five memories and stored one produces five rows here, and the four
rejections each name the rule that refused them.

So this is not a log of state changes. It is the record of a decision procedure,
and it exists to be read by somebody who was not debugging anything -- the same
distinction :class:`~agentic_erp_assistant.tools.models.AuditRow` draws against
:class:`~agentic_erp_assistant.state.events.TraceEvent`.

The five identifiers, and why each one is required
--------------------------------------------------

``trace_id`` joins the decision to the run that produced it, so "why did it
remember that?" leads to the conversation. ``session_id`` and ``project_code``
are the boundaries the decision was made inside, so a row can be read without
first working out whose memory it was about. ``memory_id`` names the thing --
present even on a rejection, because
:func:`~agentic_erp_assistant.memory.models.memory_id` derives an id from the
content before any verdict is reached. And ``actor`` says who it was for, because
a preference belongs to a person.

None of them is optional. A row missing any one is a decision nobody can place,
which is the same as no row at all.

The sink must not raise
-----------------------

:class:`MemoryAuditSink` keeps the contract
:class:`~agentic_erp_assistant.tools.audit.AuditSink` keeps, for a related but
distinct reason. There the argument is that an audit failure must not undo an
approved write. Here it is that memory consolidation runs *after* a turn has
already answered the user: an exception at this point would turn "we could not
write down that we declined to remember something" into a failed request, which
is absurd. The failure is logged, loudly, and the turn stands.
"""

import logging
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_erp_assistant.memory.models import (
    REASON_MAX_CHARS,
    MemoryDecisionKind,
    MemoryKind,
    RejectionReason,
)

__all__ = [
    "InMemoryMemoryAudit",
    "MemoryAuditRow",
    "MemoryAuditSink",
    "STATEMENT_SUMMARY_MAX_CHARS",
    "summarize_statement",
]

logger = logging.getLogger(__name__)


STATEMENT_SUMMARY_MAX_CHARS = 200
"""How much of a would-be memory an audit row may quote.

A cap for the reason
:data:`~agentic_erp_assistant.tools.models.ARGUMENTS_SUMMARY_MAX_CHARS` is one,
and the case here is sharper. The rows that matter most are the *rejections*, and
a rejection is often a rejection because the text carried something that should
not be stored. An audit row that quoted a refused statement in full would
faithfully persist the credential the policy just declined to persist -- in a
table that is exported as evidence.

The cap does not make that safe on its own; it bounds it. The caller passes a
summary, and :func:`summarize_statement` is what most callers use.
"""

_TRUNCATED = "..."


def summarize_statement(statement: str) -> str:
    """One line of a statement, whitespace collapsed and length bounded.

    Collapsing is load-bearing, not cosmetic: an audit row rendered into a report
    one line at a time would let a statement containing a newline appear as two
    rows, and manufacture a decision that was never made -- the same reason
    :func:`~agentic_erp_assistant.llm.prompts._render_evidence` collapses a
    snippet.
    """
    text = " ".join(statement.split())
    if len(text) <= STATEMENT_SUMMARY_MAX_CHARS:
        return text
    return text[: STATEMENT_SUMMARY_MAX_CHARS - len(_TRUNCATED)] + _TRUNCATED


class MemoryAuditRow(BaseModel):
    """One decision about one piece of would-be memory.

    Frozen and ``extra="forbid"``, for the reason
    :class:`~agentic_erp_assistant.tools.models.AuditRow` is: a record that can
    be edited after the fact is worse evidence than no record, and a field
    invented at one call site is a column no report knows how to read.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    occurred_at: datetime
    """When the decision was made.

    Supplied by the caller, with no ``default_factory`` reading the clock -- a
    model that stamps itself cannot be asserted on in a test, and only the caller
    knows whether it means the start or the end of consolidation.
    """

    trace_id: str = Field(min_length=1)
    """The run the decision was made in. The join back to the trace."""

    session_id: str = Field(min_length=1)
    """The conversation it was made inside."""

    project_code: str = Field(min_length=1)
    """The tenant boundary it was made inside."""

    actor: str = Field(min_length=1)
    """Who it was for."""

    memory_id: str = Field(min_length=1)
    """What was decided about.

    Required even on a rejection, which is the whole reason
    :func:`~agentic_erp_assistant.memory.models.memory_id` derives an id from
    content rather than generating one at write time. A refusal row carrying a
    null here would be a decision about nothing.
    """

    kind: MemoryKind
    """Which of the five kinds was proposed. Kept so a report can group
    refusals by kind without parsing the statement."""

    decision: MemoryDecisionKind
    """What happened. All four members occur here: ``write``, ``update`` and
    ``reject`` are verdicts on a candidate, and ``forget`` is recorded against
    each record an ``update`` retired."""

    rejection: RejectionReason | None = None
    """Which rule refused, on a rejection. ``None`` otherwise -- and the
    validator below keeps the two in step, because a row saying ``write`` with a
    rejection reason beside it is evidence that contradicts itself."""

    reason: str = Field(default="", max_length=REASON_MAX_CHARS)
    """The decision's own one-line explanation, carried through unchanged."""

    statement_summary: str = Field(
        min_length=1, max_length=STATEMENT_SUMMARY_MAX_CHARS
    )
    """What was proposed, in a line. See :data:`STATEMENT_SUMMARY_MAX_CHARS`."""

    @model_validator(mode="after")
    def _the_row_must_not_contradict_itself(self) -> "MemoryAuditRow":
        if (self.decision == "reject") != (self.rejection is not None):
            raise ValueError(
                f"rejection: decision {self.decision!r} with rejection "
                f"{self.rejection!r}; a refusal must name the rule that refused "
                f"it, and nothing else may carry one"
            )
        for name in (
            "trace_id",
            "session_id",
            "project_code",
            "actor",
            "memory_id",
            "statement_summary",
        ):
            if not getattr(self, name).strip():
                raise ValueError(
                    f"{name}: must not be blank; a row missing one of the "
                    f"identifiers is a decision nobody can place"
                )
        return self


@runtime_checkable
class MemoryAuditSink(Protocol):
    """Somewhere a memory decision can be written and not lost."""

    def record(self, row: MemoryAuditRow) -> None:
        """Persist one row.

        Must not raise, for any row and for any reason. This is called after the
        turn has already answered the user, so an exception here would turn "we
        could not write down that we declined to remember something" into a
        failed request. See the module docstring.
        """
        ...


class InMemoryMemoryAudit:
    """A list, and the default.

    The same trade
    :class:`~agentic_erp_assistant.tools.audit.InMemoryAuditLog` makes: rows
    disappear with the process, which is not an audit trail, but a consolidator
    with no sink at all would silently decide nothing -- and this at least holds
    the rows where a test, or a developer, can see them.
    """

    def __init__(self) -> None:
        self.rows: list[MemoryAuditRow] = []

    def record(self, row: MemoryAuditRow) -> None:
        self.rows.append(row)
