"""One completed run, written down: the summary columns a report reads.

The run store's write unit. Everything a query filters on is a field, and the
whole final state rides inside -- the store's copy of record vs. evidence is
the same rule ADR 0010 states for chunks: one write path, the store holds a
copy of the thing itself.

The summary fields are deliberately redundant with the state, and the
validators below are what keep the redundancy honest: a record whose
``trace_id`` disagreed with its state's would be two runs filed under one
identity, and a record whose ``outcome`` lied about its state would be
evidence a reviewer cannot trust. Cheap to check at construction, expensive
to discover in a report.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_erp_assistant.state.agent_state import AgentState

__all__ = ["RunOutcome", "RunRecord"]


RunOutcome = Literal["terminal", "paused"]
"""How a run ended when the engine gave it back.

Two members, no third: the engine returns exactly these -- a state that
finished, or a state that stopped for a human. ``failed`` is not one because a
failure is a *terminal* outcome; the failure mode lives on the state, where
the report reads it, and a store that invented its own vocabulary for the same
fact would give a report two spellings to count.
"""


class RunRecord(BaseModel):
    """A run as the store files it: searchable columns, plus the state itself.

    Frozen and ``extra="forbid"`` for the same reason
    :class:`~agentic_erp_assistant.state.events.TraceEvent` is: a record that
    can be edited after the fact is worse evidence than no record, and a field
    invented at one call site is a column no report knows how to read.

    The clocks are supplied by the caller, not stamped here -- the same
    decision :attr:`~agentic_erp_assistant.tools.models.AuditRow.occurred_at`
    records: a self-stamping model cannot be asserted on in a test, and the
    orchestrator is the only one that knows when the run started.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    """The run this record files. Must agree with :attr:`state`'s own id."""

    outcome: RunOutcome
    """How the engine gave the run back. Must agree with the state -- see
    :meth:`_the_outcome_must_not_lie`."""

    started_at: datetime
    """When the orchestrator handed the state to the engine."""

    finished_at: datetime
    """When the engine handed it back. Never before ``started_at``."""

    state: AgentState
    """The whole final state, events included.

    The store's copy of the run, not a projection of it: a reviewer asking
    "what did the graph actually do?" reads the events off this, and
    ``state_version`` is what makes an old record fail loudly on load instead
    of quietly misreading.
    """

    @model_validator(mode="after")
    def _the_record_must_describe_one_run(self) -> "RunRecord":
        if self.trace_id != self.state.trace_id:
            raise ValueError(
                f"trace_id: record says {self.trace_id!r} but its state says "
                f"{self.state.trace_id!r}; a record filed under another run's "
                f"id is two runs under one identity"
            )

        # The same two fields `engine.is_paused` reads, inlined rather than
        # imported so this package never depends on the engine: the engine's
        # orchestrator imports this one, and the dependency arrow only points
        # one way.
        paused = (
            self.state.route == "request_approval"
            and self.state.approval == "pending"
        )
        if (self.outcome == "paused") != paused:
            raise ValueError(
                f"outcome: says {self.outcome!r} for a state that is "
                f"{'paused' if paused else 'not paused'}; a label that "
                f"disagrees with the state it files is evidence that lies"
            )

        if self.finished_at < self.started_at:
            raise ValueError(
                "finished_at: a run cannot finish before it started; one of "
                "the two clocks is wrong"
            )
        return self