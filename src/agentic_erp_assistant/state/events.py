"""What happened during a turn, recorded as it happens.

The trace is the audit evidence for this project, so the event log is not
debug output that a later exporter reconstructs from whatever state survived.
It is carried in the state itself, appended as the graph moves, and it stays
complete even when a turn ends badly -- a run that failed halfway has to leave
behind the same kind of record as one that succeeded, or the failures are
exactly the runs nobody can review.

Why the kind is a closed set
----------------------------

:data:`EventKind` enumerates the transitions the audit actually depends on:
node entry and exit, the route that was chosen, evidence arriving, a tool
being called, an approval being asked for and answered, a retry being spent,
and a failure being recorded. Those are the things a reviewer counts. A free
string here would let each node invent its own vocabulary, and a report that
groups by kind would then be counting spellings.

Why there is no timestamp
-------------------------

An event does not stamp itself. Two reasons, and the second is the one that
matters: a clock read inside a frozen model makes every test that builds an
expected event non-deterministic, and -- more importantly -- the ordering that
an auditor needs is already carried by the log's position, which a clock with
millisecond ties cannot improve on. Wall-clock time is a property of the run,
and belongs to the trace store that writes the run down.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["EVENT_DETAIL_MAX_CHARS", "EventKind", "TraceEvent"]


EventKind = Literal[
    "node_entered",       # the graph handed control to a node
    "node_exited",        # the node returned a state
    "route_selected",     # a DecisionRoute was chosen
    "evidence_retrieved", # the retriever returned chunks
    "tool_called",        # a tool was executed through the gateway
    "approval_requested", # a mutating call stopped for a human
    "approval_recorded",  # the human answered
    "retry_scheduled",    # an attempt was spent and another will follow
    "rate_limited",       # a call was refused because a budget was spent
    "memory_recalled",    # background from earlier turns entered the prompt
    "memory_written",     # something from this turn was stored, or replaced
    "memory_rejected",    # something proposed was refused, and by which rule
    "history_recalled",   # the session's recent turns entered the prompt
    "history_promoted",   # turns leaving the window were folded into the summary
    "contract_declared",  # the planner said what a complete reply needs to rest on
    "failed",             # a FailureMode was assigned
    "run_failed",         # the loop guard ended a run that would not end
]
"""The transitions the audit depends on.

Closed for the same reason
:data:`~agentic_erp_assistant.reasoning.decision.DecisionRoute` is: a kind
nobody reports on has to fail at construction rather than appear later as a
category that quietly showed up in an evidence bundle.

``run_failed`` is separate from ``failed`` on the same principle. ``failed``
means a node looked at what happened and assigned a reason; ``run_failed``
means no node did -- the loop guard stopped a run that kept producing states
and never produced an ending. One is the system reporting an outcome, the other
is the system admitting it lost control of the graph, and a reviewer must be
able to count the second without the first burying it.

``rate_limited`` is its own kind rather than a ``failed`` event for the reason
``route_selected`` is not one either: a reviewer counting outages must not be
counting the safety layer working. A run that was throttled and a run whose
backend fell over look identical under one label, and only one of them is a
bug.

``memory_written`` and ``memory_rejected`` are kept apart on the same principle,
and the second is the one worth having. A memory system that only logged what it
stored would answer "what does the assistant believe?" and leave the more
telling question unanswered -- what it declined to believe, and which rule
refused it. A run where four proposals were refused and one was kept is a run
where the policy did its job, and it must not read the same as a run where five
were stored.

``history_recalled`` and ``history_promoted`` join the memory kinds on the same
grounds, and there are two of them for the same reason: a turn being shown its
session's recent past and a turn pushing the oldest of that past out of the
window -- into the session summary, where it is paraphrased and bounded -- are
different facts about one mechanism, and the second is the one a reviewer of
the promotion path counts. It is the watermark's receipt: everything named in
its detail has a summary row beside it or nothing was promoted.

``contract_declared`` joins them on the same grounds, for the reason ADR 0021
gives: what a complete reply to this turn must rest on is decided once,
before the graph runs, the same way recall is -- not a re-plan's business to
revisit mid-turn. Its detail names the declared needs, or says a declaration
was unreadable and this turn continues unchecked; the check itself, and what
it does when a declared need goes unmet, is a later event this kind does not
carry.

None of them is emitted by a node. Recall happens before the graph runs and
consolidation after it ends, both in the orchestrator, so these events describe
work the step budget deliberately does not pay for -- see ADR 0011 for memory
and ADR 0014 for the short-term window.
"""


EVENT_DETAIL_MAX_CHARS = 280
"""How much free text one event may carry.

The same cap, for the same reason, as ``RATIONALE_MAX_CHARS`` on a decision:
without one, ``detail`` becomes where a node dumps a prompt, a model's full
reasoning, or a tool's entire payload, all of which then travel into an
exported trace and get read as if they had been reviewed. Anything larger than
a line belongs in the artifact the event points at, not in the log entry.
"""


class TraceEvent(BaseModel):
    """One line of the run's record.

    Frozen: an event is a claim about something that already happened, and a
    record that can be edited afterwards is not evidence. ``extra="forbid"``
    because a field invented at one call site would be a column no report
    knows how to read.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str = Field(min_length=1)
    """Which node emitted this. The graph's own name for it, not a class name --
    the trace has to stay readable after a refactor renames the class."""

    kind: EventKind
    """What happened. See :data:`EventKind`."""

    detail: str = Field(default="", max_length=EVENT_DETAIL_MAX_CHARS)
    """A one-line note for a human reading the trace.

    Never parsed and never branched on. Every fact the runtime acts on lives in
    a typed field somewhere else; if a branch ever needs something that is only
    written here, the fix is a new field, not a regex.
    """
