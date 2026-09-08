"""Memory: durable user, project and task context, and the policy that guards it.

The rule the whole package exists to enforce is a refusal: **do not store what a
later turn could get from somewhere better.** Documents are retrieved, live state
is looked up, and what is left -- a preference, a decision, an unfinished task, a
session's residue -- is what lands here.

The layers, innermost first:

* :mod:`~agentic_erp_assistant.state.memory` holds the record itself, because the
  turn's state carries it;
* :mod:`.models` adds the candidate, the scope and the verdict -- what is
  proposed, for whom, and what was decided;
* :mod:`.policy` is the gate: a pure function, six checks, and a default of
  refusing;
* :mod:`.intent` and :mod:`.summary` are the two kinds that are *projected*
  rather than proposed -- a task with a lifecycle, and a conversation's own
  residue;
* everything else -- stores, indexes, proposers -- is machinery behind a port.

Nothing above the policy may write a record it did not return a verdict for.
"""

from agentic_erp_assistant.memory.intent import IntentState, IntentStatus
from agentic_erp_assistant.memory.models import (
    ACTOR_BOUNDED_KINDS,
    MemoryCandidate,
    MemoryDecision,
    MemoryDecisionKind,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    REASON_MAX_CHARS,
    RejectionReason,
    SESSION_BOUNDED_KINDS,
    bounds,
    in_bounds,
    memory_id,
)
from agentic_erp_assistant.memory.policy import decide, unsafe_to_store
from agentic_erp_assistant.memory.summary import summarize_session

__all__ = [
    "ACTOR_BOUNDED_KINDS",
    "IntentState",
    "IntentStatus",
    "MemoryCandidate",
    "MemoryDecision",
    "MemoryDecisionKind",
    "MemoryKind",
    "MemoryRecord",
    "MemoryScope",
    "REASON_MAX_CHARS",
    "RejectionReason",
    "SESSION_BOUNDED_KINDS",
    "bounds",
    "decide",
    "in_bounds",
    "memory_id",
    "summarize_session",
    "unsafe_to_store",
]
