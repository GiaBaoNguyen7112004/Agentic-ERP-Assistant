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
* :mod:`.store`, :mod:`.vector_store` and :mod:`.audit` are the three ports --
  where records live, how they are found again, and where every decision about
  them is written down;
* everything else is an adapter behind one of those.

Nothing above the policy may write a record it did not return a verdict for.
"""

from agentic_erp_assistant.memory.audit import (
    InMemoryMemoryAudit,
    MemoryAuditRow,
    MemoryAuditSink,
)
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
from agentic_erp_assistant.memory.store import (
    InMemoryMemoryStore,
    IntentStorePort,
    MemoryStorePort,
)
from agentic_erp_assistant.memory.summary import summarize_session
from agentic_erp_assistant.memory.vector_store import (
    InMemoryMemoryVectorStore,
    MemoryVectorStorePort,
    ScoredMemory,
)

__all__ = [
    "ACTOR_BOUNDED_KINDS",
    "InMemoryMemoryAudit",
    "InMemoryMemoryStore",
    "InMemoryMemoryVectorStore",
    "IntentState",
    "IntentStatus",
    "IntentStorePort",
    "MemoryAuditRow",
    "MemoryAuditSink",
    "MemoryCandidate",
    "MemoryDecision",
    "MemoryDecisionKind",
    "MemoryKind",
    "MemoryRecord",
    "MemoryScope",
    "MemoryStorePort",
    "MemoryVectorStorePort",
    "REASON_MAX_CHARS",
    "RejectionReason",
    "ScoredMemory",
    "SESSION_BOUNDED_KINDS",
    "bounds",
    "decide",
    "in_bounds",
    "memory_id",
    "summarize_session",
    "unsafe_to_store",
]
