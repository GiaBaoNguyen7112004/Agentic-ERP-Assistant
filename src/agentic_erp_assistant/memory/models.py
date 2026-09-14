"""What is proposed, who it is proposed for, and what was decided about it.

Three types and one function, and the split between them is the design. A
**candidate** is what something -- usually a model -- suggests be remembered; it
has no identity, no store row and no authority. A **scope** says whose memory
this is and which conversation it belongs to. A **decision** is the verdict a
pure function reached about a candidate, in typed fields a reviewer can check
without reading prose.

Nothing here writes anything. That is deliberate: the object that decides and the
object that persists are separated so the decision can be tested exhaustively
against no database at all, and so "would this have been stored?" is a question
answerable by calling one function.

Why the candidate carries the turn's sources
--------------------------------------------

:attr:`MemoryCandidate.evidence_texts` and
:attr:`MemoryCandidate.tool_summaries` are what make the source rule enforceable
rather than aspirational. The project's three sources of truth are ranked --
documents answer what was written, tools answer what is true now, memory answers
what was decided or preferred -- and a memory that restates a document or an ERP
field has taken a fact from an authority that can refresh it and frozen it
somewhere that cannot. Checking that requires having the document text and the
tool result *at the moment of the decision*, so they travel with the candidate
instead of being looked up later, when the turn they came from is gone.

Why the id is derived and not generated
---------------------------------------

:func:`memory_id` is a UUIDv5 of the scope and the content, the same trick
:func:`~agentic_erp_assistant.rag.vector_index.point_id` uses. Two consequences
worth the constraint: a rejected candidate still has an identifier, so the audit
row for a refusal names something rather than carrying a null where the id
should be; and proposing the same fact twice yields the same id, so a duplicate
is visible as one id appearing twice in the audit rather than as two rows nobody
can connect.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.state.memory import (
    KEY_MAX_CHARS,
    STATEMENT_MAX_CHARS,
    MemoryKind,
    MemoryRecord,
)

__all__ = [
    "ACTOR_BOUNDED_KINDS",
    "KEY_MAX_CHARS",
    "MemoryCandidate",
    "MemoryDecision",
    "MemoryDecisionKind",
    "MemoryKind",
    "MemoryRecord",
    "MemoryScope",
    "REASON_MAX_CHARS",
    "RejectionReason",
    "SESSION_BOUNDED_KINDS",
    "STATEMENT_MAX_CHARS",
    "bounds",
    "in_bounds",
    "memory_id",
]


REASON_MAX_CHARS = 280
"""How much free text a memory decision may carry.

The same cap, for the same reason, as
:data:`~agentic_erp_assistant.reasoning.decision.RATIONALE_MAX_CHARS`: the fact
the system acts on is the typed :attr:`MemoryDecision.rejection` beside it, and
without a cap this field becomes where a model's full reasoning about a person
gets stored, exported, and read as though somebody had reviewed it.
"""


SESSION_BOUNDED_KINDS: frozenset[str] = frozenset({"intent", "session_summary"})
"""Kinds that describe one conversation and are never recalled into another.

An intent is the task in flight *in this conversation*; a session summary is the
residue *of this conversation*. Carrying either across sessions is how an
assistant starts a fresh request by resuming a task the person abandoned last
week.
"""

ACTOR_BOUNDED_KINDS: frozenset[str] = frozenset({"preference"})
"""Kinds that belong to a person rather than to a project.

A preference is how *this actor* wants to be worked with. Sharing one across a
project would let one person's choice of language silently become everybody's,
and the trace would attribute it to nobody.
"""

# ``decision`` and ``fact`` are in neither set: they are project-level and
# outlive both the session and the person who established them, which is the
# whole reason they are worth storing. Access is still bounded -- by
# ``required_scope``, checked at recall by the same rule that guards documents.


MemoryDecisionKind = Literal[
    "write",   # nothing like it was stored; store it
    "update",  # something like it was stored and has changed; replace it
    "reject",  # do not store it, and say which rule refused
    "forget",  # retire a stored record; never a verdict on a candidate
]
"""What happened to one piece of would-be memory.

Closed, and shared by the policy and the audit so a report counts one
vocabulary. ``forget`` is the member the policy never returns, and the validator
below enforces that: forgetting is not a judgement about something proposed, it
is what happens to the record an ``update`` replaced, recorded by the writer
against that record's own id. Keeping it in the same Literal means an audit
reader sees all four outcomes in one column instead of joining two.
"""


RejectionReason = Literal[
    "instruction_like",  # it tries to steer future behaviour
    "sensitive",         # a secret, or personal data storing does not require
    "low_confidence",    # the proposer was not sure, and unsure means no
    "not_established",   # the turn did not establish it: an absence, a
                         # self-description, a restated reply, or a preference
                         # the user never stated
    "not_relevant",      # there is nothing in it a later turn could act on
    "not_durable",       # true now, false shortly
    "belongs_to_rag",    # the documents already say this, and can be re-read
    "belongs_to_tools",  # the ERP already knows this, and knows it better
    "duplicate",         # already stored, unchanged
]
"""Why a candidate was refused.

Closed, and ordered as the policy evaluates it -- see
:mod:`agentic_erp_assistant.memory.policy` for why that order is the security
property and not a formality. ``not_established`` sits between the confidence
floor and the content rules: it is a mistake about *whether this turn produced
anything to store at all*, which outranks how durable or how relevant the
proposed text would have been. Distinct members rather than one ``rejected``
because they call for opposite responses: ``instruction_like`` is somebody
attacking the assistant, ``belongs_to_rag`` is the assistant working correctly,
and a reviewer counting the first must not be counting the second.
"""


_MEMORY_NAMESPACE = uuid.UUID("2b1c9a54-6f4d-5a7e-9c30-1e8b47f2d5a1")
"""A fixed namespace so a candidate maps to the same id in every process. Any
constant uuid does; this one is arbitrary and pinned."""


@dataclass(frozen=True)
class MemoryScope:
    """Whose memory this is: an access context, plus the conversation.

    Composition rather than a second context type with four copied fields. It
    states the relationship exactly -- *memory access is document access plus a
    session* -- and it means
    :func:`~agentic_erp_assistant.rag.access.is_authorized` and
    :func:`~agentic_erp_assistant.rag.access.qdrant_filter` take
    ``scope.access`` unchanged, so there is no second access rule to keep in step
    with the first (ADR 0008).

    Frozen, and snapshotted when a turn begins, for the reason
    :class:`~agentic_erp_assistant.rag.access.RetrievalContext` is: entitlements
    that could change midway would let one turn read two people's memory and the
    trace would show neither.
    """

    access: RetrievalContext
    """Who is asking, on which project, with what entitlements."""

    session_id: str
    """The conversation. Required: a scope without one could not bound an intent
    or a session summary, which are the two kinds that must never leak between
    conversations."""

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError(
                "session_id: must name a conversation. A blank one would make "
                "every session's intent look like every other session's."
            )

    @property
    def actor(self) -> str:
        return self.access.actor

    @property
    def project_code(self) -> str:
        return self.access.project_code

    @property
    def scopes(self) -> frozenset[str]:
        return self.access.scopes

    @classmethod
    def for_actor(
        cls,
        actor: str,
        *,
        project_code: str,
        session_id: str,
        scopes: frozenset[str] | set[str] | tuple[str, ...] = (),
    ) -> "MemoryScope":
        """Build a scope without assembling a retrieval context first."""
        return cls(
            access=RetrievalContext.for_actor(
                actor, project_code=project_code, scopes=scopes
            ),
            session_id=session_id,
        )


class MemoryCandidate(BaseModel):
    """Something proposed for memory, before anything has judged it.

    Frozen and ``extra="forbid"``. Frozen because the policy is a pure function
    of this object and the audit row records what it saw -- a candidate edited
    between the decision and the write would make the audit describe a different
    proposal. ``extra="forbid"`` because a field arriving from a model's tool
    call that nothing declared is exactly the input a policy check cannot see.

    Note what is *not* here: no ``memory_id``, no ``recorded_at``, no
    ``project_code``. A candidate has no identity and no home yet; those come
    from the scope and the clock at the moment a decision is acted on, which is
    what stops a proposer from nominating which project a memory lands in.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: MemoryKind
    """Which of the five things this claims to be."""

    key: str = Field(min_length=1, max_length=KEY_MAX_CHARS)
    """What it is about. Half of the ``(kind, key)`` identity a conflict is
    detected on -- see :class:`~agentic_erp_assistant.state.memory.MemoryRecord`."""

    statement: str = Field(min_length=1, max_length=STATEMENT_MAX_CHARS)
    """The proposed fact, in one sentence."""

    confidence: float = Field(ge=0.0, le=1.0)
    """How sure the proposer is. Checked against a floor, never used as a
    substitute for the content checks."""

    required_scope: str = Field(min_length=1)
    """The entitlement a reader will need. Set by the layer that watched the turn
    -- from the evidence the fact rests on -- and never by the proposer, which
    would otherwise be able to declassify what it remembers."""

    evidence_texts: tuple[str, ...] = ()
    """The passages retrieval supplied this turn, for the source check.

    Text rather than identifiers: the question is not "did this turn retrieve
    something?" but "does this statement restate what was retrieved?", and only
    the text can answer it.
    """

    tool_summaries: tuple[str, ...] = ()
    """What this turn's tool calls reported, for the same check against the other
    authority."""

    request_text: str = ""
    """The user's request this turn, verbatim. What a preference or a decision
    must have been stated in -- see the policy's ``not_established`` checks.

    Defaulting to empty is deliberate: a candidate built without it (every
    caller written before the field existed, a replay of an old trace) is judged
    exactly as before, rather than refused for evidence nobody thought to
    attach."""

    response_text: str = ""
    """The reply this turn gave. What a fact must *not* merely restate --
    ``MEMORY_CONTRACT`` item 4 ("a transcript is not memory"), enforced in code
    rather than left to the model's agreement with its own contract."""


class MemoryDecision(BaseModel):
    """One verdict on one candidate, in fields a reviewer can check.

    Frozen for the reason
    :class:`~agentic_erp_assistant.reasoning.decision.ReasoningDecision` is: this
    is handed to a writer that acts on it, and a verdict edited on the way would
    mean the audit records a decision nobody made.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: MemoryDecisionKind
    """What to do. Never ``"forget"`` -- see :data:`MemoryDecisionKind`."""

    rejection: RejectionReason | None = None
    """Which rule refused, on a rejection, and ``None`` on every other outcome."""

    supersedes: tuple[str, ...] = ()
    """The stored records this decision retires. Non-empty only on ``update``."""

    key: str | None = None
    """The key to store the record under, when it differs from the candidate's
    own. Set only by a same-topic preference update (see
    :data:`~agentic_erp_assistant.memory.policy.TOPIC_OVERLAP_RATIO`): the key
    the store already has wins over the one the proposer invented this turn,
    so a rewritten preference stops drifting to a new key on every turn.
    ``None`` means the candidate's own key, which is every other outcome."""

    reason: str = Field(default="", max_length=REASON_MAX_CHARS)
    """One line for a person reading the audit. Never parsed, never branched on;
    every fact the system acts on is in a typed field above it."""

    @property
    def stores(self) -> bool:
        """Whether acting on this verdict puts a record in the store."""
        return self.decision in ("write", "update")

    @model_validator(mode="after")
    def _the_verdict_must_not_contradict_itself(self) -> "MemoryDecision":
        if self.decision == "forget":
            raise ValueError(
                "decision: 'forget' is not a verdict on a candidate -- it is what "
                "happens to the record an update replaced, and it is recorded by "
                "the writer against that record's own id"
            )
        if (self.decision == "reject") != (self.rejection is not None):
            raise ValueError(
                f"rejection: decision {self.decision!r} with rejection "
                f"{self.rejection!r}; a refusal must name the rule that refused "
                f"it, and nothing else may carry one"
            )
        if self.supersedes and self.decision != "update":
            raise ValueError(
                f"supersedes: decision {self.decision!r} retires nothing, so ids "
                f"here name records that would be silently retained"
            )
        if self.decision == "update" and not self.supersedes:
            raise ValueError(
                "supersedes: an update replaces something, and one that names "
                "nothing is a write wearing the wrong label"
            )
        if any(not identifier.strip() for identifier in self.supersedes):
            raise ValueError("supersedes: memory ids must not be blank")
        if self.key is not None and self.decision != "update":
            raise ValueError(
                f"key: decision {self.decision!r} does not store the candidate "
                f"under a different key, so naming one here is a rewrite nobody "
                f"asked for"
            )
        if self.key is not None and not self.key.strip():
            raise ValueError("key: must not be blank when given")
        return self


def memory_id(candidate: MemoryCandidate, scope: MemoryScope) -> str:
    """The identifier this candidate will have, decided before the verdict is.

    Derived rather than generated so that a refusal still has something to name
    in the audit, and so the same proposal made twice is one id rather than two
    unconnectable rows. See the module docstring.

    The scope is part of the input: the same sentence proposed for two projects
    is two memories, and an id that ignored the project would let one overwrite
    the other.
    """
    identity = "|".join(
        (
            scope.project_code,
            scope.session_id if candidate.kind in SESSION_BOUNDED_KINDS else "",
            scope.actor if candidate.kind in ACTOR_BOUNDED_KINDS else "",
            candidate.kind,
            candidate.key,
            candidate.statement,
        )
    )
    return f"mem-{uuid.uuid5(_MEMORY_NAMESPACE, identity)}"


def bounds(kind: MemoryKind, scope: MemoryScope) -> Mapping[str, str]:
    """The fields a stored record must match for ``kind`` to be about ``scope``.

    One place, so the conflict check in
    :mod:`agentic_erp_assistant.memory.policy` and the recall filter in
    :mod:`agentic_erp_assistant.context.memory_injection` cannot disagree about
    whether a memory belongs to this turn. Two rules that answer "is this mine?"
    differently is how a preference gets recalled for the wrong person, or an
    abandoned intent gets resumed in a new conversation.
    """
    matched: dict[str, str] = {"project_code": scope.project_code}
    if kind in SESSION_BOUNDED_KINDS:
        matched["session_id"] = scope.session_id
    if kind in ACTOR_BOUNDED_KINDS:
        matched["actor"] = scope.actor
    return matched


def in_bounds(record: MemoryRecord, scope: MemoryScope) -> bool:
    """Whether ``record`` is one of ``scope``'s, by the rules :func:`bounds` sets."""
    return all(
        getattr(record, field) == value
        for field, value in bounds(record.kind, scope).items()
    )
