"""One thing worth remembering after the turn that learned it has ended.

This is the memory layer's only type that crosses into the reasoning state, so
it lives here for the reason
:class:`~agentic_erp_assistant.state.evidence.EvidenceSnippet` does:
:class:`~agentic_erp_assistant.state.agent_state.AgentState` carries a tuple of
these, and defining them in ``memory/`` would force the innermost layer to
import a package holding a vector index, a Postgres adapter and a provider
client. :mod:`agentic_erp_assistant.memory.models` re-exports the names, so
there is exactly one definition and every memory-side import still reads
naturally.

What a memory is, and what it is emphatically not
-------------------------------------------------

A record here is *durable user, project or task context that a later turn could
not obtain from anywhere else*. It is not a transcript, not a cached answer, and
not a copy of live ERP state. The three sources of truth stay separate and stay
ranked:

* **retrieval** owns what the documents say, and is the only thing a citation
  may point at;
* **tools** own what is true right now -- a budget, a sprint's burn-down, the
  open risks;
* **memory** owns what was decided, preferred, or left unfinished, which neither
  of the other two records anywhere.

The consequence a reviewer should hold this module to: a memory can never be the
reason an answer asserts a fact about the project. It has no ``locator`` and no
``tag``, so unlike an :class:`~agentic_erp_assistant.state.evidence.EvidenceSnippet`
there is nothing here a citation could be built from -- and the grounding check
in :mod:`agentic_erp_assistant.engine.nodes` rejects any citation that does not
match a passage retrieval actually supplied. The separation is structural, not a
line in a prompt.

Why it carries ``project_code`` and ``required_scope``
------------------------------------------------------

Those two field names are not decoration. They make this class satisfy
:class:`~agentic_erp_assistant.rag.access.Restricted` structurally, so
:func:`~agentic_erp_assistant.rag.access.is_authorized` and
:func:`~agentic_erp_assistant.rag.access.qdrant_filter` apply to a memory exactly
as they apply to a chunk -- one access rule serving a third read path rather than
a second rule that will diverge from the first (ADR 0008). A memory is at least
as sensitive as the document that produced it, and a memory store with its own
private notion of who may read what is a leak waiting for the first person who
edits only one of the two.

Forgetting is superseding, never deleting
-----------------------------------------

:attr:`MemoryRecord.superseded_at` retires a record; nothing removes it. A
deleted memory takes its audit trail's other half with it, and "why did the
assistant stop believing that?" becomes unanswerable. A superseded record stops
being recalled, keeps its row, and the record that replaced it names it in
:attr:`MemoryRecord.supersedes` -- so the history of a changed fact reads
forwards.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "KEY_MAX_CHARS",
    "MemoryKind",
    "MemoryRecord",
    "STATEMENT_MAX_CHARS",
]


MemoryKind = Literal[
    "preference",       # how this actor wants the assistant to work
    "decision",         # something the project settled and may be relied on
    "fact",             # a confirmed project fact with no live source of its own
    "intent",           # the task in flight: goal, slots, completion state
    "session_summary",  # the durable residue of a session, not its transcript
]
"""The five things this system is willing to remember.

A closed set, for the reason
:data:`~agentic_erp_assistant.reasoning.decision.DecisionRoute` is one: the
policy layer branches on it, the recall layer prioritises on it, and an audit
report groups by it. A kind nobody has written a rule for has to fail at
construction rather than turn up later as a category that quietly appeared in an
evidence bundle.

What is absent is the design. There is no ``observation``, no ``history`` and no
``note``: each of those is a name under which a transcript gets stored one turn
at a time, and the whole point of this system is that the transcript is the
thing it refuses to keep.
"""


STATEMENT_MAX_CHARS = 400
"""How long one remembered statement may be.

A cap for the reason
:data:`~agentic_erp_assistant.reasoning.decision.RATIONALE_MAX_CHARS` is one, and
with a sharper edge here: without it, ``statement`` becomes where a turn stores
its own summary of everything that happened, and the store fills with paraphrased
conversation under a field named for facts. A durable fact fits in a sentence. If
it does not, it is either several facts or it is a transcript.
"""

KEY_MAX_CHARS = 120
"""How long the identity half of a memory may be. See :attr:`MemoryRecord.key`."""


class MemoryRecord(BaseModel):
    """One durable fact, with everything needed to audit and retire it.

    Frozen and ``extra="forbid"``. Frozen because a record is evidence about a
    decision that has already been made and audited -- a statement edited after
    the audit row was written would make the two disagree, with nothing to say
    which is right. ``extra="forbid"`` because a field smuggled in at one call
    site is memory content that no policy check ever looked at.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # -- identity ----------------------------------------------------------

    memory_id: str = Field(min_length=1)
    """This record's own identifier, stable for its whole life.

    Named in the audit row, in :attr:`supersedes` on whatever replaces it, and
    used as the ``candidate_id`` when the record competes for prompt space -- so
    it has to be stable across processes and across runs. A positional index or
    an object id would make the same memory a different candidate on a different
    day, and the context plan would stop being reproducible.
    """

    kind: MemoryKind
    """What sort of thing this is. See :data:`MemoryKind`."""

    key: str = Field(min_length=1, max_length=KEY_MAX_CHARS)
    """What this record is *about*, as a short stable identifier.

    ``(kind, key)`` is the identity a conflict is detected on: a second
    ``preference`` about ``"reply_language"`` supersedes the first rather than
    sitting beside it. Without it, memory grows by accumulating restatements and
    recall has to pick between three versions of one preference with nothing to
    tell it which is current -- which is exactly how stale memory starts winning
    arguments against live data.
    """

    statement: str = Field(min_length=1, max_length=STATEMENT_MAX_CHARS)
    """The fact itself, in one sentence, as it will be shown to the model.

    Read as background context, never as an instruction -- see the module
    docstring, and the memory rules in
    :data:`~agentic_erp_assistant.llm.prompts.SYSTEM_POLICY`, which say so in the
    role that carries instructions. The policy layer refuses to store an
    imperative in the first place, so the two defences are independent.
    """

    # -- who may read it ---------------------------------------------------

    project_code: str = Field(min_length=1)
    """The project this memory belongs to. The tenant boundary, and the first
    half of :func:`~agentic_erp_assistant.rag.access.is_authorized`."""

    required_scope: str = Field(min_length=1)
    """The scope an actor must hold to have this recalled into their prompt.

    Inherited from whatever produced the memory: a fact learned from a
    finance-scoped document keeps that document's requirement. A memory that
    relaxed the scope of the thing it summarises would be a way to read a
    restricted document by asking about it twice.
    """

    actor: str = Field(min_length=1)
    """Who the memory was recorded for. Carried so a refusal is attributable and
    so a preference belongs to a person rather than to a project."""

    session_id: str = Field(min_length=1)
    """The conversation this was learned in.

    Required, with no default, for the reason
    :attr:`~agentic_erp_assistant.state.agent_state.AgentState.trace_id` is: a
    record that could exist without one is a memory nobody can scope, expire or
    explain the origin of. ``intent`` and ``session_summary`` are additionally
    *bounded* by it -- they describe one conversation and are never recalled into
    another.
    """

    # -- the record --------------------------------------------------------

    recorded_in_run: str = Field(min_length=1)
    """The ``trace_id`` of the run that produced this.

    The join back to the evidence: without it a reviewer can see that something
    was remembered but not which sequence of decisions produced it, and "what
    does it believe" without "on what basis" is half an answer. The same argument
    :attr:`~agentic_erp_assistant.tools.models.AuditRow.trace_id` makes.
    """

    recorded_at: datetime
    """When it was written.

    Supplied by the caller, with no ``default_factory`` reading the clock -- the
    decision :attr:`~agentic_erp_assistant.tools.models.AuditRow.occurred_at`
    already records. It is also shown to the model beside the statement, because
    a fact whose age is invisible is a fact that gets believed over a fresher
    tool result.
    """

    confidence: float = Field(ge=0.0, le=1.0)
    """0.0-1.0, as reported by whatever proposed the memory.

    Recorded and used to order recall; never a substitute for the policy check.
    A high number on a candidate the policy refuses changes nothing, which is the
    point of having the policy be a function rather than a threshold.
    """

    supersedes: tuple[str, ...] = ()
    """The ``memory_id`` values this record replaces, if any.

    A tuple, not a list: a frozen model holding a mutable sequence is not frozen.
    Present so the history of a changed fact reads forwards -- see the module
    docstring on forgetting.
    """

    superseded_at: datetime | None = None
    """When this record stopped being current, or ``None`` while it still is.

    The only thing that retires a memory. Recall filters on it, the vector index
    filters on it, and nothing deletes the row.
    """

    links: tuple[str, ...] = ()
    """Identifiers this memory relates to -- other memories, or ERP ids.

    Deliberately a flat list of strings and not an edge type. It is the seam
    ADR 0013 leaves open: if the queries this assistant actually has to answer
    ever start needing traversal rather than lookup, the relationships are
    already recorded and a graph store becomes an adapter rather than a
    migration. Until then, storing edges nobody traverses is operational cost
    with no query behind it.
    """

    @property
    def live(self) -> bool:
        """Whether this record is still current.

        The one question recall and the vector filter both ask, asked in one
        place so the two cannot answer it differently.
        """
        return self.superseded_at is None

    @model_validator(mode="after")
    def _must_describe_a_memory_that_could_exist(self) -> "MemoryRecord":
        for name in (
            "memory_id",
            "key",
            "statement",
            "project_code",
            "required_scope",
            "actor",
            "session_id",
            "recorded_in_run",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name}: must not be blank")

        if any(not identifier.strip() for identifier in self.supersedes):
            raise ValueError("supersedes: memory ids must not be blank")
        if self.memory_id in self.supersedes:
            raise ValueError(
                "supersedes: a record cannot supersede itself; that would retire "
                "the replacement at the moment it was written"
            )
        if any(not link.strip() for link in self.links):
            raise ValueError("links: identifiers must not be blank")

        if self.superseded_at is not None and self.superseded_at < self.recorded_at:
            raise ValueError(
                "superseded_at: a memory cannot be retired before it was "
                "recorded; one of the two clocks is wrong"
            )
        return self

    def retired(self, at: datetime) -> "MemoryRecord":
        """Return this record, superseded as of ``at``.

        A new object rather than a mutation, the same rule
        :meth:`~agentic_erp_assistant.state.agent_state.AgentState.evolve`
        follows: the store writes the retirement and the caller keeps the record
        it was holding, so a reviewer can diff the two.

        Raises:
            ValueError: This record was already retired. Re-retiring would move
                the date on which a memory stopped being believed, which is a
                fact somebody may have audited.
        """
        if self.superseded_at is not None:
            raise ValueError(
                f"memory {self.memory_id!r} was already superseded at "
                f"{self.superseded_at.isoformat()}; moving that date would "
                f"rewrite when the assistant stopped believing it"
            )
        current = {name: getattr(self, name) for name in type(self).model_fields}
        return type(self)(**{**current, "superseded_at": at})
