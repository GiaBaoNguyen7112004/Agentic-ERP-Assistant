"""Which memories reach the prompt, and a typed reason for every one that does not.

The naive memory system loads everything it has ever stored into every prompt.
It fails in three ways at once: the window fills with facts nobody asked about,
the answer drifts toward whatever was remembered rather than what was retrieved,
and the trace cannot say why -- because "we injected all of it" is not a decision
anybody made about this turn.

So recall is selective, and this module is where the selection happens and gets
written down. Every available memory ends up in exactly one of three places:
included in the plan, skipped with a :data:`SkipReason`, or excluded by the
budget with the reason
:class:`~agentic_erp_assistant.context.builder.ContextBuilder` gives it.

Two filters, and they are deliberately different mechanisms
-----------------------------------------------------------

**Skipping** happens here, before any measurement: a memory that is out of
scope, retired, or has nothing to do with this request never becomes a candidate.
**Exclusion** happens in the builder, which is already the one place that decides
what fits.

Keeping them apart is what lets
:data:`~agentic_erp_assistant.context.builder.ExclusionReason` stay the closed
pair it is. A third member -- "not relevant" -- would put a retrieval judgement
into a module whose entire job is measurement, and the two questions call for
opposite fixes: an irrelevant memory means recall is too broad, while an excluded
one means the budget is too small.

Relevance has three doors, and pinning is the interesting one
--------------------------------------------------------------

A memory is relevant if any of these holds:

* its kind is **pinned** (:data:`PINNED_KINDS`) -- the task in flight, the
  actor's preferences, this session's summary. These bear on every request by
  construction: an intent says what is being worked on, and a preference says how
  to reply, so testing either against the words of the question would drop it
  from exactly the turns where the question is phrased differently;
* the vector index **matched** it, in which case relevance was already decided by
  something better at it than word overlap;
* it **shares enough words** with the request. The lexical fallback, so recall
  still works with no embeddings client -- and so a test can prove the selection
  rule without one.

Everything else is skipped ``not_relevant``, which is the common case and is what
makes this a memory system rather than a way to fill a context window.
"""

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from agentic_erp_assistant.context.builder import ContextBuilder, ContextPlan
from agentic_erp_assistant.context.candidate import ContextCandidate
from agentic_erp_assistant.llm.tokenizer import TokenCounter
from agentic_erp_assistant.memory.models import MemoryScope, in_bounds
from agentic_erp_assistant.rag.access import is_authorized
from agentic_erp_assistant.state.memory import MemoryKind, MemoryRecord

__all__ = [
    "KIND_PRIORITY",
    "MIN_TERM_OVERLAP",
    "MemorySelection",
    "PINNED_KINDS",
    "SkipReason",
    "SkippedMemory",
    "select_memories",
]

logger = logging.getLogger(__name__)


SkipReason = Literal[
    "out_of_scope",  # this actor may not read it at all
    "superseded",    # something newer replaced it
    "not_relevant",  # nothing in it bears on what was asked
]
"""Why a memory never became a candidate for the prompt.

A closed set, kept apart from
:data:`~agentic_erp_assistant.context.builder.ExclusionReason` because these are
answers to a different question. Those two say why something that *was* a
candidate did not fit; these say why something was never offered. A reviewer
reading a trace has to be able to tell "the budget was tight" from "this actor
cannot see that" -- one is a tuning problem and the other is the access rule
working.
"""


PINNED_KINDS: frozenset[MemoryKind] = frozenset(
    {"intent", "preference", "session_summary"}
)
"""Kinds that bear on every request, so relevance is not tested for them.

Each for a reason that would survive being argued with. An **intent** is what
the turn is part of; a request that does not mention it is usually the request
that advances it. A **preference** says how to reply, and it applies to answers
about anything. A **session summary** is what this conversation has already
established, which is exactly what a follow-up question assumes.

The three that are *not* here -- ``decision`` and ``fact`` -- are about
particular subjects, so a request that does not touch the subject has no use for
them.
"""


KIND_PRIORITY: dict[MemoryKind, int] = {
    "intent": 5_000,
    "preference": 4_000,
    "session_summary": 3_000,
    "decision": 2_000,
    "fact": 1_000,
}
"""Who survives when the memory budget runs out. Higher is kept.

Ordered by how badly the turn goes without it. Losing the **intent** means the
assistant does not know what it is doing; losing a **preference** means an answer
in the wrong shape; losing the **session summary** means re-asking something
already settled. A dropped **decision** or **fact** costs a retrieval or a
question, which is recoverable.

Spaced by a thousand so recency can break ties inside a kind without ever
crossing between them -- see :func:`_candidates`. The numbers are this module's
policy, passed to the builder as data, which is where
:class:`~agentic_erp_assistant.context.candidate.ContextCandidate` says such a
decision belongs: visible to a reviewer as a number rather than fixed inside the
component that measures tokens.
"""


MIN_TERM_OVERLAP = 2
"""How many content words a request and a memory must share to be related.

Two rather than one: a single shared word is the noise floor -- "project",
"sprint" and "budget" appear in nearly every request this assistant sees and in
nearly every memory it stores, so a threshold of one would pin everything and
make the whole module a no-op with extra steps.

Deliberately a count and not a ratio. A long statement and a short one are
equally relevant if they are about the same two things, and normalizing by length
would quietly favour terse memories over complete ones.
"""


_WORD = re.compile(r"[a-z0-9]+")

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "about", "an", "and", "any", "are", "as", "at", "be", "been", "by",
        "can", "did", "do", "does", "for", "from", "get", "give", "has", "have",
        "how", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or",
        "our", "please", "show", "tell", "that", "the", "then", "there", "this",
        "to", "was", "we", "were", "what", "when", "where", "which", "who",
        "why", "will", "with", "you", "your",
    }
)
"""Words too common to carry a topic.

Longer than the policy module's list, and the difference is the point: this one
filters a *question*, so it has to drop the interrogatives and the politeness
that every question contains. Two lists rather than one shared list, because they
are answering different questions -- "does this restate a source?" and "is this
about the same thing?" -- and merging them would tie a change in one rule to a
change in the other.
"""


def _terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOP_WORDS}


@dataclass(frozen=True)
class SkippedMemory:
    """A memory that never became a candidate, with the reason attached.

    A pair rather than parallel lists, for the reason
    :class:`~agentic_erp_assistant.context.builder.ExcludedCandidate` is one: the
    reason has to travel with the record, or answering "why was this not
    recalled?" means re-running the selection and hoping it decides the same way.
    """

    record: MemoryRecord
    reason: SkipReason


@dataclass(frozen=True)
class MemorySelection:
    """What recall chose, what it passed over, and what it cost.

    Frozen and holding tuples: this is the record of a decision already made, and
    the trace's answer to "what was this turn shown, and why that?".
    """

    selected: tuple[MemoryRecord, ...]
    """What goes into the prompt, in the order it will be rendered."""

    skipped: tuple[SkippedMemory, ...]
    """What never became a candidate, each with its :data:`SkipReason`."""

    plan: ContextPlan
    """The builder's own record: what fitted, what did not, and the token count.

    Kept whole rather than reduced to the selected list, because
    :attr:`~agentic_erp_assistant.context.builder.ContextPlan.overflowed` is the
    signal that recall is asking for more room than it has -- and that is a fact
    about this turn worth having in the trace rather than re-deriving.
    """

    @property
    def excluded(self) -> tuple[str, ...]:
        """The ids the budget kept out, for a one-line trace note."""
        return tuple(item.candidate.candidate_id for item in self.plan.excluded)


def _relevant(
    record: MemoryRecord, *, request_terms: set[str], matched: frozenset[str]
) -> bool:
    """Whether this memory bears on the request. See the module docstring."""
    if record.kind in PINNED_KINDS:
        return True
    if record.memory_id in matched:
        return True
    return len(request_terms & _terms(record.statement)) >= MIN_TERM_OVERLAP


def _candidates(records: Sequence[MemoryRecord]) -> list[ContextCandidate]:
    """Turn kept records into candidates, priced by kind and then by recency.

    Recency is expressed as a *rank* subtracted from the kind's band rather than
    as a timestamp: a raw clock value would swamp the band and reorder kinds, and
    the whole point of the spacing is that a fact never outranks an intent for
    being newer.
    """
    ordered = sorted(
        records, key=lambda record: (record.recorded_at, record.memory_id), reverse=True
    )
    return [
        ContextCandidate(
            candidate_id=record.memory_id,
            kind="memory",
            text=" ".join(record.statement.split()),
            priority=KIND_PRIORITY[record.kind] - position,
        )
        for position, record in enumerate(ordered)
    ]


def select_memories(
    request: str,
    available: Iterable[MemoryRecord],
    *,
    scope: MemoryScope,
    budget_tokens: int,
    model: str,
    matched: Iterable[str] = (),
    counter: TokenCounter | None = None,
) -> MemorySelection:
    """Choose the memories worth this turn's tokens, and record what was passed over.

    Args:
        request: What the user asked. Used only for the lexical relevance door;
            pinned kinds and vector matches never consult it.
        available: Everything recall gathered -- the pinned records, plus
            whatever the semantic index returned. Passing more than is in scope
            is safe; this re-checks.
        scope: Whose memory this is. Applied through
            :func:`~agentic_erp_assistant.rag.access.is_authorized` and
            :func:`~agentic_erp_assistant.memory.models.in_bounds` -- the same
            two rules the store and the index apply, so a record that reached
            here by any path is checked by the same policy a third time. That is
            deliberate belt-and-braces: this is the last point before text enters
            a prompt.
        budget_tokens: How many tokens of memory the prompt can afford.
        model: Whose tokenizer measures that. Required, with no default, for the
            reason :attr:`~agentic_erp_assistant.context.builder.ContextBuilder.model`
            is: a token count means nothing without the model that produced it.
        matched: Ids the vector index returned for this request, if any. Their
            relevance was decided by something better at it than word overlap.
        counter: How to measure. Defaults to the builder's own.

    Returns:
        A :class:`MemorySelection` in which every input record appears exactly
        once -- selected, skipped, or excluded by the plan.
    """
    request_terms = _terms(request)
    matched_ids = frozenset(matched)

    kept: list[MemoryRecord] = []
    skipped: list[SkippedMemory] = []
    for record in available:
        if not record.live:
            skipped.append(SkippedMemory(record, "superseded"))
        elif not (is_authorized(record, scope.access) and in_bounds(record, scope)):
            # Reached here despite the store and the index both filtering, which
            # means a caller assembled the list by hand or one of those filters
            # has drifted. Logged at warning for that reason: it is not expected.
            logger.warning(
                "recall was offered %s, which %s on %s may not read; the store, "
                "the index and this filter have diverged",
                record.memory_id,
                scope.actor,
                scope.project_code,
            )
            skipped.append(SkippedMemory(record, "out_of_scope"))
        elif not _relevant(record, request_terms=request_terms, matched=matched_ids):
            skipped.append(SkippedMemory(record, "not_relevant"))
        else:
            kept.append(record)

    builder = ContextBuilder(model=model, **({"counter": counter} if counter else {}))
    plan = builder.build(_candidates(kept), budget_tokens=budget_tokens)

    by_id = {record.memory_id: record for record in kept}
    selected = tuple(by_id[candidate.candidate_id] for candidate in plan.included)

    logger.info(
        "recall for %s: %d selected, %d skipped, %d over budget, %d token(s)",
        scope.session_id,
        len(selected),
        len(skipped),
        len(plan.excluded),
        plan.used_tokens,
    )
    return MemorySelection(
        selected=selected, skipped=tuple(skipped), plan=plan
    )
