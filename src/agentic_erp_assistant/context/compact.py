"""Shrinking a conversation without letting it forget a pending approval.

A long conversation eventually has to be compressed, and the obvious way to do it
is to hand the whole thing to a model and keep what comes back. That is the bug
this module exists to prevent. A summarizer optimizes for readable prose, and
"there is an unapproved request to delete record rec-9 waiting on a human" is
exactly the sort of detail that reads like housekeeping and gets compressed away.
Once it is gone, the next turn does not know an approval is pending -- and the
project rule that no write executes without a recorded decision has been defeated
by a compression step rather than by anything anyone would call a security bug.

So compaction here is an **allow-list**, and the summary is a passenger.

Six fields survive, named in :data:`PRESERVED_FIELDS`. Everything else is
dropped: a raw transcript, a scratchpad, whatever a future turn starts
accumulating. The direction matters. A deny-list -- "drop the transcript, keep
the rest" -- fails open, because a field added upstream next month passes through
a rule that was never told about it. An allow-list fails closed: the new field is
dropped, and somebody has to consciously decide to keep it.

Preserved, not validated
------------------------

The six values are copied through unchanged. This module does not check that
``citations`` contains well-formed :class:`~agentic_erp_assistant.llm.schemas.Citation`
objects, or that ``pending_approvals`` matches
:class:`~agentic_erp_assistant.llm.schemas.ApprovalRequest`, even though both
types exist one package over and the temptation is considerable.

The reason is that validation is a way to *fail*, and every way this function can
fail is a way a pending approval can vanish. A malformed approval that raises
here is an approval that did not survive compaction, which is precisely the
outcome the allow-list exists to make impossible. Whatever shape an upstream
layer created, it comes out the other side. Policing that shape is that layer's
job, and it has already had its chance.

The one normalization is that a top-level ``list`` becomes a ``tuple``, so the
record is genuinely immutable rather than a frozen dataclass wrapping a mutable
list. Element objects are untouched.

The summary is best-effort, the six fields are not
--------------------------------------------------

:func:`compact_conversation` takes an optional summarizer, which may well be an
LLM call. A provider outage must not be able to destroy conversation state, so a
summarizer that raises is logged and replaced by the deterministic structural
summary below -- the six fields survive either way. The guarantee is the
allow-list; the summary is the nice-to-have. Never the other way round.

One consequence for later: a summary derived from a transcript is generated text
that may carry instructions someone put in a document. When it enters a prompt it
goes in as history or evidence, never as policy.
"""

import logging
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "CompactedConversation",
    "compact_conversation",
    "PRESERVED_FIELDS",
    "structural_summary",
    "SUMMARY_MAX_CHARS",
    "SUMMARY_UNAVAILABLE",
    "Summarizer",
]

logger = logging.getLogger(__name__)


PRESERVED_FIELDS = (
    "user_goal",  # what the user is actually trying to do
    "accepted_facts",  # what has been established and may be relied on
    "decisions",  # what the conversation settled, and may not re-litigate
    "citations",  # the sources those facts rest on
    "pending_approvals",  # writes waiting on a human decision
    "safety_flags",  # guardrail findings that must outlive the turn
    "unresolved_questions",  # what is still open
)
"""The complete allow-list, in the order a summary lists them.

A closed tuple, and the only thing that decides what survives compaction. Adding
a field here is a deliberate act with a diff someone reviews; that is the entire
point of writing the rule this way round.

Four of the seven are safety state. ``pending_approvals`` and ``safety_flags`` in
particular are why this module cannot be a summarizer: they are small, they read
as incidental, and losing either one silently removes a control.

``decisions`` was the one deliberate addition, made when the memory layer
landed. It is here rather than folded into ``accepted_facts`` because the two
decay differently: a fact can be re-established by retrieving the document it
came from, while a decision exists only because a conversation reached it, and
losing one means re-litigating an argument that was already settled. It is also
what :mod:`agentic_erp_assistant.memory.summary` reads first when it projects a
session into durable memory.
"""


SUMMARY_MAX_CHARS = 1000
"""How much summary a compacted conversation may carry.

A cap for the same reason
:data:`~agentic_erp_assistant.reasoning.decision.RATIONALE_MAX_CHARS` is one:
without it, the field becomes the place an entire transcript is stored again
under a different name, and compaction stops compacting.
"""

SUMMARY_UNAVAILABLE = "(summary unavailable)"
"""Stands in when summarizing failed. Said plainly rather than left empty."""

_TRUNCATED = "…"


@runtime_checkable
class Summarizer(Protocol):
    """Turns a whole conversation state into one short description.

    Receives the state *before* the allow-list is applied -- including the
    transcript that is about to be dropped, since summarizing what survives
    anyway would be pointless.

    May fail. :func:`compact_conversation` treats a raised exception as "no
    summary", never as a reason to lose the preserved fields.
    """

    def __call__(self, state: Mapping[str, object]) -> str:
        """Describe ``state`` in at most a paragraph."""
        ...


@dataclass(frozen=True)
class CompactedConversation:
    """A conversation reduced to the six fields that must not be lost.

    Frozen: this is the record of a compaction that already happened, and a
    ``pending_approvals`` edited after the fact would make it a worse witness
    than no record at all.

    Every preserved field is typed ``object | None`` rather than something
    narrower, and that is the deliberate seam described in the module docstring:
    this module preserves values, it does not validate them. ``None`` means the
    field was not present in the input. A field explicitly set to ``None`` by a
    caller is treated the same way, which loses nothing -- ``None`` carries no
    information to preserve.
    """

    summary: str
    """A description of the conversation, including what was dropped."""

    user_goal: object | None = None
    """What the user is trying to accomplish."""

    accepted_facts: object | None = None
    """What has been established well enough to build on."""

    decisions: object | None = None
    """What the conversation settled. Kept apart from ``accepted_facts`` because
    a fact can be re-established from the document it came from and a decision
    cannot -- see :data:`PRESERVED_FIELDS`."""

    citations: object | None = None
    """The sources behind those facts. Grounding has to outlive compaction."""

    pending_approvals: object | None = None
    """Writes still waiting on a human. Losing one defeats the approval gate."""

    safety_flags: object | None = None
    """Guardrail findings. A flag that expires at compaction is not a control."""

    unresolved_questions: object | None = None
    """What is still open, so a later turn does not re-ask or wrongly assume."""

    def as_state(self) -> dict[str, object]:
        """The compacted conversation as a plain mapping, absences omitted.

        The shape a caller feeds back into the next turn, and the shape a test
        can meaningfully assert a dropped field's *absence* against -- asking
        whether a dataclass has a ``raw_transcript`` attribute proves nothing,
        since it never could have.
        """
        state: dict[str, object] = {"summary": self.summary}
        for name in PRESERVED_FIELDS:
            value = getattr(self, name)
            if value is not None:
                state[name] = value
        return state


def _describe_size(value: object) -> str:
    """Say how big a dropped value was, without quoting any of it.

    Deliberately never includes content. The dropped field is usually the raw
    transcript, and a summary that quotes it has reintroduced the thing
    compaction just removed.
    """
    if isinstance(value, str):
        return f"{len(value)} chars"
    if isinstance(value, Sized):
        return f"{len(value)} items"
    return type(value).__name__


def structural_summary(state: Mapping[str, object]) -> str:
    """Describe a compaction from its structure alone -- no model, no network.

    The default summarizer, and the fallback when an injected one fails. It says
    what was kept and what was dropped, which is the minimum a reader of a trace
    needs in order to ask whether the compaction was reasonable.

    Deterministic: the same state always produces the same string. Dropped names
    are sorted rather than left in the mapping's insertion order, which is a
    property of how the caller happened to build its dict and not a fact about
    the conversation.
    """
    kept = [name for name in PRESERVED_FIELDS if state.get(name) is not None]
    dropped = sorted(name for name in state if name not in PRESERVED_FIELDS)

    parts = [f"kept {', '.join(kept)}" if kept else "kept nothing"]
    if dropped:
        parts.append(
            "dropped "
            + ", ".join(f"{name} ({_describe_size(state[name])})" for name in dropped)
        )
    return "; ".join(parts)


def _cap(text: str) -> str:
    """Trim to :data:`SUMMARY_MAX_CHARS`, marking that something was cut."""
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    return text[: SUMMARY_MAX_CHARS - len(_TRUNCATED)] + _TRUNCATED


def _summarize(state: Mapping[str, object], summarizer: Summarizer | None) -> str:
    """Run the summarizer, and treat every way it can go wrong as "no summary".

    A broad ``except Exception`` on purpose. The injected summarizer may be an
    LLM call, and the failures it can produce -- timeouts, auth, a rejected
    reply, a bug in someone's adapter -- have nothing in common except that none
    of them is a reason to drop a pending approval. This is the one place in the
    runtime where swallowing an exception is the safer behavior, because the
    alternative loses state that cannot be recovered.

    A summarizer returning a non-string is treated the same way. It is a caller
    bug, but not one worth destroying a conversation over.
    """
    chosen = summarizer or structural_summary
    try:
        text = chosen(state)
        if not isinstance(text, str):
            raise TypeError(f"summarizer returned {type(text).__name__}, not str")
        return _cap(text)
    except Exception as error:
        logger.warning(
            "summarizer failed (%s: %s); compacting without it -- the preserved "
            "fields are unaffected",
            type(error).__name__,
            error,
        )

    if chosen is structural_summary:
        return SUMMARY_UNAVAILABLE
    try:
        return _cap(structural_summary(state))
    except Exception:  # pragma: no cover - a pure function over a mapping
        logger.exception("structural summary failed")
        return SUMMARY_UNAVAILABLE


def _freeze(value: object) -> object:
    """Make a preserved value immutable without changing what it holds.

    Only a top-level ``list`` is converted, and only to a ``tuple``. Elements are
    left exactly as they arrived -- this module preserves, it does not transform.
    Anything that is not a list is passed straight through, which matters:
    ``tuple()`` over a string would silently explode it into characters.
    """
    if isinstance(value, list):
        return tuple(value)
    return value


def compact_conversation(
    state: Mapping[str, object],
    *,
    summarizer: Summarizer | None = None,
) -> CompactedConversation:
    """Compact ``state`` down to the allow-list, plus a summary.

    Args:
        state: The conversation state, in whatever shape it accumulated. Untyped
            on purpose: this function is the boundary where loose state becomes
            typed, so it must accept fields it has never heard of in order to
            drop them. A strict model here would *reject* a raw transcript, and
            rejecting is not dropping.
        summarizer: How to describe the conversation, the dropped parts included.
            Defaults to :func:`structural_summary`, which needs no provider. One
            that raises is logged and replaced; it can never cost a preserved
            field.

    Returns:
        A :class:`CompactedConversation` carrying every one of
        :data:`PRESERVED_FIELDS` that was present in ``state``, unchanged, and
        nothing else.
    """
    summary = _summarize(state, summarizer)

    preserved = {
        name: _freeze(state[name])
        for name in PRESERVED_FIELDS
        if state.get(name) is not None
    }

    compacted = CompactedConversation(summary=summary, **preserved)

    dropped = [name for name in state if name not in PRESERVED_FIELDS]
    if dropped:
        logger.debug(
            "compaction dropped %d field(s) outside the allow-list: %s",
            len(dropped),
            ", ".join(sorted(dropped)),
        )
    return compacted
