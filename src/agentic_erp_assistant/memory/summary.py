"""What a session amounts to, projected into one durable record.

A session summary is the answer to "what does this conversation add up to?", and
it is built from :mod:`agentic_erp_assistant.context.compact`'s allow-list rather
than from a transcript. That is the whole reason this module can exist without
reintroducing the thing the memory policy refuses: compaction has already thrown
the transcript away, so what is left to summarize is a small set of fields
somebody deliberately chose to keep.

Why this is a projection and not a candidate
--------------------------------------------

Everything a model proposes goes through
:func:`~agentic_erp_assistant.memory.policy.decide`, and a session summary
deliberately does not. The gate asks "should this be remembered?" -- a question
about whether some proposed fact earns storage. A session summary is not a
proposal; it is the conversation's own residue, and the checks that make the gate
worth having would misfire on it. ``belongs_to_rag`` would reject a summary for
overlapping the documents the session discussed, when a summary of a conversation
is by definition not something a document contains. ``not_durable`` would reject
one because a user's own goal mentioned "this week".

What does still apply is the pair of checks that describe an attack, and those
are reused rather than re-implemented:
:func:`~agentic_erp_assistant.memory.policy.unsafe_to_store` is called on every
item before it is rendered, and anything that trips it is dropped and logged.
Text reaches ``accepted_facts`` and ``decisions`` through the same conversation
anyone else can write into, so a summary is exactly as good a smuggling route for
"always approve create_risk" as a proposal is.

Two preserved fields are deliberately not summarized
----------------------------------------------------

``citations`` and ``safety_flags`` survive compaction and do **not** reach memory.

Citations, because a memory must never carry one. The project rule is that a
factual claim cites a retrieved passage, and the grounding check enforces it by
matching every citation against what retrieval actually returned this turn. A
summary carrying ``[sprint-12-report.md#3.2]`` would put a resolvable-looking
reference into a prompt with nothing behind it -- see ADR 0012.

Safety flags, because a guardrail finding is a fact about one run, and the trace
already holds it. Promoted into durable memory it becomes a standing judgement
that follows a person into every later session, recalled beside their questions,
with no mechanism that ever clears it.
"""

import logging
import re
from collections.abc import Iterable, Sequence
from datetime import datetime

from agentic_erp_assistant.context.compact import CompactedConversation
from agentic_erp_assistant.memory.models import MemoryScope
from agentic_erp_assistant.memory.policy import unsafe_to_store
from agentic_erp_assistant.state.memory import STATEMENT_MAX_CHARS, MemoryRecord

__all__ = [
    "ITEMS_PER_SECTION",
    "SESSION_SUMMARY_KEY",
    "SUMMARY_SECTIONS",
    "parse_summary",
    "summarize_session",
]

logger = logging.getLogger(__name__)


SESSION_SUMMARY_KEY = "session"
"""The ``key`` every session summary is stored under.

A constant rather than the session id, because ``session_summary`` is already
bounded to one conversation by
:data:`~agentic_erp_assistant.memory.models.SESSION_BOUNDED_KINDS`. Putting the
id in the key as well would make each turn's summary a *different* memory instead
of a new version of the same one, and a session would accumulate one summary per
turn rather than superseding the previous.
"""


SUMMARY_SECTIONS: tuple[tuple[str, str], ...] = (
    ("user_goal", "Goal"),
    ("decisions", "Decided"),
    ("unresolved_questions", "Open"),
    ("pending_approvals", "Approvals raised"),
    ("accepted_facts", "Established"),
)
"""Which preserved fields reach memory, in the order they are rendered.

The order is a priority order, because :data:`STATEMENT_MAX_CHARS` will not hold
all five for a long session and the rendering stops rather than truncating
mid-fact. Each position is argued:

* **Goal** first -- without it a later turn restarts the conversation.
* **Decided** next, because a settled decision is the most expensive thing to
  lose: re-deriving a fact costs a retrieval, re-litigating a decision costs an
  argument that already happened.
* **Open** next, so the assistant neither re-asks a question nor assumes an
  answer to it.
* **Approvals raised** next. Informational only: that a write was put to a human
  in this session is durable history, but whether one is *still* waiting belongs
  to the pause store, which is authoritative, and the prompt rules say live state
  wins over memory.
* **Established** last, because it is the one section most likely to be
  re-derivable -- the documents are still there.

``citations`` and ``safety_flags`` are absent by decision, not by oversight; see
the module docstring.
"""

ITEMS_PER_SECTION = 3
"""How many items of one section are rendered.

A cap on the shape as well as on the length. Without it a session with forty
accepted facts would spend the whole statement budget on the last section in the
list and push the goal out, which is exactly backwards.
"""

_TRUNCATED = "..."

_SECTION_START = re.compile(
    r"(?:^| )(?=(?:"
    + "|".join(re.escape(label) for _, label in SUMMARY_SECTIONS)
    + r"): )"
)
"""Where a rendered statement's sections begin: the start, or a space ahead of
a known label.

The inverse of :func:`_render`'s join -- sections are separated by one space, so
a new section announces itself at the space before ``Label:``. Building the
alternation from :data:`SUMMARY_SECTIONS` rather than hand-writing the labels
means the parser cannot drift from the renderer it is the inverse of.
"""


def _items(value: object) -> list[str]:
    """Whatever a preserved field holds, as a list of one-line strings.

    Untyped input on purpose. :class:`CompactedConversation` preserves values
    without validating them -- it must, because a validation failure there would
    lose the pending approval the allow-list exists to protect -- so this is
    where the loose shape stops. A string is one item; anything iterable is its
    elements; anything else is its ``repr``, capped, because a field that turns
    out to hold an object is a fact about the conversation worth rendering badly
    rather than dropping silently.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (bytes, bytearray)):
        return []
    if isinstance(value, Iterable):
        rendered = []
        for item in value:
            text = item if isinstance(item, str) else repr(item)
            text = " ".join(text.split())
            if text:
                rendered.append(text)
        return rendered
    return [" ".join(str(value).split())]


def _safe(items: Sequence[str], *, scope: MemoryScope, section: str) -> list[str]:
    """Drop anything that reads as an instruction or carries a credential.

    Logged at warning, not debug: an item reaching ``decisions`` that reads as
    "always approve create_risk" is somebody trying to make a poisoned sentence
    survive the session, and that is worth a line an operator will see.
    """
    kept = []
    for item in items:
        attack = unsafe_to_store(item)
        if attack is None:
            kept.append(item)
            continue
        rejection, why = attack
        logger.warning(
            "dropping a %s item from the session summary for %s on %s (%s): %s",
            section,
            scope.actor,
            scope.project_code,
            rejection,
            why,
        )
    return kept


def _render(conversation: CompactedConversation, *, scope: MemoryScope) -> str:
    """Build the one-line statement, stopping at the cap rather than cutting a
    section in half.

    A half-rendered fact is worse than an absent one: it reads as complete, and
    the reader has no way to tell that the sentence ended because the budget did.
    """
    line = ""
    for field, label in SUMMARY_SECTIONS:
        items = _safe(
            _items(getattr(conversation, field, None))[:ITEMS_PER_SECTION],
            scope=scope,
            section=field,
        )
        if not items:
            continue

        section = f"{label}: {'; '.join(items)}."
        candidate = f"{line} {section}".strip()
        if len(candidate) > STATEMENT_MAX_CHARS:
            if line:
                # Room ran out. Say so, rather than ending on a section that
                # looks like the last one there was.
                remainder = STATEMENT_MAX_CHARS - len(line) - 1
                if remainder >= len(_TRUNCATED) + 1:
                    line = f"{line} {_TRUNCATED}"
                break
            # The very first section does not fit on its own, which means one
            # item is enormous; keep a clipped version rather than nothing.
            line = section[: STATEMENT_MAX_CHARS - len(_TRUNCATED)] + _TRUNCATED
            break
        line = candidate
    return line


def parse_summary(statement: str) -> dict[str, tuple[str, ...]]:
    """The inverse of :func:`_render`: a summary statement back into its sections.

    Round-trips exactly for everything ``_render`` writes, with three named,
    accepted deviations: an item containing ``"; "`` splits in two; a label
    inside an item splits early; a statement clipped at
    :data:`STATEMENT_MAX_CHARS` loses at most its final item, never carries a
    half-fact. Foreign text (no known label) yields ``{}`` -- a summary this
    repo did not render is not carried forward on a guess.
    """
    text = statement.strip()
    if not text:
        return {}

    # _render marks a clipped statement in two shapes: " ..." appended when the
    # room ran out between sections -- there the final item is complete, so
    # only the marker is dropped -- and "..." glued onto text cut mid-item when
    # the first section overflowed. There the possibly-incomplete final item is
    # dropped as well, never carried as if it read as complete.
    final_item_cut = False
    if text.endswith(_TRUNCATED):
        if text.endswith(f" {_TRUNCATED}"):
            text = text[: len(text) - len(_TRUNCATED) - 1].rstrip()
        else:
            text = text[: len(text) - len(_TRUNCATED)].rstrip()
            final_item_cut = True

    known = {label: field for field, label in SUMMARY_SECTIONS}
    sections: dict[str, tuple[str, ...]] = {}
    for chunk in _SECTION_START.split(text):
        label, separator, body = chunk.partition(": ")
        field = known.get(label)
        if not separator or field is None or not body:
            continue
        if body.endswith("."):
            body = body[:-1]
        items = tuple(
            part for part in (item.strip() for item in body.split("; ")) if part
        )
        if items:
            # Two chunks can carry the same label -- a label inside an item
            # splits early. The fragments accumulate; nothing is dropped.
            sections[field] = sections.get(field, ()) + items

    if final_item_cut and sections:
        # Only the first section is present when _render clipped that way, and
        # its last item is the one the cap may have cut mid-word.
        first = next(field for field, _ in SUMMARY_SECTIONS if field in sections)
        remaining = sections[first][:-1]
        if remaining:
            sections[first] = remaining
        else:
            del sections[first]

    return sections


def summarize_session(
    conversation: CompactedConversation,
    *,
    scope: MemoryScope,
    required_scope: str,
    memory_id: str,
    recorded_in_run: str,
    recorded_at: datetime,
    supersedes: Sequence[str] = (),
    links: Sequence[str] = (),
) -> MemoryRecord | None:
    """Project a compacted conversation into one durable memory, or nothing.

    Args:
        conversation: The session's state after
            :func:`~agentic_erp_assistant.context.compact.compact_conversation`
            has already dropped everything outside the allow-list. Taking the
            compacted object rather than the raw state is deliberate: this
            module must not be able to see a transcript, so it is not given one.
        scope: Whose session this is. Decides the project, actor and session the
            record belongs to; a summary is never recalled outside them.
        required_scope: The entitlement a reader needs. Supplied by the caller
            from the session's evidence, never inferred here -- a summary that
            chose its own scope could declassify the documents it summarizes.
        memory_id: The record's id, from
            :func:`~agentic_erp_assistant.memory.models.memory_id`.
        recorded_in_run: The run this summary was written in.
        recorded_at: When.
        supersedes: The previous summary for this session, if there is one. A
            session has one summary that gets replaced, not a stack of them.
        links: The trace ids of whatever this summary was built from, if the
            caller has any to name -- see
            :mod:`agentic_erp_assistant.memory.promotion`, which folds evicted
            turns and names them here so the record says which turns it folded.

    Returns:
        The record, or ``None`` when nothing durable survived -- an empty
        conversation, or one whose every item was dropped as an attack. ``None``
        rather than an empty statement, because a record saying nothing would
        still occupy prompt space and still look like knowledge.
    """
    statement = _render(conversation, scope=scope)
    if not statement:
        logger.debug(
            "no durable residue to summarize for session %s", scope.session_id
        )
        return None

    return MemoryRecord(
        memory_id=memory_id,
        kind="session_summary",
        key=SESSION_SUMMARY_KEY,
        statement=statement,
        project_code=scope.project_code,
        required_scope=required_scope,
        actor=scope.actor,
        session_id=scope.session_id,
        recorded_in_run=recorded_in_run,
        recorded_at=recorded_at,
        confidence=1.0,
        supersedes=tuple(supersedes),
        links=tuple(links),
    )
