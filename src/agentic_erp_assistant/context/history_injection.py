"""Which turns of this session's short-term window reach the prompt.

:mod:`agentic_erp_assistant.context.memory_injection` decides which *durable*
memories are worth this turn's tokens. This module answers the equivalent
question for the short-term window: given everything a session's store holds,
which recent turns are shown, in what shape, and at what cost.

The shape is decided here, not left to the store
--------------------------------------------------

A turn's own request and reply can be arbitrarily long, and this project's
answers carry a ``Sources: [doc#locator]`` trailer (see
:func:`~agentic_erp_assistant.engine.nodes._with_sources`). Neither belongs in
the history role verbatim: a long turn would crowd out the rest of the window,
and a citation tag surviving into history would make it look like a source
:mod:`agentic_erp_assistant.llm.prompts` warns the model it is not. So every
turn is stripped and clipped *before* it competes for the budget, and what
:class:`HistorySelection` returns is exactly what the prompt will show -- the
same discipline :attr:`~agentic_erp_assistant.state.agent_state.AgentState.history`
is held to.

Why the window's size is proven, not assumed
---------------------------------------------

:data:`HISTORY_TURN_LIMIT` turns must never silently lose one to the budget --
a session that quietly gets in a shorter window than the code promises is a
regression a reviewer would have no way to notice. So the four constants below
are declared once and a test asserts the arithmetic that makes every turn
inside the limit fit; a change to any of them must keep that test passing.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_erp_assistant.context.builder import ContextBuilder, ContextPlan
from agentic_erp_assistant.context.candidate import ContextCandidate
from agentic_erp_assistant.llm.tokenizer import TokenCounter, split_into_token_windows
from agentic_erp_assistant.state.conversation import ConversationTurn

__all__ = [
    "CLIP_MARKER",
    "HISTORY_BUDGET_TOKENS",
    "HISTORY_TURN_LIMIT",
    "HistorySelection",
    "REQUEST_MAX_TOKENS",
    "RESPONSE_MAX_TOKENS",
    "TURN_FRAMING_TOKENS",
    "clip_to_tokens",
    "select_history",
    "strip_citations",
]


HISTORY_TURN_LIMIT = 6
"""How many of the session's most recent turns the window holds.

Turns older than this are what
:mod:`agentic_erp_assistant.memory.promotion` folds into the session summary
-- the window and the promotion path are two ends of the same pipe, not two
independent policies.
"""

REQUEST_MAX_TOKENS = 80
"""The most one turn's request may cost, after clipping."""

RESPONSE_MAX_TOKENS = 160
"""The most one turn's reply may cost, after stripping and clipping."""

TURN_FRAMING_TOKENS = 16
"""Headroom per turn for the numbering and labels
:func:`~agentic_erp_assistant.llm.prompts._render_history` adds -- "N. User: ",
"\\n   Assistant: " -- which are not part of any candidate's measured text but
do reach the prompt."""

HISTORY_BUDGET_TOKENS = 1600
"""How many tokens of history the prompt can afford.

Sized so every turn inside :data:`HISTORY_TURN_LIMIT` fits with room to spare,
not merely on average -- see the test that pins
``HISTORY_TURN_LIMIT * (REQUEST_MAX_TOKENS + RESPONSE_MAX_TOKENS +
TURN_FRAMING_TOKENS) <= HISTORY_BUDGET_TOKENS``. The budget is never the reason
a turn inside the window goes missing; only :data:`HISTORY_TURN_LIMIT` decides
that.
"""

CLIP_MARKER = "..."
"""Appended to a request or reply that had to be cut, so a reader -- human or
model -- can tell a clipped turn from a short one."""


_SOURCES_TRAILER = re.compile(r"\n*Sources: .*\Z", re.DOTALL)
"""What :func:`~agentic_erp_assistant.engine.nodes._with_sources` appends.

A local copy of the pattern rather than an import: importing from ``engine``
here would invert the dependency direction the rest of this layer keeps
(``context`` sits below ``engine``, never above it). A test proves this
pattern undoes what that function does.
"""

_CITATION_TAG = re.compile(r"\[[^\[\]\n#]+#[^\[\]\n]+\]")
"""One evidence-style tag: ``[source_id#locator]``. The locator half may
contain spaces -- a CSV row (``row R-2``) or a split section (``§3.2 part
1``), see ADR 0009 and ``evidence/rag/retrieval-report.json`` -- so only
``[``, ``]``, and line breaks are excluded, matching what
:class:`~agentic_erp_assistant.state.evidence.EvidenceSnippet` actually
forbids on construction. A stricter pattern that rejected whitespace let a
real tag with a spaced locator survive inline into the ``history`` role,
which ADR 0014 says must never carry anything citation-shaped."""


def strip_citations(text: str) -> str:
    """Remove anything shaped like a citation from a prior turn's reply.

    Two passes, in order: the ``Sources: ...`` trailer first, because leaving
    it for the tag pattern to chew through piecemeal could strand a bare
    ``[doc#`` fragment; then any tag appearing inline in the body itself.
    Whitespace is collapsed last, for the reason every renderer in
    :mod:`agentic_erp_assistant.llm.prompts` collapses it: removing a tag
    leaves a gap, and a gap containing a line break could render as an extra
    numbered entry.

    Args:
        text: A turn's reply, exactly as it was produced.

    Returns:
        The same text with no ``[``, ``]``, or ``Sources:`` trailer left in it.
    """
    text = _SOURCES_TRAILER.sub("", text)
    text = _CITATION_TAG.sub("", text)
    return " ".join(text.split())


def clip_to_tokens(text: str, *, model: str, max_tokens: int) -> str:
    """Cut ``text`` to at most ``max_tokens``, by the one tokenizer authority.

    Token-based rather than character-based (ADR 0001): a character budget
    would clip a short Vietnamese sentence far short of its token cost and
    leave a long English one untouched, for a project whose own fixtures are
    bilingual.

    Args:
        text: What to clip. Returned unchanged if it already fits.
        model: Whose encoding decides where the cut falls.
        max_tokens: The hard ceiling.

    Returns:
        ``text`` unchanged when it already fits ``max_tokens``; otherwise its
        first ``max_tokens`` tokens with :data:`CLIP_MARKER` appended.
    """
    if not text:
        return text
    windows = split_into_token_windows(text, model=model, window_tokens=max_tokens)
    if not windows:
        return text
    if len(windows) == 1:
        return windows[0]
    return f"{windows[0]}{CLIP_MARKER}"


@dataclass(frozen=True)
class HistorySelection:
    """What the window chose, in the shape the prompt will show it, and what it cost.

    Frozen and holding tuples, for the same reason
    :class:`~agentic_erp_assistant.context.memory_injection.MemorySelection` is:
    this is the record of a decision already made, and the trace's answer to
    "what history was this turn shown, and why?".
    """

    selected: tuple[ConversationTurn, ...]
    """What goes into the prompt, oldest first -- already clipped and stripped,
    so this is exactly what :func:`~agentic_erp_assistant.llm.prompts._render_history`
    will render, not a superset of it."""

    dropped: tuple[ConversationTurn, ...]
    """Turns inside the window the budget still could not afford.

    Expected to be empty in ordinary operation -- see :data:`HISTORY_BUDGET_TOKENS`
    -- and kept rather than asserted away, because a caller that changed the
    constants without re-running the invariant test deserves to see this
    non-empty rather than silently lose a turn.
    """

    plan: ContextPlan
    """The builder's own record of what fitted and why."""


def _clipped_copy(turn: ConversationTurn, *, model: str) -> ConversationTurn:
    """This turn, with its request and reply cut down to what the prompt will show.

    Rebuilt through the constructor rather than assigned to, the same rule
    :meth:`~agentic_erp_assistant.state.agent_state.AgentState.evolve` follows:
    every invariant on :class:`ConversationTurn` applies to the copy exactly as
    it applies to the original.
    """
    request = clip_to_tokens(
        " ".join(turn.request.split()), model=model, max_tokens=REQUEST_MAX_TOKENS
    )
    response = turn.response
    if response is not None:
        response = clip_to_tokens(
            strip_citations(response), model=model, max_tokens=RESPONSE_MAX_TOKENS
        )
    current = {name: getattr(turn, name) for name in type(turn).model_fields}
    return type(turn)(**{**current, "request": request, "response": response})


def _line(turn: ConversationTurn) -> str:
    """The text one turn contributes to the budget: what was asked, then answered.

    Deliberately not :func:`~agentic_erp_assistant.llm.prompts._render_history`'s
    own numbered format -- the numbering depends on a turn's final position
    among its neighbours, which is not known until the budget has already
    chosen who is in the block. :data:`TURN_FRAMING_TOKENS` accounts for that
    small, constant difference.
    """
    request = turn.request
    if turn.response is not None:
        reply = turn.response
    elif turn.paused:
        reply = f"(waiting for approval to run {turn.tool_name})"
    else:
        reply = f"({turn.failure})"
    return f"User: {request}\nAssistant: {reply}"


def select_history(
    turns: Sequence[ConversationTurn],
    *,
    budget_tokens: int,
    model: str,
    counter: TokenCounter | None = None,
) -> HistorySelection:
    """Choose the recent turns worth this turn's tokens, newest first.

    Args:
        turns: Everything the store holds for this session and actor, in any
            order. Defensive on purpose -- :data:`HISTORY_TURN_LIMIT` is
            re-applied here even though a well-behaved store already bounds
            what it returns.
        budget_tokens: How many tokens of history the prompt can afford. See
            :data:`HISTORY_BUDGET_TOKENS`.
        model: Whose tokenizer measures that. Required, with no default, for
            the reason :attr:`~agentic_erp_assistant.context.builder.ContextBuilder.model`
            is.
        counter: How to measure. Defaults to the builder's own.

    Returns:
        A :class:`HistorySelection` whose ``selected`` are clipped, tag-free
        copies of the newest turns that fit, in chronological order.
    """
    ordered = sorted(turns, key=lambda turn: (turn.started_at, turn.trace_id))
    windowed = ordered[-HISTORY_TURN_LIMIT:] if len(ordered) > HISTORY_TURN_LIMIT else ordered

    clipped = [_clipped_copy(turn, model=model) for turn in windowed]
    turn_count = len(clipped)

    candidates = [
        ContextCandidate(
            candidate_id=turn.trace_id,
            kind="history",
            text=_line(turn),
            # age_rank is 0 for the newest turn and grows for older ones, so
            # the newest always outranks an older turn regardless of length.
            priority=10_000 - (turn_count - 1 - position),
        )
        for position, turn in enumerate(clipped)
    ]

    builder = ContextBuilder(model=model, **({"counter": counter} if counter else {}))
    plan = builder.build(candidates, budget_tokens=budget_tokens)

    included_ids = {candidate.candidate_id for candidate in plan.included}
    selected = tuple(turn for turn in clipped if turn.trace_id in included_ids)
    dropped = tuple(turn for turn in clipped if turn.trace_id not in included_ids)

    return HistorySelection(selected=selected, dropped=dropped, plan=plan)
