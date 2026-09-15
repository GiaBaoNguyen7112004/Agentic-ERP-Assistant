"""What turns leaving the short-term window are worth to the rest of a session.

:mod:`agentic_erp_assistant.context.compact` and
:mod:`agentic_erp_assistant.memory.summary` were built for this and never
wired -- see :mod:`agentic_erp_assistant.memory.service`'s own docstring:
"Nothing produces that state yet; the compaction allow-list is fed by a caller
that does not exist." This module is that caller. A turn evicted from
:class:`~agentic_erp_assistant.memory.conversation.ConversationMemory`'s window
is folded here into the session's one ``session_summary`` record, through the
same allow-list and the same per-item safety check every other summary goes
through.

The model proposes; code decides -- and code also carries
-----------------------------------------------------------

:class:`LLMSessionSummaryProposer` asks a model what the evicted turns are
worth, exactly as
:class:`~agentic_erp_assistant.memory.extractor.LLMMemoryProposer` asks what a
finished turn is worth. Its answer is never trusted directly:
:func:`conversation_state` overlays only four named fields onto what code
alone can already say about the turns (:func:`structural_state`), and
``pending_approvals`` -- whether a write is still waiting on a human -- is
never taken from the model at all. And because the contract asks the model
only for the *delta* -- what these turns add, never a restatement of what the
session already wrote down -- :func:`conversation_state` also carries the
session's previous summary forward, parsing its statement back into sections
(:func:`~agentic_erp_assistant.memory.summary.parse_summary`) and merging it
beneath what this batch contributes. Without that, a proposal of nothing
would be the one thing a summary could not survive: the model is told to
leave the goal unset when the turns did not change it, and code must make
that true. The result is a plain mapping that
:func:`~agentic_erp_assistant.context.compact.compact_conversation` then
narrows to its own allow-list, and
:func:`~agentic_erp_assistant.memory.summary.summarize_session` still runs
:func:`~agentic_erp_assistant.memory.policy.unsafe_to_store` over every item
before it reaches the statement -- carried items included, so a poisoned
sentence inside a previous statement cannot survive being folded again. A
proposer that is down, or not configured at all, degrades to the structural
half alone -- a summary is still written, just a plainer one, and the
previous one is still carried forward.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import Field, ValidationError

from agentic_erp_assistant.llm.prompts import build_promotion_messages
from agentic_erp_assistant.llm.tools import StrictArguments, ToolSpec
from agentic_erp_assistant.memory.extractor import ProposalModel
from agentic_erp_assistant.memory.summary import parse_summary
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.memory import STATEMENT_MAX_CHARS, MemoryRecord

__all__ = [
    "LLMSessionSummaryProposer",
    "MAX_ITEMS_PER_SECTION",
    "PROPOSABLE_SECTIONS",
    "PROPOSE_SESSION_SUMMARY_TOOL",
    "SessionSummaryProposal",
    "SessionSummaryProposerPort",
    "conversation_state",
    "structural_state",
]

logger = logging.getLogger(__name__)


PROPOSABLE_SECTIONS = ("user_goal", "decisions", "unresolved_questions", "accepted_facts")
"""The only fields a model's proposal may change.

Deliberately not five: ``pending_approvals`` is absent, and that absence is
the whole point of keeping this list explicit rather than reading it off
:class:`SessionSummaryProposal`'s fields. Whether a write is still waiting on
a human is a fact about the turns themselves, derived in
:func:`structural_state`, never a model's summary of them.
"""

MAX_ITEMS_PER_SECTION = 3
"""How many items one section may carry after the proposal is merged in.

Matches :data:`~agentic_erp_assistant.memory.summary.ITEMS_PER_SECTION`, which
would cap it again downstream regardless -- capping here as well means a
structural item is never pushed out by the proposal's items filling the
section first, which is possible only because both are trimmed to the same
number rather than one trusting the other not to overflow.
"""


class SessionSummaryProposal(StrictArguments):
    """What a model thinks the evicted turns are worth, before code overlays it.

    Every field is required, even though each may be empty or ``null`` --
    strict function calling has no notion of an optional property, so "there is
    nothing here" has to be said, not omitted. See
    :class:`~agentic_erp_assistant.memory.extractor.MemoryProposal` for the
    same shape of argument.
    """

    user_goal: str | None = Field(
        min_length=1,
        max_length=STATEMENT_MAX_CHARS,
        description=(
            "What the person in these turns was trying to accomplish, in "
            "their own words. null if these turns did not change what the "
            "session is working on."
        ),
    )
    decisions: list[str] = Field(
        max_length=MAX_ITEMS_PER_SECTION,
        description=(
            "Something these turns settled that a later turn must not "
            "re-litigate. Empty if these turns settled nothing."
        ),
    )
    unresolved_questions: list[str] = Field(
        max_length=MAX_ITEMS_PER_SECTION,
        description=(
            "Something these turns left open. Empty if nothing is outstanding."
        ),
    )
    accepted_facts: list[str] = Field(
        max_length=MAX_ITEMS_PER_SECTION,
        description=(
            "Something these turns themselves established, with no document "
            "or tool behind it -- never something a document said, since that "
            "is still retrievable and citing it from memory would be a claim "
            "with no citation. Empty if nothing qualifies."
        ),
    )


PROPOSE_SESSION_SUMMARY_TOOL = ToolSpec(
    name="propose_session_summary",
    description=(
        "Record what the turns now leaving the short-term window are worth to "
        "the rest of this session. Call it exactly once. Every field empty or "
        "null is the normal answer for a batch that settled nothing durable."
    ),
    arguments=SessionSummaryProposal,
    mutating=False,
)
"""The only function this module offers.

Absent from ``DEFAULT_TOOLS`` and ``PLANNING_TOOLS``, for the same reason
:data:`~agentic_erp_assistant.memory.extractor.PROPOSE_MEMORIES_TOOL` is: this
is asked in a prompt of its own, about turns that have already ended, and a
model offered it mid-turn would eventually summarize instead of acting.
"""


@runtime_checkable
class SessionSummaryProposerPort(Protocol):
    """What promotion assumes about whatever nominates a session summary."""

    def propose(
        self, turns: Sequence[ConversationTurn], *, previous: MemoryRecord | None
    ) -> SessionSummaryProposal | None:
        """What the evicted ``turns`` are worth, or ``None`` to say nothing.

        Args:
            turns: The turns leaving the window, oldest first. Never empty --
                the caller does not ask about an empty promotion.
            previous: The session's current summary, if it has one, so the
                model can extend or correct it rather than starting over.

        Returns:
            A proposal, or ``None`` when the model said nothing is worth
            keeping -- an ordinary answer, not a failure.

        Raises:
            Exception: Whatever the underlying call raises. The caller treats
                a raise here exactly like ``None``: fold the turns
                structurally instead of failing the turn that triggered it.
        """
        ...


@dataclass(frozen=True)
class LLMSessionSummaryProposer:
    """Asks the model, and turns its one call into a proposal or nothing.

    Satisfies :class:`SessionSummaryProposerPort` structurally, the same
    arrangement :class:`~agentic_erp_assistant.memory.extractor.LLMMemoryProposer`
    uses.
    """

    model: ProposalModel
    """Where the proposal comes from. The gateway, so the budget check, the
    retry engine and the cost record all apply to this call as they do to a
    routing one."""

    tools: tuple[ToolSpec, ...] = (PROPOSE_SESSION_SUMMARY_TOOL,)
    """What to offer. Injectable so a test can narrow the menu."""

    def propose(
        self, turns: Sequence[ConversationTurn], *, previous: MemoryRecord | None
    ) -> SessionSummaryProposal | None:
        result = self.model.call_tools(
            build_promotion_messages(turns, previous),
            tools=self.tools,
            temperature=0.0,
        )

        if result.tool_name is None:
            logger.info(
                "promotion of %d turn(s): the proposer answered in prose "
                "instead of calling %s; folding structurally",
                len(turns),
                PROPOSE_SESSION_SUMMARY_TOOL.name,
            )
            return None
        if result.tool_name != PROPOSE_SESSION_SUMMARY_TOOL.name:
            logger.warning(
                "promotion of %d turn(s): the proposer called %r, which it "
                "was not offered",
                len(turns),
                result.tool_name,
            )
            return None

        try:
            validated = PROPOSE_SESSION_SUMMARY_TOOL.validate_arguments(
                result.arguments or {}
            )
        except ValidationError as error:
            logger.warning(
                "promotion of %d turn(s): %s was called with arguments it "
                "does not accept (%d problem(s)); folding structurally",
                len(turns),
                PROPOSE_SESSION_SUMMARY_TOOL.name,
                error.error_count(),
            )
            return None

        return validated  # type: ignore[return-value]


def structural_state(turns: Sequence[ConversationTurn]) -> dict[str, object]:
    """What code alone can say about turns leaving the window, no model involved.

    The floor :func:`conversation_state` overlays a proposal onto, and the
    whole answer when there is no proposer, or the proposer failed or said
    nothing. Three fields, each derived rather than asked for:

    * ``user_goal`` -- the oldest evicted turn's own request. The best guess
      code can make at what the batch was about, without inventing a summary
      of it. A fallback, not a default: :func:`conversation_state` uses it
      only when neither the proposal nor the previous summary names a goal.
    * ``pending_approvals`` -- named here, not in a proposal, because whether a
      write is still waiting on a human is a fact this code already has and a
      model restating it adds a chance of getting it wrong. Ordinarily empty:
      a paused turn is never evicted (see
      :mod:`agentic_erp_assistant.memory.conversation`), so this only fires for
      a turn that was approved or denied since it was last read but whose
      settlement has not yet reached here.
    * ``unresolved_questions`` -- the request of every turn that ended in
      ``clarify``, so a question the assistant itself asked and never got
      back to is not silently dropped when its turn leaves the window.
    """
    if not turns:
        return {}

    oldest = min(turns, key=lambda turn: (turn.started_at, turn.trace_id))
    state: dict[str, object] = {"user_goal": " ".join(oldest.request.split())}

    pending_approvals = [
        f"{turn.tool_name} awaiting approval (run {turn.trace_id})"
        for turn in turns
        if turn.paused
    ]
    if pending_approvals:
        state["pending_approvals"] = pending_approvals

    unresolved = [
        " ".join(turn.request.split()) for turn in turns if turn.route == "clarify"
    ]
    if unresolved:
        state["unresolved_questions"] = unresolved

    return state


def conversation_state(
    turns: Sequence[ConversationTurn],
    proposal: SessionSummaryProposal | None,
    previous: MemoryRecord | None = None,
) -> dict[str, object]:
    """The mapping :func:`~agentic_erp_assistant.context.compact.compact_conversation`
    will narrow to its allow-list: structure first, the proposal and the
    previous summary overlaid on top.

    Only :data:`PROPOSABLE_SECTIONS` can move. ``pending_approvals`` is never
    touched here, whatever the proposal says -- see :func:`structural_state` --
    and it is never carried either: a superseded summary must not resurrect a
    settled approval, so it is re-derived from the turns every promotion.

    Args:
        proposal: What the model proposed about this batch, or ``None`` when
            there is no proposer, it failed, or it said nothing. All three are
            ordinary, not failures.
        previous: The session's current summary, if it has one. Its parsed
            statement is what a proposal of nothing still carries forward;
            without it an empty proposal would be the one thing a summary
            could not survive.

    Precedence and the sliding window
    ---------------------------------

    ``user_goal``: the proposal's, else the previous summary's, else the
    structural guess -- the oldest evicted request, which is the right answer
    only for a session's first promotion, where there is nothing to carry.

    Each list field merges ``[*structural (this batch), *proposed,
    *previous]``, whitespace-collapsed, deduplicated preserving first
    occurrence, capped at :data:`MAX_ITEMS_PER_SECTION`. Newest content
    enters at the front, so when a section saturates the **oldest** previous
    item leaves: a sliding window, the honest semantics for a record capped
    at 400 characters. Items that were dropped as attacks before they were
    rendered are not re-parsed as facts; whatever survives parsing is checked
    again by ``summarize_session`` before it reaches the statement.
    """
    carried = parse_summary(previous.statement) if previous is not None else {}
    state = structural_state(turns)

    goal = proposal.user_goal if proposal is not None and proposal.user_goal else None
    if goal is None:
        carried_goal = carried.get("user_goal") or ()
        goal = carried_goal[0] if carried_goal else None
    if goal is None and state.get("user_goal"):
        goal = state["user_goal"]
    if goal is not None:
        state["user_goal"] = goal

    for field in ("decisions", "unresolved_questions", "accepted_facts"):
        structural = list(state.get(field, []))
        proposed = list(getattr(proposal, field)) if proposal is not None else []
        previous_items = list(carried.get(field) or ())

        merged: list[str] = []
        for item in [*structural, *proposed, *previous_items]:
            text = " ".join(item.split()) if isinstance(item, str) else item
            if text and text not in merged:
                merged.append(text)
        if merged:
            state[field] = merged[:MAX_ITEMS_PER_SECTION]

    return state
