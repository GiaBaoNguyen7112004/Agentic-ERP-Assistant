"""Deciding what gets into the prompt, by measurement rather than by opinion.

The window is finite and the candidates are not, so every turn something is left
out. This module makes that a decision a reviewer can check: given the same
candidates and the same budget, it always produces the same plan, and every
exclusion carries a typed reason for why it happened.

Three rules do all the work
---------------------------

**Order first, decide second.** Candidates are sorted by descending
:attr:`~agentic_erp_assistant.context.candidate.ContextCandidate.priority` and
then by ``candidate_id`` before anything is included or excluded. A total order,
not merely a stable one: two candidates at the same priority must not swap places
because a caller built the list in a different order, or the plan stops being
reproducible for the most boring reason imaginable.

**Policy outranks the budget.** A candidate with ``allowed=False`` is excluded
``policy_denied`` before the budget is consulted at all. If the budget were
checked first, a denied candidate could be reported as ``budget_exceeded`` -- and
a reader of the trace would go looking for room that was never the problem. It
also means a denial cannot be bought: no amount of spare budget lets restricted
text in.

**An overflow is not a stopping point.** A candidate that would push the plan
over budget is excluded and the loop *continues*, so a small low-priority
candidate still gets its chance after a large high-priority one was refused.
First-fit descending, which is why :attr:`ContextPlan.used_tokens` can land
short of the budget with excluded candidates still in the plan.

What ``budget_tokens`` covers
-----------------------------

The builder measures candidate text and nothing else -- no chat framing, no role
names -- which is what makes :attr:`ContextPlan.used_tokens` independently
checkable: re-encode the included texts with ``tiktoken`` and you get the same
number.

That is deliberately *not* the number
:class:`~agentic_erp_assistant.llm.gateway.LLMGateway` compares against the
context window, which does include per-message framing. The two are not in
conflict; they measure different things. The caller derives this budget from the
window by subtracting what it already knows it will spend::

    budget_tokens = context_window - output_reserve - fixed_block_cost

where ``fixed_block_cost`` is the counted cost of the system, developer and user
blocks. The builder budgets the variable part, because the variable part is the
only part it is allowed to drop.

The gateway's own check stays where it is. This layer makes an overflow
survivable -- drop the least important candidates and answer anyway -- and the
gateway's :class:`~agentic_erp_assistant.llm.gateway.ContextWindowExceeded`
remains the last line of defense for when even that is not enough.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from agentic_erp_assistant.context.candidate import ContextCandidate
from agentic_erp_assistant.llm.tokenizer import TiktokenCounter, TokenCounter

__all__ = [
    "ContextBuilder",
    "ContextPlan",
    "ExcludedCandidate",
    "ExclusionReason",
]

logger = logging.getLogger(__name__)


ExclusionReason = Literal[
    "policy_denied",  # policy refused it; the budget was never consulted
    "budget_exceeded",  # it was permitted, and there was no room left for it
]
"""Why a candidate did not make it into the prompt.

Two reasons, kept apart on purpose. They call for opposite fixes -- retrieve less
versus change a policy -- and a single ``excluded`` list with no reason attached
would leave a reviewer unable to tell which of those a run is asking for.
"""


@dataclass(frozen=True)
class ExcludedCandidate:
    """A candidate that was left out, with the reason attached to it.

    A pair rather than two parallel lists: the reason has to travel *with* the
    candidate, or answering "why was this dropped?" means re-running the builder
    and hoping it decides the same way.
    """

    candidate: ContextCandidate
    """The candidate that was left out."""

    reason: ExclusionReason
    """Which rule left it out. See :data:`ExclusionReason`."""


@dataclass(frozen=True)
class ContextPlan:
    """What the prompt will contain, what it will not, and what it costs.

    Frozen, and holding tuples rather than lists. A frozen dataclass wrapping a
    mutable list is not frozen, it merely looks it -- the same reason
    :class:`~agentic_erp_assistant.reasoning.decision.ReasoningDecision` holds a
    tuple. This is the record the trace keeps of a decision that has already been
    made; editing it afterwards would make it a worse witness than no record.
    """

    included: tuple[ContextCandidate, ...]
    """What goes into the prompt, in the order the builder settled on."""

    excluded: tuple[ExcludedCandidate, ...]
    """What does not, each with its reason. Empty when everything fitted."""

    used_tokens: int
    """The token count of the included candidates' text.

    Measured, never accumulated from anything a caller supplied. Verifiable
    independently: encode each included candidate's ``text`` and sum.
    """

    budget_tokens: int
    """The budget the plan was made against. Kept so the plan explains itself."""

    @property
    def overflowed(self) -> bool:
        """Whether anything was dropped for want of room.

        The signal
        :func:`~agentic_erp_assistant.reasoning.decision.classify_failure` takes
        as ``budget_overflow``. Distinct from "nothing was excluded": a plan can
        exclude a great deal on policy grounds and still have had room to spare.
        """
        return any(item.reason == "budget_exceeded" for item in self.excluded)

    @property
    def remaining_tokens(self) -> int:
        """Budget left unspent. Never negative -- the builder never overspends."""
        return self.budget_tokens - self.used_tokens


@dataclass
class ContextBuilder:
    """Turns candidates and a budget into a plan, the same way every time.

    Usage::

        builder = ContextBuilder(model="gpt-4o")
        plan = builder.build(candidates, budget_tokens=4_000)
    """

    model: str
    """Whose tokenizer decides these counts. Required, with no default.

    A token count means nothing without the model that produced it -- the same
    text is a different number under a different encoding -- so there is no
    sensible value to default to. A default would be one model's arithmetic
    silently applied to whichever model the run actually uses, which is the exact
    failure the budget check exists to prevent, reintroduced one layer up. Pass
    the model you configured, the same one the client is bound to.
    """

    counter: TokenCounter = field(default_factory=TiktokenCounter)
    """How text is measured.

    The port from :mod:`agentic_erp_assistant.llm.tokenizer`, not a second
    implementation, and that is a decision worth stating: if this module counted
    tokens its own way, it and the gateway's pre-call budget check could disagree
    about the same text, and a trace showing a refused request would not say
    which of the two numbers refused it. One measuring authority, one answer.
    """

    def build(
        self,
        candidates: Iterable[ContextCandidate],
        budget_tokens: int,
    ) -> ContextPlan:
        """Choose what fits, and record what did not and why.

        Args:
            candidates: The contenders, in any order. The order they arrive in
                does not affect the result -- see the module docstring.
            budget_tokens: How many tokens of candidate text the prompt can
                afford. ``0`` is legitimate and means everything permitted is
                excluded ``budget_exceeded``.

        Returns:
            A :class:`ContextPlan` whose ``included`` and ``excluded`` together
            account for every candidate given, exactly once.

        Raises:
            ValueError: ``budget_tokens`` is negative, which is not a budget any
                caller can honestly have computed; or two candidates share a
                ``candidate_id``, which would make the plan ambiguous about which
                of them was kept.
        """
        if budget_tokens < 0:
            raise ValueError(
                f"budget_tokens: cannot be negative, got {budget_tokens}"
            )

        ordered = self._ordered(candidates)

        included: list[ContextCandidate] = []
        excluded: list[ExcludedCandidate] = []
        used = 0

        for candidate in ordered:
            # Policy first, and before any measurement: a denied candidate must
            # not be reported as one that failed to fit, whatever the budget is.
            if not candidate.allowed:
                excluded.append(ExcludedCandidate(candidate, "policy_denied"))
                continue

            cost = self.counter.count_tokens(candidate.text, model=self.model)
            if used + cost > budget_tokens:
                # Excluded, but the loop goes on: a smaller candidate further
                # down may still fit in what this one could not use.
                excluded.append(ExcludedCandidate(candidate, "budget_exceeded"))
                continue

            included.append(candidate)
            used += cost

        plan = ContextPlan(
            included=tuple(included),
            excluded=tuple(excluded),
            used_tokens=used,
            budget_tokens=budget_tokens,
        )

        if plan.overflowed:
            logger.info(
                "context budget %d reached: kept %d candidate(s) at %d tokens, "
                "dropped %d for want of room",
                budget_tokens,
                len(plan.included),
                plan.used_tokens,
                sum(1 for item in plan.excluded if item.reason == "budget_exceeded"),
            )
        return plan

    def _ordered(
        self, candidates: Iterable[ContextCandidate]
    ) -> list[ContextCandidate]:
        """Sort into the one order the builder is allowed to decide from.

        Descending priority, then ascending ``candidate_id``. The tiebreak is not
        decoration: without it two equal-priority candidates would be separated
        by whatever order the caller happened to build its list in, and the same
        inputs would produce different plans on different runs.

        Duplicate ids are rejected here rather than tolerated. A plan that
        reports ``candidate_id="mem-1"`` as included and also as excluded is not
        a record of anything, and a tiebreak on a non-unique key is not a total
        order in the first place.
        """
        listed = list(candidates)

        seen: set[str] = set()
        duplicates: set[str] = set()
        for candidate in listed:
            if candidate.candidate_id in seen:
                duplicates.add(candidate.candidate_id)
            seen.add(candidate.candidate_id)

        if duplicates:
            raise ValueError(
                f"candidate_id: must be unique within one plan; repeated: "
                f"{', '.join(sorted(duplicates))}"
            )

        return sorted(listed, key=lambda c: (-c.priority, c.candidate_id))
