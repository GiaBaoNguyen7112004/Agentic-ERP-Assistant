"""The plan is a measurement, so the tests measure it independently.

Two habits run through this file. Expected token counts are computed here with
``tiktoken`` directly, never copied from what the builder reported -- a test that
asks the builder to confirm its own arithmetic proves only that it is
self-consistent. And the assertions are about *reasons*, not just about who
survived: "excluded" without "policy_denied" would pass equally well for a
candidate that was refused and one that merely did not fit, which are the two
outcomes this module exists to keep apart.
"""

import tiktoken

from agentic_erp_assistant.context import (
    ContextBuilder,
    ContextCandidate,
    ContextPlan,
)
from agentic_erp_assistant.llm.tokenizer import FALLBACK_ENCODING

import pytest

MODEL = "gpt-4o"
UNKNOWN_MODEL = "gpt-9-imaginary-2099"


def tokens(text: str, *, model: str = MODEL) -> int:
    """What tiktoken itself says, with no help from the code under test."""
    return len(tiktoken.encoding_for_model(model).encode(text))


def candidate(
    candidate_id: str,
    text: str,
    *,
    priority: int = 0,
    kind: str = "evidence",
    allowed: bool = True,
) -> ContextCandidate:
    return ContextCandidate(
        candidate_id=candidate_id,
        kind=kind,
        text=text,
        priority=priority,
        allowed=allowed,
    )


def excluded_reason(plan: ContextPlan, candidate_id: str) -> str | None:
    """The reason ``candidate_id`` was dropped, or ``None`` if it was not."""
    for item in plan.excluded:
        if item.candidate.candidate_id == candidate_id:
            return item.reason
    return None


def included_ids(plan: ContextPlan) -> list[str]:
    return [c.candidate_id for c in plan.included]


# The worked example, kept in one place because several tests below reason about
# it. The token counts in the comments are real: 'policy note' and 'old
# preference' encode to 2, 'milestone status' to 3, under gpt-4o and under the
# o200k_base fallback alike.
def _worked_example() -> list[ContextCandidate]:
    return [
        candidate("mem-1", "old preference", priority=10, kind="memory"),  # 2
        candidate("pol-1", "policy note", priority=40, kind="policy"),  # 2
        candidate("doc-2", "milestone status", priority=20),  # 3
        candidate("doc-9", "secret document", priority=30, allowed=False),  # 2
    ]


def test_the_worked_example_lands_exactly_where_it_should() -> None:
    """policy + status fill the budget; restricted is denied, memory misses out."""
    plan = ContextBuilder(model=MODEL).build(_worked_example(), budget_tokens=5)

    assert included_ids(plan) == ["pol-1", "doc-2"]
    assert plan.used_tokens == 5
    assert excluded_reason(plan, "doc-9") == "policy_denied"
    assert excluded_reason(plan, "mem-1") == "budget_exceeded"


def test_a_denied_candidate_stays_out_when_there_is_ample_budget() -> None:
    """The named acceptance criterion.

    ``doc-9`` outranks everything that got in and the budget is far larger than
    the whole set, so the only thing that can keep it out is the policy check
    running before the budget one.
    """
    plan = ContextBuilder(model=MODEL).build(_worked_example(), budget_tokens=10_000)

    assert "doc-9" not in included_ids(plan)
    assert excluded_reason(plan, "doc-9") == "policy_denied"
    assert plan.remaining_tokens > 0


def test_a_denied_candidate_is_never_blamed_on_the_budget() -> None:
    """Even with no budget at all, a denial is reported as a denial."""
    plan = ContextBuilder(model=MODEL).build(_worked_example(), budget_tokens=0)

    assert excluded_reason(plan, "doc-9") == "policy_denied"


def test_used_tokens_is_the_real_count_of_the_included_text() -> None:
    """The named acceptance criterion, verified against tiktoken directly."""
    plan = ContextBuilder(model=MODEL).build(_worked_example(), budget_tokens=5)

    assert plan.used_tokens == sum(tokens(c.text) for c in plan.included)
    assert plan.used_tokens == tokens("policy note") + tokens("milestone status")


def test_the_same_input_always_produces_the_same_plan() -> None:
    """What good looks like: re-running is a measurement, not a coin flip."""
    builder = ContextBuilder(model=MODEL)

    first = builder.build(_worked_example(), budget_tokens=5)
    second = builder.build(_worked_example(), budget_tokens=5)

    assert first == second


def test_the_order_candidates_arrive_in_does_not_change_the_plan() -> None:
    builder = ContextBuilder(model=MODEL)
    forwards = _worked_example()
    backwards = list(reversed(_worked_example()))

    assert builder.build(forwards, budget_tokens=5) == builder.build(
        backwards, budget_tokens=5
    )


def test_equal_priorities_are_broken_by_candidate_id_not_by_arrival() -> None:
    """Only one of the two fits, and it must always be the same one."""
    budget = tokens("alpha")
    pair = [
        candidate("b-second", "bravo", priority=5),
        candidate("a-first", "alpha", priority=5),
    ]

    plan = ContextBuilder(model=MODEL).build(pair, budget_tokens=budget)

    assert included_ids(plan) == ["a-first"]
    assert excluded_reason(plan, "b-second") == "budget_exceeded"


def test_a_lower_priority_candidate_is_still_tried_after_an_overflow() -> None:
    """First-fit descending: one refusal does not end the loop.

    ``big`` outranks ``small`` and does not fit. If the builder stopped there,
    ``small`` would be lost too, and the plan would waste room it was given.
    """
    small_text = "tiny"
    budget = tokens(small_text)
    candidates = [
        candidate("big", "a considerably longer passage than the budget", priority=90),
        candidate("small", small_text, priority=1),
    ]

    plan = ContextBuilder(model=MODEL).build(candidates, budget_tokens=budget)

    assert included_ids(plan) == ["small"]
    assert excluded_reason(plan, "big") == "budget_exceeded"
    assert plan.used_tokens == budget


def test_a_candidate_that_fills_the_budget_exactly_still_fits() -> None:
    """The comparison is strictly greater, so equality is not a second margin."""
    text = "milestone status"
    plan = ContextBuilder(model=MODEL).build(
        [candidate("doc-2", text)], budget_tokens=tokens(text)
    )

    assert included_ids(plan) == ["doc-2"]
    assert plan.remaining_tokens == 0


def test_every_candidate_is_accounted_for_exactly_once() -> None:
    plan = ContextBuilder(model=MODEL).build(_worked_example(), budget_tokens=5)

    accounted = included_ids(plan) + [
        item.candidate.candidate_id for item in plan.excluded
    ]

    assert sorted(accounted) == ["doc-2", "doc-9", "mem-1", "pol-1"]


def test_overflowed_distinguishes_a_denial_from_a_shortage() -> None:
    """A plan can exclude a lot on policy grounds and still have had room."""
    builder = ContextBuilder(model=MODEL)

    denied_only = builder.build(
        [candidate("doc-9", "secret document", allowed=False)], budget_tokens=100
    )
    assert denied_only.excluded
    assert not denied_only.overflowed

    assert builder.build(_worked_example(), budget_tokens=5).overflowed


def test_an_empty_candidate_list_is_a_valid_empty_plan() -> None:
    plan = ContextBuilder(model=MODEL).build([], budget_tokens=100)

    assert plan.included == ()
    assert plan.excluded == ()
    assert plan.used_tokens == 0
    assert not plan.overflowed


def test_a_zero_budget_excludes_everything_permitted() -> None:
    plan = ContextBuilder(model=MODEL).build(_worked_example(), budget_tokens=0)

    assert plan.included == ()
    assert plan.used_tokens == 0
    assert plan.overflowed


def test_a_negative_budget_is_a_caller_bug() -> None:
    with pytest.raises(ValueError, match="budget_tokens"):
        ContextBuilder(model=MODEL).build([], budget_tokens=-1)


def test_duplicate_candidate_ids_are_refused_before_anything_is_planned() -> None:
    """A plan that reports one id twice is not a record of anything."""
    clash = [candidate("doc-1", "alpha"), candidate("doc-1", "bravo")]

    with pytest.raises(ValueError, match="doc-1"):
        ContextBuilder(model=MODEL).build(clash, budget_tokens=100)


def test_an_unknown_model_still_produces_a_plan() -> None:
    """The tokenizer's fallback is inherited, not re-implemented here.

    A model name tiktoken does not recognize must not take context construction
    down; it falls back to the default encoding, and under it these particular
    strings happen to cost the same.
    """
    plan = ContextBuilder(model=UNKNOWN_MODEL).build(
        _worked_example(), budget_tokens=5
    )

    fallback = tiktoken.get_encoding(FALLBACK_ENCODING)
    assert plan.used_tokens == sum(len(fallback.encode(c.text)) for c in plan.included)
    assert included_ids(plan) == ["pol-1", "doc-2"]


def test_the_builder_measures_the_text_and_not_what_a_caller_claims() -> None:
    """The count cannot be supplied, so it cannot be stale.

    Asserted structurally: a candidate carrying its own token count is not a
    thing that can be constructed.
    """
    with pytest.raises(ValueError):
        ContextCandidate(
            candidate_id="doc-1",
            kind="evidence",
            text="milestone status",
            priority=0,
            token_count=1,
        )


def test_a_plan_cannot_be_edited_after_it_is_made() -> None:
    plan = ContextBuilder(model=MODEL).build(_worked_example(), budget_tokens=5)

    with pytest.raises(Exception):
        plan.used_tokens = 0  # type: ignore[misc]


def test_a_kind_nobody_budgets_for_is_refused() -> None:
    with pytest.raises(ValueError):
        candidate("doc-1", "alpha", kind="something_new")
