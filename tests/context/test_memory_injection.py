"""Recall is a decision about this turn, so every memory ends up either in the
prompt or in a list with a typed reason beside it."""

from datetime import UTC, datetime

import pytest

from agentic_erp_assistant.context.memory_injection import (
    KIND_PRIORITY,
    MIN_TERM_OVERLAP,
    PINNED_KINDS,
    select_memories,
)
from agentic_erp_assistant.llm.tokenizer import TokenCounter
from agentic_erp_assistant.memory.models import MemoryScope
from agentic_erp_assistant.state.memory import MemoryKind, MemoryRecord

MODEL = "gpt-4o"
RECORDED = datetime(2026, 9, 8, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 9, 9, 0, tzinfo=UTC)

REQUEST = "How is the cutover window looking for milestone M2?"


class CountingWords:
    """One token per whitespace-separated word.

    A real ``TokenCounter``, and deliberately not tiktoken: a budget test whose
    expected numbers depend on a vendor's encoding is a test that has to be
    re-derived every time the model changes, and what is being proven here is
    the selection rule, not the arithmetic (``tests/context/test_builder.py``
    already proves that).
    """

    def count_tokens(self, text: str, *, model: str) -> int:
        return len(text.split())

    def count_message_tokens(self, messages, *, model: str) -> int:  # pragma: no cover
        return sum(self.count_tokens(m["content"], model=model) for m in messages)


def scope(**overrides: object) -> MemoryScope:
    fields: dict[str, object] = {
        "project_code": "atlas",
        "session_id": "sess-1",
        "scopes": frozenset({"project.docs.read"}),
    }
    fields.update(overrides)
    actor = fields.pop("actor", "priya")
    return MemoryScope.for_actor(actor, **fields)  # type: ignore[arg-type]


def record(**overrides: object) -> MemoryRecord:
    fields: dict[str, object] = {
        "memory_id": "mem-1",
        "kind": "fact",
        "key": "cutover_owner",
        "statement": "The cutover window is owned by the delivery lead.",
        "project_code": "atlas",
        "required_scope": "project.docs.read",
        "actor": "priya",
        "session_id": "sess-1",
        "recorded_in_run": "run-1",
        "recorded_at": RECORDED,
        "confidence": 0.9,
    }
    fields.update(overrides)
    return MemoryRecord(**fields)  # type: ignore[arg-type]


def select(*records: MemoryRecord, **overrides: object):
    fields: dict[str, object] = {
        "scope": scope(),
        "budget_tokens": 1_000,
        "model": MODEL,
        "counter": CountingWords(),
    }
    request = overrides.pop("request", REQUEST)
    fields.update(overrides)
    return select_memories(request, records, **fields)  # type: ignore[arg-type]


def reasons(selection) -> dict[str, str]:
    return {item.record.memory_id: item.reason for item in selection.skipped}


# --------------------------------------------------------------------------
# Every record lands in exactly one place
# --------------------------------------------------------------------------


def test_a_relevant_memory_is_selected() -> None:
    selection = select(record())

    assert [r.memory_id for r in selection.selected] == ["mem-1"]
    assert selection.skipped == ()


def test_every_input_is_accounted_for_exactly_once() -> None:
    """The property that makes the selection a record rather than a summary."""
    records = [
        record(memory_id="mem-keep"),
        record(memory_id="mem-old", statement="Invoices are paid quarterly.", key="a"),
        record(memory_id="mem-gone").retired(LATER),
    ]

    selection = select(*records)

    seen = {r.memory_id for r in selection.selected} | set(reasons(selection))
    assert seen == {"mem-keep", "mem-old", "mem-gone"}


# --------------------------------------------------------------------------
# Skipping: out of scope, retired, or nothing to do with the question
# --------------------------------------------------------------------------


def test_a_retired_memory_is_skipped_as_superseded() -> None:
    selection = select(record().retired(LATER))

    assert reasons(selection) == {"mem-1": "superseded"}


def test_a_memory_this_actor_may_not_read_is_skipped_out_of_scope() -> None:
    """The last check before text enters a prompt, and deliberately a third
    copy of a rule the store and the index both already apply."""
    selection = select(record(required_scope="project.docs.finance.read"))

    assert reasons(selection) == {"mem-1": "out_of_scope"}


def test_another_projects_memory_is_skipped_out_of_scope() -> None:
    selection = select(record(project_code="borealis"))

    assert reasons(selection) == {"mem-1": "out_of_scope"}


def test_another_actors_preference_is_skipped_out_of_scope() -> None:
    selection = select(record(kind="preference", key="lang", actor="marco"))

    assert reasons(selection) == {"mem-1": "out_of_scope"}


def test_a_memory_about_something_else_is_skipped_not_relevant() -> None:
    """The common case, and what makes this a memory system rather than a way
    to fill a context window."""
    selection = select(
        record(statement="Vendor invoices are paid quarterly in arrears.")
    )

    assert reasons(selection) == {"mem-1": "not_relevant"}


def test_one_shared_word_is_not_enough() -> None:
    """A single shared word is the noise floor: "project" and "budget" appear in
    nearly every request and nearly every memory."""
    selection = select(
        record(statement="The window cleaner attends on alternate Fridays."),
        request="How is the cutover window looking?",
    )

    assert reasons(selection) == {"mem-1": "not_relevant"}


def test_the_overlap_threshold_is_what_decides() -> None:
    shared = " ".join(["cutover", "milestone"][:MIN_TERM_OVERLAP])

    selection = select(record(statement=f"Notes about {shared} arrangements."))

    assert [r.memory_id for r in selection.selected] == ["mem-1"]


# --------------------------------------------------------------------------
# Pinned kinds: relevance is not tested for them
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(PINNED_KINDS))
def test_a_pinned_kind_reaches_the_prompt_however_the_question_is_phrased(
    kind: MemoryKind,
) -> None:
    """An intent says what is being worked on and a preference says how to
    reply; testing either against the words of the question would drop it from
    exactly the turns where the phrasing differs."""
    selection = select(
        record(kind=kind, key="k", statement="Prefers figures rounded to thousands."),
        request="What about the vendor?",
    )

    assert [r.memory_id for r in selection.selected] == ["mem-1"]


@pytest.mark.parametrize("kind", ["decision", "fact"])
def test_an_unpinned_kind_still_has_to_be_about_something(kind: MemoryKind) -> None:
    selection = select(
        record(kind=kind, key="k", statement="Prefers figures rounded to thousands."),
        request="What about the vendor?",
    )

    assert reasons(selection) == {"mem-1": "not_relevant"}


def test_a_vector_match_is_relevant_without_sharing_any_words() -> None:
    """Relevance was already decided by something better at it than word
    overlap."""
    selection = select(
        record(statement="Deployment is handed to operations at sign-off."),
        matched=["mem-1"],
    )

    assert [r.memory_id for r in selection.selected] == ["mem-1"]


# --------------------------------------------------------------------------
# Priority: what survives when the budget runs out
# --------------------------------------------------------------------------


def test_the_intent_outranks_everything_else() -> None:
    """Losing the intent means the assistant does not know what it is doing;
    losing a fact costs a retrieval."""
    selection = select(
        record(memory_id="mem-fact", kind="fact", key="f"),
        record(memory_id="mem-intent", kind="intent", key="i", statement="Working on the cutover."),
        budget_tokens=4,
    )

    assert [r.memory_id for r in selection.selected] == ["mem-intent"]


def test_the_kind_bands_are_ordered_by_what_losing_one_costs() -> None:
    assert (
        KIND_PRIORITY["intent"]
        > KIND_PRIORITY["preference"]
        > KIND_PRIORITY["session_summary"]
        > KIND_PRIORITY["decision"]
        > KIND_PRIORITY["fact"]
    )


def test_recency_breaks_ties_inside_a_kind() -> None:
    selection = select(
        record(memory_id="mem-old", key="a", recorded_at=RECORDED),
        record(memory_id="mem-new", key="b", recorded_at=LATER),
    )

    assert [r.memory_id for r in selection.selected] == ["mem-new", "mem-old"]


def test_a_newer_fact_never_outranks_an_intent() -> None:
    """What the thousand-wide spacing buys: recency reorders inside a band and
    never across one."""
    selection = select(
        record(memory_id="mem-fact", kind="fact", key="f", recorded_at=LATER),
        record(
            memory_id="mem-intent",
            kind="intent",
            key="i",
            statement="Working on the cutover.",
            recorded_at=RECORDED,
        ),
        budget_tokens=4,
    )

    assert [r.memory_id for r in selection.selected] == ["mem-intent"]


# --------------------------------------------------------------------------
# The budget, which is the builder's job and stays there
# --------------------------------------------------------------------------


def test_what_does_not_fit_is_excluded_by_the_builder_not_skipped_here() -> None:
    """Two mechanisms, deliberately: an irrelevant memory means recall is too
    broad, an excluded one means the budget is too small, and the fixes are
    opposite."""
    selection = select(
        record(memory_id="mem-a", key="a"),
        record(memory_id="mem-b", key="b"),
        budget_tokens=9,  # one statement's worth, under CountingWords
    )

    assert selection.skipped == ()
    assert len(selection.selected) == 1
    assert selection.excluded == ("mem-a",)
    assert selection.plan.overflowed is True


def test_a_budget_of_zero_selects_nothing_and_skips_nothing() -> None:
    selection = select(record(), budget_tokens=0)

    assert selection.selected == ()
    assert selection.skipped == ()
    assert selection.excluded == ("mem-1",)


def test_the_plan_is_kept_whole_so_the_trace_can_see_the_overflow() -> None:
    selection = select(record(), budget_tokens=1_000)

    assert selection.plan.budget_tokens == 1_000
    assert selection.plan.overflowed is False


def test_the_selection_is_reproducible() -> None:
    records = [
        record(memory_id="mem-a", key="a"),
        record(memory_id="mem-b", key="b"),
        record(memory_id="mem-c", key="c"),
    ]

    first = select(*records)
    second = select(*reversed(records))

    assert [r.memory_id for r in first.selected] == [
        r.memory_id for r in second.selected
    ]


def test_the_counter_is_the_ports_and_not_a_second_one() -> None:
    assert isinstance(CountingWords(), TokenCounter)
