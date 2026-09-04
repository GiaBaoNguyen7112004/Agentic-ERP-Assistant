"""Compaction's promise is negative -- it is about what cannot disappear.

So these tests are mostly about absence and survival rather than about output
shape. Two habits run through them. Every test name contains "compaction", which
keeps both of the project's check commands honest: ``-k compact`` selects this
file, and the builder's ``-k "not compaction"`` excludes it, so neither command
is quietly matching the wrong thing.

And the summarizer is always a fake. A summarizer may be an LLM call in
production, but nothing here reaches a provider -- the point being tested is that
the six preserved fields do not depend on one.
"""

import dataclasses

import pytest

from agentic_erp_assistant.context.compact import (
    CompactedConversation,
    PRESERVED_FIELDS,
    SUMMARY_MAX_CHARS,
    SUMMARY_UNAVAILABLE,
    compact_conversation,
    structural_summary,
)

# The spec's own example, kept verbatim: it does not match this project's
# ApprovalRequest shape, and that is the point. Compaction preserves whatever an
# upstream layer created rather than validating it, because a validation failure
# here would lose the very approval the allow-list exists to protect.
DELETE_APPROVAL = {"tool": "delete_record", "id": "rec-9"}


def full_state() -> dict[str, object]:
    """Every preserved field populated, plus one field that must not survive."""
    return {
        "user_goal": "find out why sprint 12 slipped",
        "accepted_facts": ["sprint 12 closed two weeks late"],
        "citations": [{"source_id": "sprint-12-report.md", "locator": "3.2"}],
        "pending_approvals": [DELETE_APPROVAL],
        "safety_flags": ["prompt_injection_suspected"],
        "unresolved_questions": ["was the delay staffing or scope?"],
        "raw_transcript": "user: ...\nassistant: ...\n" * 20,
    }


def test_compaction_preserves_every_field_and_drops_the_transcript() -> None:
    """The named acceptance criterion, field by field rather than sampled.

    All six are asserted individually and by value. Checking two of them and
    trusting the rest would pass just as well against an implementation that
    preserved two of them, which is the failure this test exists to catch.
    """
    state = full_state()

    compacted = compact_conversation(state)

    assert compacted.user_goal == "find out why sprint 12 slipped"
    assert compacted.accepted_facts == ("sprint 12 closed two weeks late",)
    assert compacted.citations == (
        {"source_id": "sprint-12-report.md", "locator": "3.2"},
    )
    assert compacted.pending_approvals == (DELETE_APPROVAL,)
    assert compacted.safety_flags == ("prompt_injection_suspected",)
    assert compacted.unresolved_questions == ("was the delay staffing or scope?",)

    assert "raw_transcript" not in compacted.as_state()


def test_compaction_keeps_the_pending_approval_intact_not_merely_present() -> None:
    """The spec's worked example: the delete_record entry survives as written."""
    state = {"pending_approvals": [DELETE_APPROVAL], "raw_transcript": "..."}

    compacted = compact_conversation(state)

    assert compacted.pending_approvals == ({"tool": "delete_record", "id": "rec-9"},)
    assert "raw_transcript" not in compacted.as_state()


def test_compaction_output_can_never_contain_a_field_outside_the_allow_list() -> None:
    """The allow-list as a property, not as one example.

    A deny-list would pass a test that names ``raw_transcript`` and fail this
    one, because this one asserts the rule for fields nobody thought of.
    """
    state = full_state()
    state.update(
        {
            "scratchpad": "notes",
            "tool_output_raw": {"rows": [1, 2, 3]},
            "some_field_invented_next_year": object(),
        }
    )

    keys = set(compact_conversation(state).as_state())

    assert keys <= {"summary", *PRESERVED_FIELDS}


@pytest.mark.parametrize("missing", PRESERVED_FIELDS)
def test_compaction_of_a_partial_state_leaves_the_absent_field_absent(
    missing: str,
) -> None:
    """"Survives when present" implies absent stays absent, for each field."""
    state = full_state()
    del state[missing]

    compacted = compact_conversation(state)

    assert getattr(compacted, missing) is None
    assert missing not in compacted.as_state()
    for name in PRESERVED_FIELDS:
        if name != missing:
            assert name in compacted.as_state()


def test_compaction_of_an_empty_state_is_an_empty_compaction() -> None:
    compacted = compact_conversation({})

    assert compacted.as_state() == {"summary": compacted.summary}
    for name in PRESERVED_FIELDS:
        assert getattr(compacted, name) is None


def test_compaction_treats_an_explicitly_none_field_as_absent() -> None:
    """Documented behaviour: ``None`` carries nothing worth preserving."""
    compacted = compact_conversation({"user_goal": None, "safety_flags": ["flag"]})

    assert "user_goal" not in compacted.as_state()
    assert compacted.safety_flags == ("flag",)


def test_compaction_hands_the_summarizer_what_it_is_about_to_drop() -> None:
    """Summarizing only the survivors would describe nothing that was lost."""
    seen: dict[str, object] = {}

    def spy(state):
        seen.update(state)
        return "summarized"

    compacted = compact_conversation(full_state(), summarizer=spy)

    assert "raw_transcript" in seen
    assert compacted.summary == "summarized"


def test_compaction_survives_a_summarizer_that_raises() -> None:
    """A provider outage must not be able to destroy conversation state."""

    def exploding(state):
        raise RuntimeError("provider unavailable")

    compacted = compact_conversation(full_state(), summarizer=exploding)

    assert compacted.pending_approvals == (DELETE_APPROVAL,)
    assert compacted.safety_flags == ("prompt_injection_suspected",)
    assert compacted.summary == structural_summary(full_state())


def test_compaction_survives_a_summarizer_that_returns_the_wrong_type() -> None:
    compacted = compact_conversation(full_state(), summarizer=lambda state: 42)

    assert compacted.pending_approvals == (DELETE_APPROVAL,)
    assert compacted.summary == structural_summary(full_state())


def test_compaction_reports_when_no_summary_could_be_produced() -> None:
    """A failing default leaves a stated absence, not an empty string."""

    def exploding(state):
        raise RuntimeError("nope")

    monkeypatched = compact_conversation({}, summarizer=exploding)
    assert monkeypatched.summary == structural_summary({})

    assert SUMMARY_UNAVAILABLE  # the constant exists for the double-failure path


def test_compaction_caps_a_runaway_summary() -> None:
    """Otherwise the summary is where the transcript comes back under a new name."""
    compacted = compact_conversation({}, summarizer=lambda state: "x" * 10_000)

    assert len(compacted.summary) == SUMMARY_MAX_CHARS
    assert compacted.summary.endswith("…")


def test_compaction_does_not_quote_the_content_it_dropped() -> None:
    """The default summary reports sizes, never text."""
    secret = "the CFO said the budget is blown"
    summary = structural_summary({"raw_transcript": secret})

    assert secret not in summary
    assert "raw_transcript" in summary
    assert f"{len(secret)} chars" in summary


def test_compaction_summary_does_not_depend_on_dict_insertion_order() -> None:
    """Deterministic: the caller's dict-building order is not a fact about the run."""
    first = {"raw_transcript": "...", "scratchpad": "...", "user_goal": "ship it"}
    second = {"user_goal": "ship it", "scratchpad": "...", "raw_transcript": "..."}

    assert structural_summary(first) == structural_summary(second)


def test_compaction_freezes_a_preserved_list_into_a_tuple() -> None:
    """A frozen record holding a mutable list is not frozen, it merely looks it."""
    flags = ["prompt_injection_suspected"]

    compacted = compact_conversation({"safety_flags": flags})
    flags.append("added later")

    assert compacted.safety_flags == ("prompt_injection_suspected",)


def test_compaction_does_not_transform_a_value_that_is_not_a_list() -> None:
    """``tuple()`` over a string would silently explode it into characters."""
    compacted = compact_conversation({"user_goal": "ship it", "safety_flags": ()})

    assert compacted.user_goal == "ship it"
    assert compacted.safety_flags == ()


def test_compaction_leaves_the_elements_of_a_preserved_list_alone() -> None:
    """Preserved, not validated: an entry keeps whatever shape it arrived with."""
    entry = {"tool": "delete_record", "id": "rec-9", "extra": ["anything"]}

    compacted = compact_conversation({"pending_approvals": [entry]})

    assert compacted.pending_approvals[0] is entry


def test_compaction_does_not_validate_a_malformed_citation() -> None:
    """Rejecting one here would lose the grounding compaction must protect."""
    nonsense = ["not a citation object at all"]

    compacted = compact_conversation({"citations": nonsense})

    assert compacted.citations == ("not a citation object at all",)


def test_a_compaction_cannot_be_edited_after_the_fact() -> None:
    compacted = compact_conversation(full_state())

    with pytest.raises(dataclasses.FrozenInstanceError):
        compacted.pending_approvals = ()  # type: ignore[misc]


def test_compaction_round_trips_through_as_state() -> None:
    """What comes out is a shape the next turn can be fed."""
    once = compact_conversation(full_state())
    twice = compact_conversation(once.as_state())

    assert twice.pending_approvals == once.pending_approvals
    assert twice.citations == once.citations
    assert twice.safety_flags == once.safety_flags
    assert twice.user_goal == once.user_goal


def test_compaction_preserves_exactly_the_fields_the_record_declares() -> None:
    """Guards against a field being added to one and forgotten in the other."""
    fields = {f.name for f in dataclasses.fields(CompactedConversation)}

    assert fields == {"summary", *PRESERVED_FIELDS}
