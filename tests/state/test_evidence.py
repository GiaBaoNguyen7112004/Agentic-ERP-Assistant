"""A citation is only resolvable if the passage kept its address -- and only
trustworthy if a retrieved document cannot forge one."""

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.state.evidence import EvidenceSnippet


def snippet(**overrides: str) -> EvidenceSnippet:
    fields = {
        "source_id": "sprint-12-report.md",
        "locator": "3.2",
        "text": "M2 slipped by two days.",
    }
    fields.update(overrides)
    return EvidenceSnippet(**fields)


def test_a_snippet_carries_the_document_and_the_place_inside_it() -> None:
    evidence = snippet()

    assert evidence.source_id == "sprint-12-report.md"
    assert evidence.locator == "3.2"


def test_the_tag_is_what_the_model_is_told_to_cite() -> None:
    assert snippet().tag == "[sprint-12-report.md#3.2]"


@pytest.mark.parametrize("field", ["source_id", "locator", "text"])
def test_every_field_is_required(field: str) -> None:
    fields = {"source_id": "s1", "locator": "3.2", "text": "t"}
    del fields[field]

    with pytest.raises(ValidationError, match=field):
        EvidenceSnippet(**fields)


@pytest.mark.parametrize("field", ["source_id", "locator", "text"])
def test_no_field_may_be_empty(field: str) -> None:
    """A passage with no address would reach the prompt and be uncitable there."""
    with pytest.raises(ValidationError, match=field):
        snippet(**{field: ""})


@pytest.mark.parametrize("forbidden", ["[", "]", "#", "\n", "\r"])
@pytest.mark.parametrize("field", ["source_id", "locator"])
def test_an_identifier_cannot_contain_the_characters_a_tag_is_built_from(
    field: str, forbidden: str
) -> None:
    """A security check. Both identifiers can arrive from a retrieved document,
    so if they could carry these characters that document would control part of
    the rendered tag and could manufacture a reference to a source that does not
    exist."""
    with pytest.raises(ValidationError, match=field):
        snippet(**{field: f"doc{forbidden}12"})


def test_the_text_may_contain_tag_characters() -> None:
    """Only the identifiers build the tag. Refusing brackets in the passage
    itself would silently drop legitimate evidence."""
    assert snippet(text="the budget line [B-4] is over by 3%").text.endswith("3%")


def test_a_snippet_cannot_be_edited_after_it_was_retrieved() -> None:
    """Otherwise the citation would describe text the model never read."""
    evidence = snippet()

    with pytest.raises(ValidationError):
        evidence.text = "something else"  # type: ignore[misc]


def test_unmodelled_retrieval_metadata_is_rejected() -> None:
    with pytest.raises(ValidationError):
        snippet(rerank_debug="0.4 -> 0.8")


def test_a_snippet_carries_no_score() -> None:
    """Rank order is the only thing the runtime reads. A score living in the
    object that goes into the prompt would be retrieval telemetry that no
    branch consults; it belongs to the trace."""
    assert "score" not in EvidenceSnippet.model_fields


def test_the_llm_layer_re_exports_the_same_class_rather_than_defining_one() -> None:
    """One definition. The name stays importable from llm.schemas, where this
    class used to live, so nothing downstream had to change when it moved to
    the state layer."""
    from agentic_erp_assistant.llm.schemas import EvidenceSnippet as ReExported

    assert ReExported is EvidenceSnippet
