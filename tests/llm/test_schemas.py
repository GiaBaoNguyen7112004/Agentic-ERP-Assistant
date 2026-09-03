"""The response contracts are guardrails, so the rejections are what get tested.

Each negative case below is a malformed answer that must not be able to exist.
If one of these ever constructs cleanly, a model can assert something unsupported
and no later layer will notice.
"""

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.llm.schemas import (
    ApprovalRequest,
    Citation,
    GroundedAnswer,
)

A_CITATION = Citation(source_id="sprint-12-report.md", locator="3.2")


def test_a_grounded_answer_without_citations_cannot_be_constructed() -> None:
    """The named acceptance criterion: grounding must be backed by a source."""
    with pytest.raises(ValidationError, match="citations"):
        GroundedAnswer(answer="Sprint 12 shipped.", grounded=True, confidence=0.9)


def test_a_refusal_without_a_reason_cannot_be_constructed() -> None:
    """The named acceptance criterion: a refusal must say why it refused."""
    with pytest.raises(ValidationError, match="refusal_reason"):
        GroundedAnswer(
            answer="",
            grounded=False,
            confidence=0.1,
            refusal_reason=None,
        )


def test_a_blank_refusal_reason_is_no_reason_at_all() -> None:
    with pytest.raises(ValidationError, match="refusal_reason"):
        GroundedAnswer(answer="", grounded=False, confidence=0.1, refusal_reason="  ")


def test_a_well_formed_grounded_answer_constructs_and_keeps_its_evidence() -> None:
    answer = GroundedAnswer(
        answer="Sprint 12 shipped 14 of 16 points.",
        citations=[A_CITATION],
        grounded=True,
        confidence=0.82,
    )

    assert answer.citations == [A_CITATION]
    assert answer.refusal_reason is None


def test_a_well_formed_refusal_constructs() -> None:
    answer = GroundedAnswer(
        answer="",
        grounded=False,
        confidence=0.05,
        refusal_reason="no retrieved passage covers Q3 headcount",
    )

    assert answer.citations == []


def test_a_validated_answer_cannot_be_edited_afterwards() -> None:
    """Frozen: the check would mean nothing if the object could change later."""
    answer = GroundedAnswer(
        answer="ok", citations=[A_CITATION], grounded=True, confidence=0.5
    )

    with pytest.raises(ValidationError):
        answer.citations = []  # type: ignore[misc]


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_outside_zero_to_one_is_rejected(confidence: float) -> None:
    with pytest.raises(ValidationError):
        GroundedAnswer(
            answer="ok",
            citations=[A_CITATION],
            grounded=True,
            confidence=confidence,
        )


def test_a_citation_must_actually_point_somewhere() -> None:
    with pytest.raises(ValidationError):
        Citation(source_id="", locator="3.2")
    with pytest.raises(ValidationError):
        Citation(source_id="sprint-12-report.md", locator="")


@pytest.mark.parametrize("field", ["tool_name", "reason", "arguments_summary"])
def test_an_approval_request_cannot_have_an_empty_field(field: str) -> None:
    """An approver cannot weigh a blank. Every field is shown to a human."""
    kwargs = {
        "tool_name": "erp.update_sprint",
        "reason": "the user asked to close sprint 12",
        "arguments_summary": "move SPR-14 to done",
    }
    kwargs[field] = ""

    with pytest.raises(ValidationError):
        ApprovalRequest(**kwargs)


def test_a_well_formed_approval_request_constructs() -> None:
    request = ApprovalRequest(
        tool_name="erp.update_sprint",
        reason="the user asked to close sprint 12",
        arguments_summary="move SPR-14 to done",
    )

    assert request.tool_name == "erp.update_sprint"


@pytest.mark.parametrize("model", [Citation, GroundedAnswer, ApprovalRequest])
def test_the_emitted_schema_is_usable_as_a_provider_response_format(model) -> None:
    """Structured output requires additionalProperties: false on every object."""
    assert model.model_json_schema()["additionalProperties"] is False
