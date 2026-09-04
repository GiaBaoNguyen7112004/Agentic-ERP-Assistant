"""A routing decision is only auditable if the invalid ones cannot be built."""

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.reasoning.decision import (
    classify_failure,
    RATIONALE_MAX_CHARS,
    ReasoningDecision,
)


# --------------------------------------------------------------------------
# The two constructions the unit says must raise
# --------------------------------------------------------------------------


def test_a_tool_call_without_a_tool_name_cannot_exist() -> None:
    """The dispatcher would have nothing to dispatch, and would find out late."""
    with pytest.raises(ValidationError, match="required_tool"):
        ReasoningDecision(route="call_tool", confidence=0.9, required_tool=None)


def test_approval_cannot_be_required_on_a_route_that_executes_nothing() -> None:
    """'clarify, but get it approved first' is not a state the runtime has."""
    with pytest.raises(ValidationError, match="approval_required"):
        ReasoningDecision(route="clarify", confidence=0.4, approval_required=True)


def test_the_invariants_run_before_the_object_exists() -> None:
    """Not a downstream check: a second caller could skip that one."""
    try:
        ReasoningDecision(route="call_tool", confidence=0.9)
    except ValidationError:
        return
    raise AssertionError("an invalid decision was constructed")


# --------------------------------------------------------------------------
# The valid shapes
# --------------------------------------------------------------------------


def test_a_tool_call_that_names_its_tool_is_valid() -> None:
    decision = ReasoningDecision(
        route="call_tool",
        confidence=0.9,
        required_tool="get_project_status",
    )

    assert decision.required_tool == "get_project_status"
    assert decision.approval_required is False


def test_a_mutating_call_carries_both_the_tool_and_the_approval_flag() -> None:
    decision = ReasoningDecision(
        route="request_approval",
        confidence=0.8,
        required_tool="close_milestone",
        approval_required=True,
        rationale="Mutating call; a human must confirm before it runs.",
    )

    assert decision.approval_required is True


def test_retrieval_records_the_sources_it_expects_to_need() -> None:
    decision = ReasoningDecision(
        route="retrieve_project_documents",
        confidence=0.7,
        required_evidence=("sprint-12-report.md", "budget-q3.md"),
    )

    assert decision.required_evidence == ("sprint-12-report.md", "budget-q3.md")


@pytest.mark.parametrize("route", ["clarify", "refuse"])
def test_the_terminal_routes_need_neither_tool_nor_evidence(route: str) -> None:
    decision = ReasoningDecision(route=route, confidence=0.2)

    assert decision.required_tool is None
    assert decision.required_evidence == ()


def test_an_unroutable_label_is_rejected_at_the_decision() -> None:
    """Closed set, so a label nobody can dispatch fails here, not at dispatch."""
    with pytest.raises(ValidationError):
        ReasoningDecision(route="do_something_clever", confidence=0.9)


def test_the_intent_route_names_are_not_accepted_here() -> None:
    """Two axes, deliberately: 'erp_write' is what the user asked for, not what
    the runtime does next."""
    with pytest.raises(ValidationError):
        ReasoningDecision(route="erp_write", confidence=0.9)


# --------------------------------------------------------------------------
# Field-level rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_confidence_admits_both_ends_of_the_range(confidence: float) -> None:
    decision = ReasoningDecision(route="refuse", confidence=confidence)

    assert decision.confidence == confidence


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_outside_zero_to_one_is_rejected(confidence: float) -> None:
    with pytest.raises(ValidationError, match="confidence"):
        ReasoningDecision(route="refuse", confidence=confidence)


def test_a_rationale_at_the_cap_is_accepted() -> None:
    decision = ReasoningDecision(
        route="refuse",
        confidence=0.1,
        rationale="x" * RATIONALE_MAX_CHARS,
    )

    assert len(decision.rationale) == RATIONALE_MAX_CHARS


def test_a_rationale_over_the_cap_is_rejected() -> None:
    """The cap is what stops this field becoming a chain-of-thought transcript
    that lands in the trace and gets read as audited reasoning."""
    with pytest.raises(ValidationError, match="rationale"):
        ReasoningDecision(
            route="refuse",
            confidence=0.1,
            rationale="x" * (RATIONALE_MAX_CHARS + 1),
        )


def test_a_tool_name_on_a_route_that_calls_nothing_is_rejected() -> None:
    """A field no branch reads is a routing input the reviewer cannot verify."""
    with pytest.raises(ValidationError, match="required_tool"):
        ReasoningDecision(
            route="clarify",
            confidence=0.4,
            required_tool="get_project_status",
        )


def test_a_blank_tool_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="required_tool"):
        ReasoningDecision(route="call_tool", confidence=0.9, required_tool="   ")


def test_a_blank_source_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="required_evidence"):
        ReasoningDecision(
            route="retrieve_project_documents",
            confidence=0.7,
            required_evidence=("sprint-12-report.md", ""),
        )


def test_an_unmodelled_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReasoningDecision(route="refuse", confidence=0.1, escalate=True)


def test_a_decision_cannot_be_edited_after_it_is_audited() -> None:
    """An approval gate must read the value the decision was reviewed with."""
    decision = ReasoningDecision(
        route="call_tool",
        confidence=0.9,
        required_tool="get_project_status",
    )

    with pytest.raises(ValidationError):
        decision.approval_required = True  # type: ignore[misc]


def test_required_evidence_cannot_be_appended_to_in_place() -> None:
    """A frozen model holding a list would only look frozen."""
    decision = ReasoningDecision(
        route="retrieve_project_documents",
        confidence=0.7,
        required_evidence=("sprint-12-report.md",),
    )

    assert isinstance(decision.required_evidence, tuple)
    with pytest.raises(AttributeError):
        decision.required_evidence.append("smuggled.md")  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# classify_failure -- the precedence, proved as a table
# --------------------------------------------------------------------------


@pytest.mark.parametrize("evidence_count", [0, 3])
@pytest.mark.parametrize("provider_error", [False, True])
def test_budget_overflow_outranks_everything(
    evidence_count: int, provider_error: bool
) -> None:
    """Every combination, because one lucky branch proves nothing about
    precedence. A request too large to send explains the other two."""
    assert (
        classify_failure(
            evidence_count=evidence_count,
            budget_overflow=True,
            provider_error=provider_error,
        )
        == "context_budget_exceeded"
    )


@pytest.mark.parametrize("evidence_count", [0, 3])
def test_a_provider_failure_outranks_thin_evidence(evidence_count: int) -> None:
    """A call that never returned explains an empty result; not the reverse."""
    assert (
        classify_failure(
            evidence_count=evidence_count,
            budget_overflow=False,
            provider_error=True,
        )
        == "provider_failure"
    )


def test_no_evidence_and_no_other_fault_is_insufficient_evidence() -> None:
    assert (
        classify_failure(
            evidence_count=0, budget_overflow=False, provider_error=False
        )
        == "insufficient_evidence"
    )


def test_a_clean_turn_reports_none() -> None:
    assert (
        classify_failure(
            evidence_count=3, budget_overflow=False, provider_error=False
        )
        == "none"
    )


def test_the_clean_result_is_the_string_none_not_the_object() -> None:
    """'none' is a member of the Literal, so callers must match a branch rather
    than writing `if failure:` and collapsing four outcomes into two."""
    result = classify_failure(
        evidence_count=3, budget_overflow=False, provider_error=False
    )

    assert result is not None
    assert result == "none"


def test_the_arguments_are_keyword_only() -> None:
    """Two of three are booleans; a positional swap would be silent."""
    with pytest.raises(TypeError):
        classify_failure(0, True, False)  # type: ignore[misc]


def test_a_negative_evidence_count_is_a_caller_bug() -> None:
    with pytest.raises(ValueError, match="evidence_count"):
        classify_failure(
            evidence_count=-1, budget_overflow=False, provider_error=False
        )
