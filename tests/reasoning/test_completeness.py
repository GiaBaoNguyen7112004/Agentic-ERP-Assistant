"""assess(): a pure, deterministic check against typed state, and nothing
else -- no reply text, no rationale, no second model call (ADR 0021)."""

from agentic_erp_assistant.reasoning.completeness import (
    REDIRECT_ORDER,
    assess,
    next_redirect,
)
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.reply_contract import ReplyContract
from agentic_erp_assistant.state.tool_outcome import ToolOutcome

SNIPPET = EvidenceSnippet(
    source_id="status-report-2026-09.md", locator="2.2", text="Two days late."
)


def ok_observation(tool_name: str = "get_project_status") -> ToolOutcome:
    return ToolOutcome(
        tool_name=tool_name,
        status="ok",
        summary="2 days late.",
        source_ids=("milestone-m2",),
    )


def failed_observation(tool_name: str = "get_project_status") -> ToolOutcome:
    return ToolOutcome(
        tool_name=tool_name, status="invalid_arguments", error="bad milestone id"
    )


def a_mutating_ok_observation() -> ToolOutcome:
    return ToolOutcome(
        tool_name="create_risk",
        status="ok",
        summary="Recorded R-6 against orion.",
        source_ids=("risk-r-6",),
    )


def state(**changes) -> AgentState:
    base = AgentState(
        request="Why is milestone M2 late and by how much?",
        actor="priya",
        project_code="atlas",
        trace_id="run-1",
    )
    return base.evolve(**changes) if changes else base


# --------------------------------------------------------------------------
# contract=None: nothing to hold the turn to
# --------------------------------------------------------------------------


def test_no_contract_is_complete() -> None:
    completeness = assess(None, state())

    assert completeness.complete
    assert completeness.missing == frozenset()
    assert completeness.satisfied == frozenset()


# --------------------------------------------------------------------------
# A contract with no needs: a write, a refusal, a clarification
# --------------------------------------------------------------------------


def test_a_contract_with_no_needs_is_always_complete() -> None:
    contract = ReplyContract(needs=frozenset())

    completeness = assess(contract, state())

    assert completeness.complete


# --------------------------------------------------------------------------
# document_passage
# --------------------------------------------------------------------------


def test_document_passage_is_missing_with_no_evidence() -> None:
    contract = ReplyContract(needs=frozenset({"document_passage"}), document_query="why")

    completeness = assess(contract, state())

    assert completeness.missing == frozenset({"document_passage"})
    assert not completeness.complete


def test_document_passage_is_satisfied_once_evidence_is_present() -> None:
    contract = ReplyContract(needs=frozenset({"document_passage"}), document_query="why")

    completeness = assess(contract, state(evidence=(SNIPPET,)))

    assert completeness.satisfied == frozenset({"document_passage"})
    assert completeness.complete


def test_an_ok_observation_alone_does_not_satisfy_document_passage() -> None:
    """A live field is not a quoted passage, whatever its status."""
    contract = ReplyContract(needs=frozenset({"document_passage"}), document_query="why")

    completeness = assess(contract, state(observations=(ok_observation(),)))

    assert completeness.missing == frozenset({"document_passage"})


# --------------------------------------------------------------------------
# erp_field
# --------------------------------------------------------------------------


def test_erp_field_is_missing_with_no_observations() -> None:
    contract = ReplyContract(needs=frozenset({"erp_field"}))

    completeness = assess(contract, state())

    assert completeness.missing == frozenset({"erp_field"})


def test_erp_field_is_missing_when_every_observation_failed() -> None:
    contract = ReplyContract(needs=frozenset({"erp_field"}))

    completeness = assess(contract, state(observations=(failed_observation(),)))

    assert completeness.missing == frozenset({"erp_field"})


def test_erp_field_is_satisfied_by_one_ok_observation_among_failures() -> None:
    contract = ReplyContract(needs=frozenset({"erp_field"}))

    completeness = assess(
        contract, state(observations=(failed_observation(), ok_observation()))
    )

    assert completeness.satisfied == frozenset({"erp_field"})


def test_a_mutating_ok_observation_satisfies_erp_field_too() -> None:
    """A write's own receipt is a live ERP fact -- see the module docstring
    for why this is not narrowed to read-only tools."""
    contract = ReplyContract(needs=frozenset({"erp_field"}))

    completeness = assess(contract, state(observations=(a_mutating_ok_observation(),)))

    assert completeness.satisfied == frozenset({"erp_field"})


def test_evidence_alone_does_not_satisfy_erp_field() -> None:
    contract = ReplyContract(needs=frozenset({"erp_field"}))

    completeness = assess(contract, state(evidence=(SNIPPET,)))

    assert completeness.missing == frozenset({"erp_field"})


# --------------------------------------------------------------------------
# Both needs
# --------------------------------------------------------------------------


def test_both_needs_missing_with_nothing_yet() -> None:
    contract = ReplyContract(
        needs=frozenset({"document_passage", "erp_field"}), document_query="why"
    )

    completeness = assess(contract, state())

    assert completeness.missing == frozenset({"document_passage", "erp_field"})


def test_one_of_two_needs_satisfied_leaves_the_other_missing() -> None:
    contract = ReplyContract(
        needs=frozenset({"document_passage", "erp_field"}), document_query="why"
    )

    completeness = assess(contract, state(observations=(ok_observation(),)))

    assert completeness.satisfied == frozenset({"erp_field"})
    assert completeness.missing == frozenset({"document_passage"})


def test_both_needs_satisfied_is_complete() -> None:
    contract = ReplyContract(
        needs=frozenset({"document_passage", "erp_field"}), document_query="why"
    )

    completeness = assess(
        contract, state(evidence=(SNIPPET,), observations=(ok_observation(),))
    )

    assert completeness.complete


# --------------------------------------------------------------------------
# next_redirect: order and the once-only bound
# --------------------------------------------------------------------------


def test_redirect_order_is_field_before_passage() -> None:
    assert REDIRECT_ORDER == ("erp_field", "document_passage")


def test_erp_field_is_redirected_before_document_passage() -> None:
    contract = ReplyContract(
        needs=frozenset({"document_passage", "erp_field"}), document_query="why"
    )
    completeness = assess(contract, state())

    assert next_redirect(completeness, frozenset()) == "erp_field"


def test_a_need_already_redirected_is_not_offered_again() -> None:
    contract = ReplyContract(needs=frozenset({"erp_field"}))
    completeness = assess(contract, state())

    assert next_redirect(completeness, frozenset({"erp_field"})) is None


def test_document_passage_is_offered_once_erp_field_is_redirected() -> None:
    contract = ReplyContract(
        needs=frozenset({"document_passage", "erp_field"}), document_query="why"
    )
    completeness = assess(contract, state())

    assert next_redirect(completeness, frozenset({"erp_field"})) == "document_passage"


def test_nothing_missing_offers_no_redirect() -> None:
    contract = ReplyContract(needs=frozenset({"erp_field"}))
    completeness = assess(contract, state(observations=(ok_observation(),)))

    assert next_redirect(completeness, frozenset()) is None
