"""A contract's document_query is required exactly when document_passage is
declared needed -- the same "a field that only means something on the route
that uses it" rule ReasoningDecision already applies elsewhere."""

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.state.reply_contract import EMPTY_CONTRACT, ReplyContract


def test_no_needs_is_a_complete_answer() -> None:
    contract = ReplyContract(needs=frozenset())

    assert contract.needs == frozenset()
    assert contract.document_query is None


def test_empty_contract_constant_has_no_needs() -> None:
    assert EMPTY_CONTRACT.needs == frozenset()
    assert EMPTY_CONTRACT.document_query is None


def test_erp_field_alone_needs_no_query() -> None:
    contract = ReplyContract(needs=frozenset({"erp_field"}))

    assert contract.document_query is None


def test_document_passage_requires_a_query() -> None:
    with pytest.raises(ValidationError, match="document_query"):
        ReplyContract(needs=frozenset({"document_passage"}))


def test_document_passage_with_a_query_is_valid() -> None:
    contract = ReplyContract(
        needs=frozenset({"document_passage"}), document_query="why is M2 late"
    )

    assert contract.document_query == "why is M2 late"


def test_both_needs_still_requires_a_query() -> None:
    with pytest.raises(ValidationError, match="document_query"):
        ReplyContract(needs=frozenset({"document_passage", "erp_field"}))


def test_a_query_without_document_passage_is_rejected() -> None:
    """The query would be an input no redirect ever reads -- rejected at
    construction rather than silently carried and ignored."""
    with pytest.raises(ValidationError, match="document_query"):
        ReplyContract(needs=frozenset({"erp_field"}), document_query="why is M2 late")


def test_contract_is_frozen() -> None:
    contract = ReplyContract(needs=frozenset())

    with pytest.raises(ValidationError):
        contract.needs = frozenset({"erp_field"})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ReplyContract(needs=frozenset(), extra_field="unexpected")
