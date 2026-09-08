"""A memory is only safe to recall if it says who may read it, when it was
learned, and whether anyone still believes it."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.rag.access import RetrievalContext, is_authorized
from agentic_erp_assistant.state.memory import (
    KEY_MAX_CHARS,
    STATEMENT_MAX_CHARS,
    MemoryRecord,
)

RECORDED = datetime(2026, 9, 8, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 9, 9, 0, tzinfo=UTC)


def record(**overrides: object) -> MemoryRecord:
    fields: dict[str, object] = {
        "memory_id": "mem-1",
        "kind": "preference",
        "key": "reply_language",
        "statement": "Prefers replies in Vietnamese.",
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


# --------------------------------------------------------------------------
# The expected path
# --------------------------------------------------------------------------


def test_a_new_record_is_live() -> None:
    assert record().live is True


def test_a_record_carries_the_run_and_the_session_that_produced_it() -> None:
    """The join back to the trace. Without both, "what does it believe" has no
    "on what basis" beside it."""
    memory = record()

    assert memory.recorded_in_run == "run-1"
    assert memory.session_id == "sess-1"


def test_a_record_is_frozen() -> None:
    with pytest.raises(ValidationError):
        record().statement = "something else"  # type: ignore[misc]


def test_an_unmodelled_field_is_refused() -> None:
    """Memory content that no policy check ever looked at."""
    with pytest.raises(ValidationError):
        record(sensitivity="low")


def test_a_record_round_trips_through_json() -> None:
    """It rides inside AgentState, which is stored as one jsonb document."""
    memory = record(supersedes=("mem-0",), links=("RISK-3",))

    assert MemoryRecord.model_validate(memory.model_dump(mode="json")) == memory


# --------------------------------------------------------------------------
# One access rule, reused rather than reimplemented
# --------------------------------------------------------------------------


def test_the_document_access_rule_applies_to_a_memory_unchanged() -> None:
    """MemoryRecord satisfies rag.access.Restricted structurally, so a memory
    is filtered by the same function a chunk is (ADR 0008)."""
    context = RetrievalContext.for_actor(
        "priya", project_code="atlas", scopes={"project.docs.read"}
    )

    assert is_authorized(record(), context) is True


def test_a_memory_from_another_project_is_refused() -> None:
    context = RetrievalContext.for_actor(
        "priya", project_code="borealis", scopes={"project.docs.read"}
    )

    assert is_authorized(record(), context) is False


def test_a_memory_whose_scope_the_actor_lacks_is_refused() -> None:
    """A fact learned from a finance document keeps the finance requirement:
    remembering it must not be a way to read it without the scope."""
    context = RetrievalContext.for_actor(
        "priya", project_code="atlas", scopes={"project.docs.read"}
    )

    assert (
        is_authorized(record(required_scope="project.docs.finance.read"), context)
        is False
    )


# --------------------------------------------------------------------------
# Retiring, which is the only kind of forgetting there is
# --------------------------------------------------------------------------


def test_retiring_returns_a_new_record_and_leaves_the_original_alone() -> None:
    memory = record()

    retired = memory.retired(LATER)

    assert retired is not memory
    assert memory.live is True
    assert retired.live is False
    assert retired.superseded_at == LATER


def test_retiring_changes_nothing_else() -> None:
    memory = record()

    retired = memory.retired(LATER)

    assert retired.model_dump(exclude={"superseded_at"}) == memory.model_dump(
        exclude={"superseded_at"}
    )


def test_a_retired_record_cannot_be_retired_again() -> None:
    """The date a memory stopped being believed is a fact somebody may have
    audited; moving it would rewrite that."""
    retired = record().retired(LATER)

    with pytest.raises(ValueError, match="already superseded"):
        retired.retired(LATER)


# --------------------------------------------------------------------------
# The failure paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "memory_id",
        "key",
        "statement",
        "project_code",
        "required_scope",
        "actor",
        "session_id",
        "recorded_in_run",
    ],
)
def test_no_identifier_may_be_blank(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        record(**{field: "   "})


def test_a_kind_nobody_wrote_a_rule_for_is_refused() -> None:
    with pytest.raises(ValidationError, match="kind"):
        record(kind="observation")


def test_a_statement_longer_than_the_cap_is_refused() -> None:
    """The cap is what stops `statement` from becoming a transcript field."""
    with pytest.raises(ValidationError, match="statement"):
        record(statement="x" * (STATEMENT_MAX_CHARS + 1))


def test_a_key_longer_than_the_cap_is_refused() -> None:
    with pytest.raises(ValidationError, match="key"):
        record(key="k" * (KEY_MAX_CHARS + 1))


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_stays_a_probability(confidence: float) -> None:
    with pytest.raises(ValidationError, match="confidence"):
        record(confidence=confidence)


def test_a_record_cannot_supersede_itself() -> None:
    """It would retire the replacement at the moment it was written."""
    with pytest.raises(ValidationError, match="supersede"):
        record(supersedes=("mem-1",))


def test_superseding_a_blank_id_is_refused() -> None:
    with pytest.raises(ValidationError, match="supersedes"):
        record(supersedes=("",))


def test_a_blank_link_is_refused() -> None:
    with pytest.raises(ValidationError, match="links"):
        record(links=(" ",))


def test_a_memory_cannot_be_retired_before_it_was_recorded() -> None:
    with pytest.raises(ValidationError, match="superseded_at"):
        record(superseded_at=datetime(2026, 9, 7, tzinfo=UTC))
