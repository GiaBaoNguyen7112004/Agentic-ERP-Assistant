"""A candidate has no authority, a scope has no ambiguity, and a verdict cannot
contradict itself."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.memory.models import (
    REASON_MAX_CHARS,
    MemoryCandidate,
    MemoryDecision,
    MemoryScope,
    bounds,
    in_bounds,
    memory_id,
)
from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.state.memory import MemoryRecord


def scope(**overrides: object) -> MemoryScope:
    fields: dict[str, object] = {
        "project_code": "atlas",
        "session_id": "sess-1",
        "scopes": frozenset({"project.docs.read"}),
    }
    fields.update(overrides)
    actor = fields.pop("actor", "priya")
    return MemoryScope.for_actor(actor, **fields)  # type: ignore[arg-type]


def candidate(**overrides: object) -> MemoryCandidate:
    fields: dict[str, object] = {
        "kind": "preference",
        "key": "reply_language",
        "statement": "Prefers replies written in Vietnamese.",
        "confidence": 0.9,
        "required_scope": "project.docs.read",
    }
    fields.update(overrides)
    return MemoryCandidate(**fields)  # type: ignore[arg-type]


def record(**overrides: object) -> MemoryRecord:
    fields: dict[str, object] = {
        "memory_id": "mem-1",
        "kind": "preference",
        "key": "reply_language",
        "statement": "Prefers replies written in Vietnamese.",
        "project_code": "atlas",
        "required_scope": "project.docs.read",
        "actor": "priya",
        "session_id": "sess-1",
        "recorded_in_run": "run-1",
        "recorded_at": datetime(2026, 9, 8, tzinfo=UTC),
        "confidence": 0.9,
    }
    fields.update(overrides)
    return MemoryRecord(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# MemoryScope: document access plus a session, stated as composition
# --------------------------------------------------------------------------


def test_a_scope_delegates_to_the_access_context_it_wraps() -> None:
    """Composition rather than four copied fields, so there is nothing that can
    drift out of step with the access rule."""
    memory_scope = scope()

    assert memory_scope.actor == "priya"
    assert memory_scope.project_code == "atlas"
    assert memory_scope.scopes == frozenset({"project.docs.read"})


def test_the_wrapped_context_is_the_one_the_access_rule_takes() -> None:
    assert isinstance(scope().access, RetrievalContext)


def test_a_scope_without_a_session_is_refused() -> None:
    """A blank one would make every session's intent look like every other's."""
    with pytest.raises(ValueError, match="session_id"):
        MemoryScope(
            access=RetrievalContext.for_actor("priya", project_code="atlas"),
            session_id="  ",
        )


# --------------------------------------------------------------------------
# bounds: one answer to "is this memory mine?"
# --------------------------------------------------------------------------


def test_a_preference_is_bounded_by_project_and_actor() -> None:
    assert bounds("preference", scope()) == {
        "project_code": "atlas",
        "actor": "priya",
    }


def test_an_intent_is_bounded_by_project_and_session() -> None:
    assert bounds("intent", scope()) == {
        "project_code": "atlas",
        "session_id": "sess-1",
    }


def test_a_decision_is_bounded_by_the_project_alone() -> None:
    """Decisions outlive the session and the person who made them, which is why
    they are worth storing at all."""
    assert bounds("decision", scope()) == {"project_code": "atlas"}


def test_another_actors_preference_is_out_of_bounds() -> None:
    assert in_bounds(record(actor="marco"), scope()) is False


def test_another_sessions_preference_is_still_in_bounds() -> None:
    """A preference follows the person between conversations; only intents and
    session summaries are pinned to one."""
    assert in_bounds(record(session_id="sess-9"), scope()) is True


def test_another_sessions_intent_is_out_of_bounds() -> None:
    assert in_bounds(record(kind="intent", session_id="sess-9"), scope()) is False


# --------------------------------------------------------------------------
# memory_id: derived, so even a refusal has something to name
# --------------------------------------------------------------------------


def test_the_same_proposal_gets_the_same_id_every_time() -> None:
    """A duplicate is then one id appearing twice in the audit, rather than two
    rows nobody can connect."""
    assert memory_id(candidate(), scope()) == memory_id(candidate(), scope())


def test_the_same_sentence_on_two_projects_is_two_memories() -> None:
    """An id that ignored the project would let one overwrite the other."""
    assert memory_id(candidate(), scope()) != memory_id(
        candidate(), scope(project_code="borealis")
    )


def test_a_preference_is_scoped_to_its_actor_in_the_id() -> None:
    assert memory_id(candidate(), scope()) != memory_id(
        candidate(), scope(actor="marco")
    )


def test_a_project_decision_is_not_scoped_to_actor_or_session() -> None:
    proposal = candidate(kind="decision", key="deployment_window")

    assert memory_id(proposal, scope()) == memory_id(
        proposal, scope(actor="marco", session_id="sess-9")
    )


def test_an_intent_is_scoped_to_its_session_in_the_id() -> None:
    proposal = candidate(kind="intent", key="open_intent")

    assert memory_id(proposal, scope()) != memory_id(
        proposal, scope(session_id="sess-9")
    )


# --------------------------------------------------------------------------
# MemoryCandidate: a proposal, with no identity and no home
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["memory_id", "project_code", "recorded_at"])
def test_a_candidate_cannot_nominate_its_own_identity(field: str) -> None:
    """Those come from the scope and the clock when a verdict is acted on, which
    is what stops a proposer choosing which project a memory lands in."""
    with pytest.raises(ValidationError):
        candidate(**{field: "atlas"})


def test_a_candidate_is_frozen() -> None:
    with pytest.raises(ValidationError):
        candidate().statement = "something else"  # type: ignore[misc]


# --------------------------------------------------------------------------
# MemoryDecision: the invalid verdicts simply have no instances
# --------------------------------------------------------------------------


def test_a_rejection_must_name_the_rule_that_refused_it() -> None:
    with pytest.raises(ValidationError, match="rejection"):
        MemoryDecision(decision="reject")


def test_only_a_rejection_may_carry_a_rejection_reason() -> None:
    with pytest.raises(ValidationError, match="rejection"):
        MemoryDecision(decision="write", rejection="duplicate")


def test_an_update_must_name_what_it_retires() -> None:
    """One that names nothing is a write wearing the wrong label."""
    with pytest.raises(ValidationError, match="supersedes"):
        MemoryDecision(decision="update")


def test_a_write_may_not_retire_anything() -> None:
    with pytest.raises(ValidationError, match="supersedes"):
        MemoryDecision(decision="write", supersedes=("mem-old",))


def test_forget_is_not_a_verdict_on_a_candidate() -> None:
    """It is what happens to the record an update replaced, recorded by the
    writer against that record's own id."""
    with pytest.raises(ValidationError, match="forget"):
        MemoryDecision(decision="forget")


def test_a_reason_longer_than_the_cap_is_refused() -> None:
    with pytest.raises(ValidationError, match="reason"):
        MemoryDecision(decision="write", reason="x" * (REASON_MAX_CHARS + 1))


def test_stores_is_true_for_the_two_verdicts_that_write() -> None:
    assert MemoryDecision(decision="write").stores is True
    assert (
        MemoryDecision(decision="update", supersedes=("mem-old",)).stores is True
    )
    assert (
        MemoryDecision(decision="reject", rejection="duplicate").stores is False
    )
