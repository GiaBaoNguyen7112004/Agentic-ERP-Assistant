"""The fake keeps the rules the SQL will have to keep: every read is scoped,
nothing is deleted, and a session never has two open tasks."""

import pytest
from tests.memory.builders import (
    LATER,
    LATEST,
    RECORDED,
    make_intent,
    make_record,
    make_scope,
)

from agentic_erp_assistant.memory.store import (
    InMemoryMemoryStore,
    IntentStorePort,
    MemoryStorePort,
)


def store() -> InMemoryMemoryStore:
    return InMemoryMemoryStore()


def test_the_fake_satisfies_both_ports_structurally() -> None:
    """One adapter implements both against one database, so the fake has to
    prove both are implementable together."""
    assert isinstance(store(), MemoryStorePort)
    assert isinstance(store(), IntentStorePort)


# --------------------------------------------------------------------------
# Writing and retiring
# --------------------------------------------------------------------------


def test_a_written_record_comes_back_live() -> None:
    kept = store()
    kept.write(make_record())

    assert kept.live(make_scope()) == (make_record(),)


def test_writing_the_same_record_twice_leaves_one() -> None:
    """The id is derived from the content, so a re-proposal is ordinary; a store
    that forked or raised would turn that into a failed consolidation."""
    kept = store()

    kept.write(make_record())
    kept.write(make_record())

    assert len(kept.live(make_scope())) == 1


def test_retiring_hands_back_what_it_retired() -> None:
    """The writer has to audit a forget against each id; a store returning a
    count would leave it guessing which."""
    kept = store()
    kept.write(make_record())

    retired = kept.supersede(["mem-1"], at=LATER)

    assert [record.memory_id for record in retired] == ["mem-1"]
    assert retired[0].superseded_at == LATER


def test_a_retired_record_is_kept_and_simply_stops_being_live() -> None:
    """Forgetting is superseding: a deleted memory takes the other half of its
    audit trail with it."""
    kept = store()
    kept.write(make_record())

    kept.supersede(["mem-1"], at=LATER)

    assert kept.live(make_scope()) == ()
    assert kept.records["mem-1"].superseded_at == LATER


def test_retiring_an_unknown_or_already_retired_id_is_not_an_error() -> None:
    """The caller is asking for a state of the world, and a second call after a
    crash has to be able to finish the job."""
    kept = store()
    kept.write(make_record())
    kept.supersede(["mem-1"], at=LATER)

    assert kept.supersede(["mem-1", "mem-nope"], at=LATEST) == ()


# --------------------------------------------------------------------------
# Every read is scoped
# --------------------------------------------------------------------------


def test_another_projects_memory_is_never_returned() -> None:
    kept = store()
    kept.write(make_record(memory_id="mem-b", project_code="borealis"))

    assert kept.live(make_scope()) == ()


def test_another_actors_preference_is_never_returned() -> None:
    kept = store()
    kept.write(make_record(memory_id="mem-m", actor="marco"))

    assert kept.live(make_scope()) == ()


def test_another_sessions_intent_projection_is_never_returned() -> None:
    kept = store()
    kept.write(make_record(memory_id="mem-o", kind="intent", key="intent:x", session_id="sess-9"))

    assert kept.live(make_scope()) == ()


def test_a_project_decision_crosses_sessions_and_actors() -> None:
    """Decisions outlive both, which is why they are worth storing."""
    kept = store()
    kept.write(
        make_record(
            memory_id="mem-d",
            kind="decision",
            key="deployment_window",
            statement="The cutover happens on a Thursday evening.",
            actor="marco",
            session_id="sess-other",
        )
    )

    assert len(kept.live(make_scope())) == 1


def test_kinds_narrows_without_changing_the_bounds() -> None:
    kept = store()
    kept.write(make_record())
    kept.write(
        make_record(memory_id="mem-d", kind="decision", key="deployment_window")
    )

    found = kept.live(make_scope(), kinds=("decision",))

    assert [record.memory_id for record in found] == ["mem-d"]


def test_records_come_back_newest_first() -> None:
    """Recall keeps the head of the list when the budget is tight, so an
    unordered store would make that truncation arbitrary."""
    kept = store()
    kept.write(make_record(memory_id="mem-old", key="a", recorded_at=RECORDED))
    kept.write(make_record(memory_id="mem-new", key="b", recorded_at=LATER))

    assert [record.memory_id for record in kept.live(make_scope())] == [
        "mem-new",
        "mem-old",
    ]


def test_the_limit_takes_the_newest() -> None:
    kept = store()
    kept.write(make_record(memory_id="mem-old", key="a", recorded_at=RECORDED))
    kept.write(make_record(memory_id="mem-new", key="b", recorded_at=LATER))

    assert [r.memory_id for r in kept.live(make_scope(), limit=1)] == ["mem-new"]


# --------------------------------------------------------------------------
# Hydration -- what makes the vector index an optimization
# --------------------------------------------------------------------------


def test_hydrating_returns_the_records_for_known_ids() -> None:
    kept = store()
    kept.write(make_record())

    assert kept.by_id(["mem-1"], make_scope()) == (make_record(),)


def test_hydrating_drops_a_retired_id() -> None:
    """The index may be slow to retire a memory; this is where that stops
    mattering."""
    kept = store()
    kept.write(make_record())
    kept.supersede(["mem-1"], at=LATER)

    assert kept.by_id(["mem-1"], make_scope()) == ()


def test_hydrating_drops_an_out_of_scope_id() -> None:
    kept = store()
    kept.write(make_record(memory_id="mem-b", project_code="borealis"))

    assert kept.by_id(["mem-b"], make_scope()) == ()


def test_hydrating_an_unknown_id_is_not_an_error() -> None:
    assert store().by_id(["mem-nope"], make_scope()) == ()


# --------------------------------------------------------------------------
# One open task per session
# --------------------------------------------------------------------------


def test_the_open_task_comes_back_for_its_own_session() -> None:
    kept = store()
    kept.save_intent(make_intent())

    found = kept.open_intent(make_scope())

    assert found is not None
    assert found.intent_id == "int-1"


def test_another_sessions_task_is_not_this_sessions_task() -> None:
    kept = store()
    kept.save_intent(make_intent(session_id="sess-9"))

    assert kept.open_intent(make_scope()) is None


def test_a_closed_task_is_history_and_is_not_returned() -> None:
    kept = store()
    kept.save_intent(make_intent().close(at=LATER))

    assert kept.open_intent(make_scope()) is None


def test_a_session_may_not_hold_two_open_tasks() -> None:
    """The fake's version of the partial unique index. Two open tasks mean
    recall has to choose, and the one it chooses is the one whose slots leak."""
    kept = store()
    kept.save_intent(make_intent())

    with pytest.raises(ValueError, match="already has"):
        kept.save_intent(make_intent(intent_id="int-2", goal="Check the M2 budget"))


def test_switching_is_allowed_because_it_closes_the_old_task_first() -> None:
    kept = store()
    kept.save_intent(make_intent())

    previous, fresh = make_intent().switch_to(
        "Check the M2 budget", intent_id="int-2", at=LATER
    )
    kept.save_intent(previous)
    kept.save_intent(fresh)

    found = kept.open_intent(make_scope())
    assert found is not None
    assert found.intent_id == "int-2"


def test_updating_the_same_task_is_not_a_clash() -> None:
    kept = store()
    kept.save_intent(make_intent())

    kept.save_intent(make_intent().continue_with({"quarter": "Q4"}, at=LATER))

    found = kept.open_intent(make_scope())
    assert found is not None
    assert found.confirmed_slots["quarter"] == "Q4"
