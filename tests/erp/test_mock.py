"""A fixture nobody validates is a fixture that rots quietly."""

import json

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.erp.mock import (
    DEFAULT_DATASET_PATH,
    ErpAccessError,
    ErpNotPersistedError,
    Milestone,
    MockErp,
)


@pytest.fixture
def erp(erp_file) -> MockErp:
    """A fresh store per test, loaded from a disposable copy of the fixture.

    Writes really reach the file now, so the copy is what keeps one test's
    write from being the next test's row -- and the repo's fixture from
    becoming test exhaust.
    """
    return MockErp.load(erp_file)


# --------------------------------------------------------------------------
# The fixture on disk
# --------------------------------------------------------------------------


def test_the_repo_fixture_loads_and_validates(erp: MockErp) -> None:
    assert erp.projects and erp.milestones and erp.sprints and erp.budgets


def test_the_fixture_lives_where_a_reviewer_would_look() -> None:
    assert DEFAULT_DATASET_PATH.exists()
    assert DEFAULT_DATASET_PATH.parts[-3:] == ("data", "erp", "project.json")


def test_every_record_carries_something_to_cite(erp: MockErp) -> None:
    """A row that cannot be cited is a row a grounded answer cannot use, and
    finding that out at answer time is too late."""
    rows = (*erp.projects, *erp.milestones, *erp.sprints, *erp.budgets, *erp.risks)

    assert all(row.source_id.strip() for row in rows)


def test_a_misspelled_column_fails_at_load_not_at_query() -> None:
    """extra='forbid' is what turns 'daysLate' into a named error here rather
    than a tool that answers with nothing."""
    with pytest.raises(ValidationError):
        Milestone(
            milestone_id="M2",
            project_id="atlas",
            title="Finance module cutover",
            due_on="2026-09-11",
            schedule_status="at_risk",
            daysLate=2,
            source_id="milestone-m2",
        )


def test_a_severity_outside_the_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="severity"):
        MockErp.from_mapping(
            {
                "projects": [],
                "milestones": [],
                "sprints": [],
                "budgets": [],
                "risks": [
                    {
                        "risk_id": "R-9",
                        "project_id": "atlas",
                        "title": "x",
                        "severity": "catastrophic",
                        "status": "open",
                        "source_id": "risk-r-9",
                    }
                ],
            }
        )


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def test_a_known_milestone_comes_back(erp: MockErp) -> None:
    found = erp.milestone("M2")

    assert found is not None
    assert found.source_id == "milestone-m2"


def test_an_unknown_id_is_none_and_not_an_exception(erp: MockErp) -> None:
    """Which milestone was asked about is the model's guess; a bad guess is an
    ordinary answer, not a broken backend."""
    assert erp.milestone("M9") is None


def test_a_project_with_no_risks_reads_as_none_open(erp: MockErp) -> None:
    """Empty, not None: 'none open' is a reportable answer, while None would
    make the handler guess whether it meant that or 'no such project'."""
    assert erp.risks_for("no-such-project") == ()


def test_closed_risks_are_kept_but_not_open(erp: MockErp) -> None:
    """The dataset mirrors the register, which keeps closed rows on file --
    a proposed risk duplicated against a closed one is still a duplicate --
    but what list_risks promises is the open risks, and the store does the
    filtering so the rule is not a check a handler has to remember."""
    all_ids = [risk.risk_id for risk in erp.risks_for("atlas")]
    open_ids = [risk.risk_id for risk in erp.open_risks_for("atlas")]

    assert "R-5" in all_ids and "R-6" in all_ids
    assert "R-5" not in open_ids and "R-6" not in open_ids
    assert len(open_ids) == 6


def test_a_risk_without_a_status_fails_at_load() -> None:
    """Required, no default: the register distinguishes open from closed, and
    a row that forgot to say which is a row list_risks would misreport."""
    with pytest.raises(ValidationError, match="status"):
        MockErp.from_mapping(
            {
                "projects": [],
                "milestones": [],
                "sprints": [],
                "budgets": [],
                "risks": [
                    {
                        "risk_id": "R-1",
                        "project_id": "atlas",
                        "title": "x",
                        "severity": "high",
                        "source_id": "risk-r-1",
                    }
                ],
            }
        )


def test_the_budget_is_found_by_project(erp: MockErp) -> None:
    budget = erp.budget("atlas")

    assert budget is not None
    assert budget.source_id == "budget-summary-q3"


# --------------------------------------------------------------------------
# The one write
# --------------------------------------------------------------------------


def test_creating_a_risk_records_it(erp: MockErp) -> None:
    before = len(erp.risks_for("atlas"))

    created = erp.create_risk(
        project_id="atlas", title="Vendor may miss the test window", severity="medium"
    )

    assert len(erp.risks_for("atlas")) == before + 1
    assert erp.risks[-1] is created


def test_a_created_risk_can_be_cited(erp: MockErp) -> None:
    created = erp.create_risk(project_id="atlas", title="x", severity="low")

    assert created.source_id == f"risk-r-{len(erp.risks)}"


def test_the_store_enforces_no_policy_of_its_own(erp: MockErp) -> None:
    """It writes when asked. Approval and permission are the gateway's job, and
    a store that also enforced them would be a second place to look for the
    rule -- and the place someone forgets to update."""
    erp.create_risk(project_id="atlas", title="unapproved", severity="high")

    assert erp.risks[-1].title == "unapproved"


def test_a_write_persists_to_the_file_on_disk(erp: MockErp, erp_file) -> None:
    """The one property this module exists for: a caller that was told
    'recorded' can reload the file and find it."""
    erp.create_risk(project_id="atlas", title="persisted, not just appended", severity="low")

    reloaded = MockErp.load(erp_file)
    titles = [risk.title for risk in reloaded.risks_for("atlas")]
    assert "persisted, not just appended" in titles


def test_concurrent_writes_do_not_collide_on_the_same_id(erp: MockErp) -> None:
    """Two request threads can now reach one store; without the lock, both
    could read ``len(self.risks)`` before either appended and hand out the
    same id twice."""
    import threading

    errors: list[BaseException] = []

    def write(title: str) -> None:
        try:
            erp.create_risk(project_id="atlas", title=title, severity="low")
        except BaseException as error:  # noqa: BLE001 - captured, not swallowed
            errors.append(error)

    threads = [
        threading.Thread(target=write, args=(f"concurrent risk {i}",))
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    new_titles = {f"concurrent risk {i}" for i in range(8)}
    written = [risk for risk in erp.risks if risk.title in new_titles]
    assert len(written) == 8
    assert len({risk.risk_id for risk in written}) == 8


def test_a_write_keeps_the_documentation_keys(erp: MockErp, erp_file) -> None:
    """The ``_readme`` is review material; a rewrite that erased it would be a
    write that destroyed the thing it was writing to."""
    erp.create_risk(project_id="atlas", title="x", severity="low")

    raw = json.loads(erp_file.read_text(encoding="utf-8"))
    assert "_readme" in raw


def test_a_store_with_no_file_refuses_to_write() -> None:
    """Loudly, not silently: the append-then-vanish behavior this replaces was
    the defect, and a pathless store that quietly kept the row would be it
    again."""
    store = MockErp.from_mapping(
        {
            "projects": [
                {
                    "project_id": "atlas",
                    "name": "Atlas ERP rollout",
                    "source_id": "project-atlas",
                }
            ],
            "milestones": [],
            "sprints": [],
            "budgets": [],
            "risks": [],
        }
    )

    with pytest.raises(ErpNotPersistedError):
        store.create_risk(project_id="atlas", title="nowhere to go", severity="low")

    assert store.risks == [], "a refused write must not leave a row behind"


def test_the_repo_fixture_is_untouched_by_the_write_tests() -> None:
    """The suite above writes to copies; this is the guard that says so. If it
    fails, some fixture stopped using ``erp_file`` and the repo's fixture is
    being edited by a test."""
    raw = json.loads(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"))
    risk_ids = [risk["risk_id"] for risk in raw["risks"]]
    assert risk_ids == ["R-1", "R-2", "R-3", "R-4", "R-5", "R-6", "R-7", "R-8"]


# --------------------------------------------------------------------------
# ProjectErp: a project's slice of the store, and nothing else
# --------------------------------------------------------------------------


def test_a_record_of_another_project_does_not_exist_through_the_view(
    erp: MockErp,
) -> None:
    """The same "does not exist for you" a filtered document gets: a handler
    bound to orion cannot see atlas's M2, even though the record is in the
    file."""
    orion = erp.for_project("orion")

    assert orion.milestone("M2") is None
    assert orion.sprint("SPR-12") is None
    assert orion.budget("atlas") is None
    assert orion.project("atlas") is None
    assert orion.risks_for("atlas") == ()


def test_a_record_of_the_bound_project_is_visible(erp: MockErp) -> None:
    orion = erp.for_project("orion")

    milestone = orion.milestone("O2")

    assert milestone is not None
    assert milestone.title == "Pipeline migration"


def test_the_view_never_widens_what_the_store_would_answer(erp: MockErp) -> None:
    """A milestone that does not exist anywhere reads the same through a view
    as through the store directly -- the view narrows, it never invents."""
    atlas = erp.for_project("atlas")

    assert atlas.milestone("no-such-milestone") is None
    assert erp.milestone("no-such-milestone") is None


def test_create_risk_through_the_matching_view_succeeds(erp: MockErp) -> None:
    orion = erp.for_project("orion")

    created = orion.create_risk(
        project_id="orion", title="Legal has not released the extract", severity="medium"
    )

    assert created.project_id == "orion"
    assert created in erp.risks


def test_create_risk_through_the_wrong_view_is_refused(erp: MockErp) -> None:
    """Defence in depth: the gateway's project check refuses this first, but
    the view must still refuse if it is ever called directly."""
    atlas = erp.for_project("atlas")
    before = len(erp.risks)

    with pytest.raises(ErpAccessError):
        atlas.create_risk(project_id="orion", title="x", severity="low")

    assert len(erp.risks) == before


def test_for_project_returns_a_fresh_view_each_time_over_the_same_store(
    erp: MockErp,
) -> None:
    """Two views of one store still see one write: for_project is a lens, not
    a copy."""
    atlas = erp.for_project("atlas")
    atlas.create_risk(project_id="atlas", title="x", severity="low")

    assert erp.for_project("atlas").risks_for("atlas")[-1].title == "x"
