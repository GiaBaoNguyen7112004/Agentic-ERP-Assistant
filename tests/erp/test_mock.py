"""A fixture nobody validates is a fixture that rots quietly."""

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.erp.mock import DEFAULT_DATASET_PATH, Milestone, MockErp


@pytest.fixture
def erp() -> MockErp:
    """A fresh store per test: writes must not leak between them."""
    return MockErp.load()


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


def test_a_write_does_not_touch_the_file_on_disk(erp: MockErp) -> None:
    erp.create_risk(project_id="atlas", title="in memory only", severity="low")

    assert len(MockErp.load().risks_for("atlas")) == len(erp.risks_for("atlas")) - 1
