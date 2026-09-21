"""The guard the fixture never had: the two data sources cannot drift apart.

``data/erp/project.json`` (what the tools serve) and ``data/documents/`` (what
retrieval serves) state facts about the same fictional programme. Nothing else
in the suite loads the real files -- every other test copies them or fakes
them, which is exactly why a mismatch between the sources stayed invisible
until a tool answer said "two open risks" where the register said six.

Every test here reads the *real* files, read-only, through a store built with
``from_mapping`` and no path -- a store whose writes refuse, so a consistency
test can never become the thing it guards against. When one of these fails,
either side of the mirror was edited without the other: fix the data, and
update the fixture readme if the mirroring rule itself changed.
"""

import csv
import json
from pathlib import Path

import pytest

from agentic_erp_assistant.erp.mock import ErpNotPersistedError, MockErp

REPO_ROOT = Path(__file__).resolve().parents[2]
ERP_JSON = REPO_ROOT / "data" / "erp" / "project.json"
DOCUMENTS = REPO_ROOT / "data" / "documents"
RISK_REGISTER_CSV = DOCUMENTS / "risk-register.csv"
MANIFEST_JSON = DOCUMENTS / "manifest.json"


def _store() -> MockErp:
    """The real dataset, with no write path -- a consistency test reads."""
    return MockErp.from_mapping(json.loads(ERP_JSON.read_text(encoding="utf-8")))


def _register_rows() -> list[dict[str, str]]:
    with RISK_REGISTER_CSV.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _document_text(name: str) -> str:
    return (DOCUMENTS / name).read_text(encoding="utf-8")


def _manifest_ids() -> set[str]:
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    return {row["document_id"] for row in manifest["documents"]}


# --------------------------------------------------------------------------
# The risks: the register is the record of truth, the dataset mirrors it
# --------------------------------------------------------------------------


def test_every_register_row_is_mirrored_on_every_field() -> None:
    store = _store()
    mirrored = {risk.risk_id: risk for risk in store.risks_for("atlas")}

    for row in _register_rows():
        risk = mirrored.get(row["risk_id"])
        assert risk is not None, (
            f"register row {row['risk_id']} has no row in {ERP_JSON.name}; "
            "the dataset mirrors the register, so the row belongs in both"
        )
        assert risk.title == row["title"], row["risk_id"]
        assert risk.severity == row["severity"], row["risk_id"]
        assert risk.status == row["status"], row["risk_id"]


def test_the_dataset_has_no_risk_row_the_register_does_not() -> None:
    store = _store()
    register_ids = {row["risk_id"] for row in _register_rows()}
    mirrored_ids = {risk.risk_id for risk in store.risks_for("atlas")}

    assert mirrored_ids == register_ids, (
        f"rows only in the dataset: {sorted(mirrored_ids - register_ids)}; "
        f"rows only in the register: {sorted(register_ids - mirrored_ids)}"
    )


def test_the_mirrored_risk_ids_cover_the_register_without_gaps() -> None:
    """The ids must stay the register's own dense run, because create_risk
    mints the next id from the largest suffix -- a dropped row would make the
    next write collide with a row the register still owns."""
    register_ids = sorted(row["risk_id"] for row in _register_rows())

    assert register_ids == [f"R-{number}" for number in range(1, len(register_ids) + 1)]


# --------------------------------------------------------------------------
# The numbers: what a tool serves is what the documents state
# --------------------------------------------------------------------------


def test_the_atlas_budget_figures_match_the_budget_summary() -> None:
    store = _store()
    budget = store.budget("atlas")
    text = _document_text("sources/budget-summary-q3.md")

    assert budget is not None
    for figure in (budget.approved_usd, budget.spent_usd, budget.forecast_at_completion_usd):
        assert f"{figure:,.0f}" in text, (
            f"{figure:,.0f} is not in the Atlas budget summary"
        )
    assert budget.as_of in text


def test_the_orion_budget_figures_match_the_orion_report() -> None:
    store = _store()
    budget = store.budget("orion")
    text = _document_text("orion-status-report-2026-09.md")

    assert budget is not None
    for figure in (budget.approved_usd, budget.spent_usd, budget.forecast_at_completion_usd):
        assert f"{figure:,.0f}" in text, (
            f"{figure:,.0f} is not in the Orion status report"
        )
    assert budget.as_of in text


def test_the_atlas_milestone_dates_match_the_status_report() -> None:
    store = _store()
    text = _document_text("status-report-2026-09.md")

    for milestone_id in ("M1", "M2", "M3"):
        milestone = store.milestone(milestone_id)
        assert milestone is not None
        assert milestone.due_on in text, (
            f"{milestone_id}'s due date {milestone.due_on} is not in the "
            "September status report"
        )

    m2 = store.milestone("M2")
    assert m2 is not None
    assert m2.days_late == 2 and "two days late" in text


def test_the_orion_milestone_dates_match_the_orion_report() -> None:
    store = _store()
    text = _document_text("orion-status-report-2026-09.md")

    for milestone_id in ("O1", "O2", "O3"):
        milestone = store.milestone(milestone_id)
        assert milestone is not None
        assert milestone.due_on in text, (
            f"{milestone_id}'s due date {milestone.due_on} is not in the "
            "Orion status report"
        )


def test_the_sprint_figures_match_the_sprint_report() -> None:
    store = _store()
    text = _document_text("sprint-13-report.md")
    sprint = store.sprint("SPR-13")

    assert sprint is not None
    assert f"committed {sprint.points_committed} story points" in text
    assert f"{sprint.points_completed} points are complete" in text
    assert f"{sprint.days_remaining} working days remaining" in text

    # Sprint 12's row matches the delivered figure the report carries; the
    # full Sprint 11 history is document-only on purpose (fixture readme).
    sprint_12 = store.sprint("SPR-12")
    assert sprint_12 is not None
    assert f"{sprint_12.points_completed} of {sprint_12.points_committed}" in text


# --------------------------------------------------------------------------
# The citation-id split: which source ids name documents, and which do not
# --------------------------------------------------------------------------


def test_a_tool_citation_that_names_a_document_names_a_manifest_document() -> None:
    """The manifest's promise: where a document backs a field the mock ERP
    also serves, its document_id reuses the record's source_id. Today that is
    the Atlas budget summary and the sprint-13 report."""
    store = _store()
    store_ids = {
        *(row.source_id for row in store.projects),
        *(row.source_id for row in store.milestones),
        *(row.source_id for row in store.sprints),
        *(row.source_id for row in store.budgets),
        *(row.source_id for row in store.risks),
    }

    assert store_ids & _manifest_ids() == {"budget-summary-q3", "sprint-13-report"}


def test_the_non_document_citation_families_are_exactly_the_documented_ones() -> None:
    """A bare source_id that names no document renders as a plain ``erp``
    chip -- fine, as long as the set is explicit. A new id family outside it
    means either a backing document was forgotten or the readme's rule needs
    updating to say so."""
    store = _store()
    store_ids = {
        *(row.source_id for row in store.projects),
        *(row.source_id for row in store.milestones),
        *(row.source_id for row in store.sprints),
        *(row.source_id for row in store.budgets),
        *(row.source_id for row in store.risks),
    }
    documented_non_documents = {
        "project-atlas",
        "project-orion",
        "milestone-m1",
        "milestone-m2",
        "milestone-m3",
        "milestone-o1",
        "milestone-o2",
        "milestone-o3",
        "sprint-12-report",
        # Orion's budget figures live in its status report; no separate
        # budget document exists for it, unlike Atlas's budget-summary-q3.
        "budget-orion-q3",
        *(f"risk-r-{number}" for number in range(1, 9)),
    }

    unexplained = store_ids - _manifest_ids() - documented_non_documents
    assert not unexplained, (
        f"{sorted(unexplained)} name no document and no documented family; "
        "the fixture readme's citation rule must say what they are"
    )


# --------------------------------------------------------------------------
# The store this module builds can never become the thing it guards against
# --------------------------------------------------------------------------


def test_the_consistency_store_cannot_write() -> None:
    """A pathless store refuses writes, so a consistency test can never edit
    the fixture it is reading -- the same guard the write tests rely on."""
    with pytest.raises(ErpNotPersistedError):
        _store().create_risk(project_id="atlas", title="x", severity="low")