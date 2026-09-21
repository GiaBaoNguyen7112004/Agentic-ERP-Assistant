"""ADR 0026: what a turn's actor is told exists, before it searches anything.

:func:`build_catalogue` is a pure filter over a loaded manifest and a
:class:`RetrievalContext` -- no file is read here, and the policy it applies
is exactly :func:`~agentic_erp_assistant.rag.access.is_authorized`, reused
rather than a second comparison. These tests prove the filter, not the
policy underneath it -- ``tests/rag/test_access.py`` already owns that.
"""

from datetime import date

from agentic_erp_assistant.context.catalogue import build_catalogue, DocumentCatalogue
from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.rag.manifest import ManifestEntry


def entry(
    document_id: str,
    *,
    project_code: str = "atlas",
    required_scope: str = "project.docs.read",
    title: str | None = None,
    document_type: str = "status_report",
) -> ManifestEntry:
    return ManifestEntry(
        document_id=document_id,
        path=f"{document_id}.md",
        title=title or document_id,
        document_type=document_type,
        project_code=project_code,
        required_scope=required_scope,
        classification="internal",
        effective_date=date(2026, 9, 1),
    )


def test_a_readable_document_is_listed() -> None:
    manifest = {"status-report-2026-09": entry("status-report-2026-09")}
    context = RetrievalContext.for_actor(
        "priya", project_code="atlas", scopes=["project.docs.read"]
    )

    catalogue = build_catalogue(manifest, context)

    assert bool(catalogue) is True
    assert [e.document_id for e in catalogue.entries] == ["status-report-2026-09"]


def test_a_document_needing_a_scope_the_actor_lacks_is_left_off() -> None:
    """The finance-scoped budget summary, read by an actor with only the
    ordinary docs scope -- the same shape priya vs. the Q3 PDF."""
    manifest = {
        "budget-summary-q3": entry(
            "budget-summary-q3", required_scope="project.docs.finance.read"
        )
    }
    context = RetrievalContext.for_actor(
        "priya", project_code="atlas", scopes=["project.docs.read"]
    )

    catalogue = build_catalogue(manifest, context)

    assert catalogue.entries == ()
    assert bool(catalogue) is False


def test_a_document_from_another_project_is_left_off() -> None:
    manifest = {"orion-status": entry("orion-status", project_code="orion")}
    context = RetrievalContext.for_actor(
        "priya", project_code="atlas", scopes=["project.docs.read"]
    )

    catalogue = build_catalogue(manifest, context)

    assert catalogue.entries == ()


def test_an_actor_with_the_finance_scope_sees_the_finance_document_too() -> None:
    manifest = {
        "status-report-2026-09": entry("status-report-2026-09"),
        "budget-summary-q3": entry(
            "budget-summary-q3", required_scope="project.docs.finance.read"
        ),
    }
    context = RetrievalContext.for_actor(
        "wei",
        project_code="atlas",
        scopes=["project.docs.read", "project.docs.finance.read"],
    )

    catalogue = build_catalogue(manifest, context)

    assert {e.document_id for e in catalogue.entries} == {
        "status-report-2026-09",
        "budget-summary-q3",
    }


def test_entries_carry_the_title_type_and_date_the_prompt_needs() -> None:
    manifest = {
        "risk-register": entry(
            "risk-register", title="Atlas Risk Register", document_type="risk_register"
        )
    }
    context = RetrievalContext.for_actor(
        "priya", project_code="atlas", scopes=["project.docs.read"]
    )

    catalogue = build_catalogue(manifest, context)

    (result,) = catalogue.entries
    assert result.title == "Atlas Risk Register"
    assert result.document_type == "risk_register"
    assert result.effective_date == "2026-09-01"


def test_an_actor_with_no_scopes_gets_an_empty_catalogue_not_an_error() -> None:
    manifest = {"status-report-2026-09": entry("status-report-2026-09")}
    context = RetrievalContext.for_actor("guest", project_code="atlas", scopes=[])

    catalogue = build_catalogue(manifest, context)

    assert catalogue == DocumentCatalogue(entries=())


def test_manifest_order_is_preserved() -> None:
    manifest = {
        "b-doc": entry("b-doc"),
        "a-doc": entry("a-doc"),
    }
    context = RetrievalContext.for_actor(
        "priya", project_code="atlas", scopes=["project.docs.read"]
    )

    catalogue = build_catalogue(manifest, context)

    assert [e.document_id for e in catalogue.entries] == ["b-doc", "a-doc"]
