"""What the model is told exists, before it is ever asked to search for it.

Retrieval's own tool (``search_project_documents``) is a plain query string --
the model reads a description of the tool and nothing about the corpus behind
it. That is enough for a single, unambiguous question, and it is not enough
for a request that names a document by its format ("the risk register CSV",
"the Q3 budget summary PDF"): with no catalogue to check against, the only
honest-sounding move left to the model is to guess whether such a document
exists, and a model that guesses wrong guesses ``refuse`` (see
``multi-document-turn-plan.md``, RC1, and ADR 0025's trace).

The fix is not a smarter tool description. It is telling the model, plainly
and in the system role, which documents this turn's actor may search -- the
same fact ``rag/access.py::is_authorized`` already decides for every chunk a
search returns, read one level up, before any search has run.

Why this is a snapshot, not a query
------------------------------------

:func:`build_catalogue` is a pure filter over the manifest this project loads
once at process startup (``composition/resources.py::AppResources.manifest``)
against the same :class:`~agentic_erp_assistant.rag.access.RetrievalContext`
the turn's retriever is already bound to. No file is read, no index is
queried, and the check reuses ``is_authorized`` rather than a second
comparison of project code and scope -- the one-rule promise
``rag/access.py``'s own module docstring makes would mean nothing if this
were the place a second, drifting copy of it lived.

Why an unreadable document is left off the list, not shown as locked
----------------------------------------------------------------------

Listing a confidential title to an actor who may not open it -- even
labelled "you cannot read this" -- discloses the one fact access control
exists to hide: that the document exists at all, and roughly what it is
about. The catalogue only ever names what ``context`` is authorized to
search; a named document outside that set is handled by wording, not by
data (see :func:`~agentic_erp_assistant.llm.prompts.render_catalogue`): the
model is told to say it cannot access something it is told about by the
user, never to consult a list of what it may not see.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from agentic_erp_assistant.rag.access import RetrievalContext, is_authorized
from agentic_erp_assistant.rag.manifest import ManifestEntry

__all__ = ["build_catalogue", "CatalogueEntry", "DocumentCatalogue"]


@dataclass(frozen=True)
class CatalogueEntry:
    """One document the model may name or search this turn.

    A view over a few of :class:`~agentic_erp_assistant.rag.manifest.
    ManifestEntry`'s own fields, deliberately not the whole row:
    ``required_scope`` and ``classification`` are *why* a document is or is
    not on this list, never something to explain to the model in the
    prompt itself, and ``path``/``content_hash`` are storage detail it has
    no use for.
    """

    document_id: str
    """The citation ``source_id`` -- what a search result and this entry
    name the same document with."""

    title: str
    document_type: str
    effective_date: str
    """ISO 8601 (``YYYY-MM-DD``). A string here, not a ``date``: this object
    exists to be rendered into a prompt, and every other field already is
    one."""


@dataclass(frozen=True)
class DocumentCatalogue:
    """Every document one turn's actor may search, snapshotted like the
    :class:`~agentic_erp_assistant.rag.access.RetrievalContext` it was built
    from -- entitlements that could change mid-turn would let one turn be
    told about a document under one permission and search it under another,
    the same argument ``RetrievalContext`` itself makes for its own fields.
    """

    entries: tuple[CatalogueEntry, ...]

    def __bool__(self) -> bool:
        return bool(self.entries)


def build_catalogue(
    manifest: Mapping[str, ManifestEntry], context: RetrievalContext
) -> DocumentCatalogue:
    """Every manifest row ``context`` is authorized to read, in manifest order.

    Args:
        manifest: The whole corpus's rows, by ``document_id`` -- as already
            loaded once at process startup
            (``composition/resources.py::AppResources.manifest``). Never
            read from disk here.
        context: The turn's own entitlements -- the same object its
            retriever is bound through
            (``rag/retriever.py::RetrievalService.for_context``).

    Returns:
        A :class:`DocumentCatalogue` naming only what ``context`` may
        search. Empty, never an error, for an actor entitled to nothing --
        the same "deny by default" ``is_authorized`` already applies to
        every chunk.
    """
    entries = tuple(
        CatalogueEntry(
            document_id=entry.document_id,
            title=entry.title,
            document_type=entry.document_type,
            effective_date=entry.effective_date.isoformat(),
        )
        for entry in manifest.values()
        if is_authorized(entry, context)
    )
    return DocumentCatalogue(entries=entries)
