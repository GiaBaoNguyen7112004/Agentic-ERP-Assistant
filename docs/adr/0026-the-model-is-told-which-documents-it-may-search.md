# 0026 — The model is told which documents it may search

**Status:** Accepted (2026-09-15)

## Context

ADR 0025 stopped a refusal from ending a turn unchecked, but did not explain
*why* trace `run-e4feb394274f42c288be90ed37ad8c8e` refused in the first
place. The request named three documents by file format: "the risk
register CSV", "the Q3 budget summary PDF", "the latest sprint report".
`search_project_documents`'s own description, the only thing the model is
told about the corpus, read:

> Search the project documents -- status reports, meeting notes, contracts
> -- and return the passages that answer a question...

No register, no CSV, no PDF, no sprint report, and nothing anywhere in the
prompt saying which documents exist for this project or which this actor
may open. Two of the three named documents (`risk-register`,
`sprint-13-report`) are `project.docs.read` -- priya's own scope. The model
had a working, offered search tool and no way to know it would find
anything, so it guessed, and the guess was `refuse`.

`data/documents/manifest.json` already answers this exactly:
`document_id`, `title`, `document_type`, `required_scope`,
`project_code`. It is loaded once at process startup
(`composition/resources.py::AppResources.manifest`, already built for
`/api/documents/{id}`) and none of it reaches a routing prompt.

## Decision

**A typed `DocumentCatalogue`, filtered by this turn's own entitlements,
rendered into the system role of the one prompt that decides whether to
search: the planner's.**

- `context/catalogue.py::build_catalogue(manifest, context) ->
  DocumentCatalogue` is a pure filter, no file read: for every row in the
  already-loaded manifest, keep it if
  `rag/access.py::is_authorized(entry, context)` says so. Reused rather
  than a second scope comparison -- `is_authorized` is already the one
  place "what may this actor read" is decided for a chunk, and a
  manifest row and a chunk both carry the same two fields
  (`project_code`, `required_scope`) that decide it.
- `context: DocumentCatalogue`'s job is what the package's own name says:
  what gets into the prompt, and why the rest did not. `CatalogueEntry`
  carries only `document_id`, `title`, `document_type`,
  `effective_date` -- not `required_scope` or `classification`, which are
  *why* a row is or is not on the list, never something to explain to the
  model.
- **A document outside `context`'s project or scope is left off the list
  entirely, never shown as locked.** Listing a confidential title to an
  actor who may not open it discloses the fact access control exists to
  hide -- that the document exists, and roughly what it is about. The
  model's script for a document the user names that is not on the list is
  wording, not data: "say you cannot access it; never say it does not
  exist, and never guess which one it is" (`llm/prompts.py::
  render_catalogue`).
- **The catalogue reaches exactly one prompt: `decide`'s.**
  `LLMGateway.catalogue: DocumentCatalogue | None = None`, bound at
  construction alongside `principal` and read only by `decide()`;
  `answer`, `declare`, and the memory-proposal gateway built without one
  never see it. Composing from evidence already retrieved, and declaring
  what a reply needs in the abstract, have no occasion to weigh whether a
  specific document exists -- only the routing decision does, and only
  routing was the failure. `build_planner_messages` gains a `catalogue`
  parameter; every other message builder's signature is unchanged.
  `system_content(principal, catalogue=None)` appends the catalogue block
  after the principal block when both are given, and ignores a catalogue
  passed with no principal -- a catalogue is a fact about *this actor's*
  entitlements, and there is no actor to state it for on an unbound call
  (a replay, `eval/routing.py`'s comparison).
- **One snapshot, shared.** `composition/turn.py` builds the turn's
  `RetrievalContext` once and reuses it for both the retriever
  (`resources.retrieval.for_context`) and the catalogue
  (`build_catalogue(resources.manifest, ...)`), rather than building two
  contexts that could drift. What the model is told it may search and what
  the retriever will actually let it read come from the same entitlement
  snapshot, by construction.
- **Wording, not just data.** `SEARCH_PROJECT_DOCUMENTS_TOOL`'s
  description now points at the catalogue ("Search the documents listed
  under 'Documents you can search'...") instead of naming three example
  genres. `REFUSE_TOOL`'s description and `PLANNER_CONTRACT` rule 4 both
  gain the same sentence: a document's file format is never a reason to
  refuse it instead of searching it, if it is on the list; a document the
  user names that is *not* on the list is refused by saying so, never by
  claiming it does not exist.

**Rejected: teach the model about the corpus through few-shot examples in
the tool description instead of a typed block.** The corpus changes per
project and per actor's entitlements; a description baked at authoring
time would be stale the day a document is added, removed, or its scope
changes, and would have to be hand-synchronized with the manifest a second
time. The catalogue is built from the same manifest row that already
governs access, so there is exactly one place that can drift out of sync
with itself -- none.

**Rejected: show the catalogue in every prompt (answer, declare,
memory).** Considered for consistency with `principal`, which does reach
every prompt. Rejected because only `decide` chooses whether to call
`search_project_documents` at all; widening the block's reach would cost
tokens in every other call for a fact none of them act on, and would
blur `system_content`'s own rule -- a block belongs in a prompt only when
that prompt has a decision to make from it.

## Consequences

Live check, same request as the trace that motivated ADR 0025, against the
real corpus and a real model call (`scripts/run_turn.py --actor priya`,
2026-09-15): the fourth model call now chooses
`retrieve_project_documents` instead of `refuse`. The turn no longer
claims the risk register or sprint report are inaccessible.

This does not close the trace end to end. `retrieve_and_answer` still
runs one query per turn, and one query cannot serve three separately-named
documents -- the same live check found the sprint report never retrieved
at all, because the query the model chose ("risk severity in risk
register") crowded it out of the top four. That is
`multi-document-turn-plan.md` step 3's problem, `rag:`'s to close, not this
one's: a catalogue tells the model a document exists and may be searched;
it does not change how many searches one turn gets to run.

`eval/routing_cases.py`'s existing cases are unaffected in shape -- none of
them name a document by file format -- but `PLANNER_CONTRACT`'s text
changed (rule 1 and rule 4), so the ADR 0020 routing comparison needs a
re-run before its numbers can be trusted again; that re-run belongs with
`multi-document-turn-plan.md` step 4, alongside the new A12/A13 cases and
step 3's own contract change.

## Tests

`tests/context/test_catalogue.py`: `build_catalogue` includes an
authorized document, excludes one needing a scope the actor lacks and one
from another project, an actor with the finance scope sees the finance
document too, a scopeless actor gets an empty catalogue rather than an
error, entries carry the fields the prompt needs, and manifest order is
preserved.

`tests/llm/test_prompts.py`: `render_catalogue` renders every entry and
the empty case; a catalogue is appended after the principal in the
planner's system role only; a catalogue with no principal is never shown;
the block appears in no other role; every builder but the planner's
rejects a `catalogue` keyword outright.

`tests/llm/test_gateway.py`: a gateway without a catalogue sends no
catalogue block (unchanged default); a gateway with one sends it from
`decide()` only, never from `declare()` or `answer()`.

`tests/composition/test_turn.py`: the answering gateway carries a
catalogue built from the actor's own scopes (finance-scoped budget summary
left off for an actor without that scope); the memory gateway carries
none.
