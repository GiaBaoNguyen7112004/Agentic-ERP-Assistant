# 0017 — Tools are bound to the actor's project, at the store and at the gateway

**Status:** Accepted (2026-09-11)

## Context

`rag/access.py` binds every document chunk to a project through
`RetrievalContext`, and ADR 0008 records why that check runs once, before
ranking, on both search paths. The tool layer had no equivalent. `MockErp`
held one project's data; the gateway checked `required_scope` and nothing
else; neither `AgentState` nor `ToolRequest` carried a project code at all.
The omission was invisible for exactly the reason it was dangerous: a store
with one project has no second project to leak, so nothing in the existing
suite could fail by leaking one.

This is fact 10 in the end-to-end code plan, and the second of the two gaps
the plan commits to closing before the web layer lands. A browser chat with
an actor switcher is the moment the omission stops being theoretical: two
actors on two different projects, sharing one process, one mock ERP, and one
registry, is exactly the shape the web layer introduces.

## Decision

The rule is enforced twice, mirroring the document access rule's own
two-sided design (ADR 0008: one rule, checked before a handler or a ranker
ever sees the wrong project's data) — at the store, and at the gateway.

**At the store**, `MockErp.for_project(project_code)` returns a `ProjectErp`
view: the only thing a handler reads through. `milestone`, `sprint`,
`budget`, and `project` return `None` for a record of another project;
`risks_for` returns `()`. A milestone that belongs to another project reads
exactly the way a milestone that does not exist at all reads — existence
itself is hidden, the same "does not exist for you" a filtered document
gets. `create_risk` through the wrong view raises `ErpAccessError`: defence
in depth, since the gateway (below) refuses a mismatched write before a
handler is ever reached, but the view must still refuse on its own if that
first line is ever bypassed or a handler is called directly. Handlers now
take `(arguments, ExecutionContext)` — `ExecutionContext.project_code`
supplies what `for_project` needs, carried separately from the arguments a
model produced, because a project bound by policy and an argument the model
chose are not the same kind of fact and must not be confused for one.

**At the gateway**, `ToolDefinition.project_argument` names the argument
(when a tool's arguments name a project at all — `get_budget_summary`,
`list_risks`, `create_risk` do; `get_project_status` and
`get_sprint_progress` do not, because a milestone or sprint id is unambiguous
across the fixture without one) that must match the request's
`project_code`. A mismatch is refused `denied`, before validation's result
ever reaches a handler, with both projects named in the error. `AgentState`
and `ToolRequest` both gain a required `project_code`, snapshotted alongside
`actor` and `scopes` for the same reason `actor` is: every authorization
decision downstream is "this project and this entitlement", and a turn or a
call that could exist without a project is one whose writes cannot be bound
to one.

Adding a required field to `AgentState` bumps `STATE_VERSION` to 2. A v1
state refuses to load rather than being silently read as a turn on no
project — there is no honest default to invent for a field this load-bearing,
the same reasoning the version guard was built for in the first place.

The ERP fixture gains a second project, Orion, mirroring the RAG corpus's
`orion-status-report-2026-09.md` (already in the document corpus for
cross-project retrieval scenarios, with no ERP-side counterpart until now):
a project row, milestone `O2` ("Pipeline migration", on track), and its
budget. Orion deliberately has no sprint and no risk row — the status report
narrates one open risk nobody has recorded in the tracker, so a project-bound
actor recording it through `create_risk` is a real write to prove, not a
duplicate of a fixture row.

## Consequences

`STATE_VERSION = 2` means every stored `runs.state` and `pauses.state` from
before this change refuses to load. In development that is a `TRUNCATE` of
the evidence tables (`docs/manual-test.md` §1.3); a deployed system would need
an actual migration, which this project does not have and does not need yet
— there is no data written under v1 outside development.

`runs.project_code` is now its own column (a new `ALTER TABLE ... ADD COLUMN
IF NOT EXISTS`, following the pattern `pauses.decided_by` already set), so a
future screen listing runs by project does not have to parse the jsonb state
to filter on it.

Roughly forty `AgentState(`/`ToolRequest(` construction sites across the test
suite and two demo scripts needed a `project_code`. Mechanical, and paid once
here rather than deferred, because every phase after this one builds states
that assume the field exists.

Every tool in the default registry is now covered by the drift test in
`tests/tools/test_gateway.py`: a call bound to one project, naming or
implying another, either is `denied` at the gateway (a `project_argument`
tool) or reads as nonexistent through the view (one that has none). No path
returns another project's data — proved for all six tools in one test rather
than trusted to six separate reviews.

## Alternatives considered

**A per-handler check inside each handler body**, each one calling
`erp.project(...)` and comparing to the actor's project before doing
anything else. Rejected: five copies of one rule, and the copy that drifts
when a sixth tool is added is the one nobody notices until it leaks.

**A per-request registry, built fresh over a project-filtered store for
every call.** Rejected: `build_default_registry` binds the flaky-read
counter and every handler closure once, at composition time; rebuilding it
per request would reset that state on every call for no benefit the view
does not already provide, and would multiply the cost of what is otherwise a
cheap, per-turn assembly (D6 in the code plan).

**Trust the model's own `project_id` argument as the authority.** Rejected
outright. The model is not the authority on what an actor is entitled to
touch — it is data the model chose, from a prompt, and the entire point of
the gateway's ordered checks is that nothing upstream of them is trusted
until verified. `ProjectErp` and `project_argument` both exist because the
argument has to be checked against something the caller does not control.
