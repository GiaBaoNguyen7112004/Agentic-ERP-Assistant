# 0013 — Postgres is the memory record, Qdrant indexes it, and graph memory is rejected

**Status:** Accepted (2026-09-08)

## Context

Three topologies were on the table for memory, and the brief asks for them to be
compared rather than for one to be asserted: relational state, a vector database
for semantic recall, and a graph store for linked project entities.

The repo already has a decision that pulls one way. ADR 0010 says the vector
store *is* the chunk store: Qdrant holds the chunk payload, the lexical index is
built from it, and there is one write path. Following that precedent for memory
would mean Qdrant holds the memory records too, and it would be the consistent
choice.

It is also wrong here, and the reason is a difference between the two kinds of
data rather than a preference between two databases.

A chunk has no identity that outlives an ingest, no supersession, and no
uniqueness constraint. Re-ingesting a document replaces its chunks wholesale; a
copy of one in Qdrant cannot be stale, because there is nothing it could be stale
*relative to*.

A memory has all three. It is retired the moment a better one replaces it —
`superseded_at`, never a delete, so the audit keeps both halves. Exactly one
intent may be open per session, and two would mean recall choosing between tasks
and leaking a slot from the wrong one. And every record has an audit row that has
to agree with it: `write`, `update`, `reject` and `forget` are a decision
procedure, not a change log.

Those are relational obligations. A store that cannot express them enforces them
in application code that two processes can race past.

The graph question is separate and is the one the brief presses hardest on. The
case for graph memory is real when the required queries are traversals — "which
risks touch the milestones this sprint feeds". The question is whether *these*
queries are.

## Decision

**Postgres is the record.** Three tables: `memories`, `intents`, `memory_audit`.
Two invariants that must hold across processes are database constraints, not only
Python checks — `intents_one_open_per_session` as a partial unique index, and
`memory_audit_rejection_matches_decision` as a CHECK. Both are enforced in the
models as well; the Python check protects one process and the index protects the
database, which is the move `pauses_one_pending_per_run` already makes.

**Qdrant is an index over it, in a collection of its own.** `QdrantMemoryIndex`
stores the memory id and the filter's inputs — project, scope, kind, session,
actor, retired — and never the statement. There is exactly one copy of a
memory's text, in the store that owns it, so the index cannot serve a stale
version. `search` returns `(memory_id, score)` and the caller hydrates through
`MemoryStorePort.by_id`, which re-checks liveness and scope.

Freshness is a filter clause so a retired memory does not occupy a slot in the
top *k*, but it is an optimization and never the guarantee: hydration is what
makes a stale recall impossible.

The collection is separate from `project_documents`. Documents and memories are
ingested on different schedules and rebuilt from different sources, and a corpus
re-ingest that dropped the collection would take every remembered preference with
it — a failure nobody would predict from the command they typed.

**The access rule is ADR 0008's, extended to a third path.** `MemoryRecord`
carries `project_code` and `required_scope`, so it satisfies `rag.access.
Restricted` structurally and `is_authorized` applies unchanged.
`memory_filter` takes `qdrant_filter(scope.access)` and *appends* to it rather
than restating its two conditions.

**Graph memory is rejected, and the door is left open at zero cost.** The queries
this assistant has to answer are lookups: "what is this person's preference for
X", "what did we decide about Y", "what is the task in flight". Every one is a
`(kind, key)` or a scope lookup, and every one is indexed. The traversal queries
a graph would serve — a risk to its sprint to its milestone — are already
answered by the ERP's own relational data behind `list_risks` and
`get_sprint_progress`, which is authoritative and current in a way memory never
is. Adding a graph store would mean a third database, a third backup, a third
consistency story and an ingestion path keeping it in step, to serve queries
that either have a home already or nobody has asked.

`MemoryRecord.links` records related identifiers as a flat tuple of strings. If
the required queries ever become traversals, the relationships are already
recorded and a graph store becomes an adapter rather than a migration.

## Consequences

Three stores now hold state: Qdrant for both search paths, Postgres for evidence
and memory. A memory write touches two of them, and the order is stated — store
first, index second — so a failure leaves the record stored and unsearchable
rather than searchable and absent. `MemoryVectorStorePort.retire` must not raise
for that reason, and recall runs degraded instead of wrong.

The index is derived state and can be rebuilt from `memories` by re-embedding.
That is what makes a changed `OPENAI_EMBEDDING_MODEL` recoverable here in a way
it is not for the corpus: `ensure_ready` refuses the width change, and the fix is
a rebuild that loses nothing.

Recall costs one embedding call per turn and consolidation costs one more for the
batch it stored. Both are stated in `memory/service.py` because both are money.

Two SQL queries per recall in the common case (the pinned kinds, then the
hydration), plus one Qdrant query. `PostgresMemoryStore.live` issues one query
per kind rather than one with an OR of scope clauses, because the bounds differ
by kind — a preference is bounded by actor, an intent by session — and a single
WHERE would encode that difference a second time. At this size that is a handful
of indexed lookups; the alternative is the SQL and `memory.models.bounds`
drifting apart, which shows up as a preference recalled for the wrong person.

Rejecting graph memory means a question like "which decisions relate to the
risks on this milestone" has no direct answer from memory. It is answerable by
`links` plus a second lookup, and it has not been asked.

## Alternatives considered

**Qdrant as the memory store, following ADR 0010.** Rejected. Supersession,
one-open-intent and the audit join are the reasons, and none of them applies to a
chunk. Consistency with the earlier decision would have been consistency about
the wrong property.

**Postgres alone, with no semantic recall.** Seriously considered, and it would
work today: a session holds dozens of memories, and lexical overlap plus the
pinned kinds answers most turns. Rejected because the lexical door alone misses
the case memory is most useful for — a request phrased nothing like the fact that
bears on it — and because the port would have existed unused, which is a seam
nobody has tested.

**One Qdrant collection for chunks and memories, with a type field.** Rejected.
It couples two lifecycles that have nothing in common and makes a corpus rebuild
a memory outage.

**Store the statement in the Qdrant payload so recall needs no hydration.**
Rejected. It is one round trip saved and two copies of a mutable record created,
and the copy that drifts is always the one being read.

**A graph store for linked project entities.** Rejected, above. Worth restating
the criterion rather than the conclusion: graph memory earns its operational cost
when relationships *materially improve required queries*. Here the required
queries are lookups, and the traversals belong to the ERP, which owns those
relationships and keeps them current.

**Skip the `links` field, since nothing traverses it.** Rejected. It costs a
`text[]` column and it is what makes the graph decision reversible without a
migration.
