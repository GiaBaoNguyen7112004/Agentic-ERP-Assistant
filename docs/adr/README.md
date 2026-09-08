# Architecture decision records

One file per decision that a reviewer could reasonably ask "why?" about and that
the code alone does not answer. The diff shows *what* was built; these show what
else was on the table and why it lost.

A decision belongs here when reversing it would cost real work — a port
boundary, a policy ordering, a place where a default was deliberately refused.
Routine choices (a variable name, a helper's shape) belong in the commit message.

## Index

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-one-tokenizer-authority.md) | One tokenizer authority for both the budget check and the context plan | Accepted |
| [0002](0002-context-builder-has-no-default-model.md) | `ContextBuilder` takes a model with no default | Accepted |
| [0003](0003-compaction-is-an-allow-list.md) | Compaction is an allow-list, and the summary is best-effort | Accepted |
| [0004](0004-rate-limits-are-checked-before-approval-and-counted-after.md) | Rate limits are registry policy, checked before approval and counted after | Accepted |
| [0005](0005-the-graph-is-a-cycle-bounded-by-a-step-budget.md) | The graph is a cycle, bounded by a step budget rather than by missing edges | Accepted |
| [0006](0006-function-calling-is-the-only-decision-channel.md) | Function calling is the only decision channel, and retrieval is offered through it | Accepted |
| [0007](0007-retrieval-context-is-bound-to-the-retriever.md) | The retrieval context is bound to the retriever, not passed to `search` | Accepted |
| [0008](0008-one-access-rule-for-both-search-paths.md) | One access rule, enforced before ranking on both search paths | Accepted |
| [0009](0009-the-locator-is-structural-and-the-chunk-id-is-the-tag.md) | The locator is the format's own address, and the chunk id is the citation tag | Accepted |
| [0010](0010-the-vector-store-is-the-chunk-store.md) | Qdrant holds the chunks, and the lexical index is built from it | Accepted |
| [0011](0011-memory-is-written-after-the-turn-by-a-pure-policy.md) | Memory is written after the turn, and only a pure policy may write it | Accepted |
| [0012](0012-memory-is-context-and-never-a-citation.md) | Memory is context in a role of its own, and can never be a citation | Accepted |
| [0013](0013-postgres-is-the-memory-record-and-graph-memory-is-rejected.md) | Postgres is the memory record, Qdrant indexes it, and graph memory is rejected | Accepted |

## Format

Numbered `NNNN-kebab-case-title.md`, never renumbered. Sections: **Status**,
**Context**, **Decision**, **Consequences**, **Alternatives considered**. Status
is `Proposed`, `Accepted`, or `Superseded by NNNN` — an ADR is never deleted or
edited into a different decision, because the point of the record is that it
remains readable after the decision changes.
