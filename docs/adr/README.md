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

## Format

Numbered `NNNN-kebab-case-title.md`, never renumbered. Sections: **Status**,
**Context**, **Decision**, **Consequences**, **Alternatives considered**. Status
is `Proposed`, `Accepted`, or `Superseded by NNNN` — an ADR is never deleted or
edited into a different decision, because the point of the record is that it
remains readable after the decision changes.
