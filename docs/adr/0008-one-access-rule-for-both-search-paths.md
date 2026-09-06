# 0008 — One access rule, enforced before ranking on both search paths

**Status:** Accepted (2026-09-06)

## Context

Hybrid retrieval has two independent search paths. Dense search runs inside
Qdrant, which can filter on point payload. Lexical search is BM25 running in this
process over a list of chunks. Both must apply the same access policy.

The failure this invites is well known and specific: one path gets a fix the
other one misses. A project check is added to the lexical scorer and not to the
vector query, or the payload filter is tightened and the in-process comparison is
left as it was. Nothing fails loudly. The system keeps answering, and one of the
two paths quietly returns documents the reader is not entitled to.

There is also a question of *when* the check runs. Filtering after ranking is
easier — one code path, applied to whatever came back — and it is wrong in a way
that is invisible: a restricted chunk that was ranked has already taken a slot in
the top `k`, so the reader with fewer permissions silently receives fewer
results than the reader with more, for the same `limit`.

A subtler version of the same question applies to BM25 itself. Its scores depend
on corpus statistics — inverse document frequency, average document length — so a
restricted document can influence the ranking of visible ones even if it never
appears in a result.

## Decision

`rag/access.py` holds the entire policy:

```python
def is_authorized(chunk: Restricted, context: RetrievalContext) -> bool:
    return (
        chunk.project_code == context.project_code
        and chunk.required_scope in context.scopes
    )
```

Both halves must hold; there is no public tier and no wildcard scope. An actor
holding no scopes is refused everything.

`qdrant_filter(context)`, which renders the same rule as a server-side payload
filter, lives in the same module, immediately below it. The vector adapter
re-checks returned points with `is_authorized` as well, which is unreachable
while the two agree and costs one comparison per hit.

Enforcement happens before ranking on both paths. The Qdrant filter is part of
the query. BM25 filters the corpus before scoring — and computes its corpus
statistics over the filtered corpus, cached per context, so a restricted
document cannot influence a ranking it never appears in even through idf.

Each document declares exactly one `required_scope`, not a set.

## Consequences

A reviewer can audit the whole access-control policy by reading one function, and
the parametrised drift test in `tests/rag/test_access.py` proves the two
expressions agree: it loads the real corpus into a real in-memory Qdrant and
asserts that the filter accepts exactly the chunks the comparison accepts, across
seven contexts including an actor with no scopes and a project that does not
exist. It runs against the real filter evaluator, not a reimplementation of what
the filter is assumed to do — that would only prove the assumption agrees with
itself.

One scope per document is what makes the rule expressible as a payload filter at
all. A set would need subset logic, which Qdrant cannot express as a condition,
so the vector path would have had to fetch-then-filter while the lexical path
compared sets — reintroducing exactly the divergence this ADR exists to prevent.
The cost is that a document readable by two unrelated groups needs a scope that
names that combination.

BM25 statistics being per-context means scores depend on who is asking, which is
correct and slightly surprising: the same query from two actors can rank the same
two visible chunks differently. It is cached per context, so a multi-step turn
pays once. At a corpus size where recomputing is expensive, this is the point at
which the lexical half belongs in the same engine as the dense one, inheriting
its filtering — the design does not pretend otherwise.

## Alternatives considered

**Filter after ranking, on both paths.** Rejected. Simpler, and it silently
shrinks results for the least privileged users — the failure mode nobody reports
because the answer still looks plausible.

**Compute BM25 statistics globally and filter only the results.** Rejected,
though it is the conventional choice and would have been defensible. The rule
"a restricted chunk must never influence ranking" is cleaner to hold and to test
than "must not influence it very much", and at this corpus size the per-context
computation is microseconds.

**Put `qdrant_filter` in `vector_index.py`, beside the code that uses it.**
Rejected. It is the second expression of a security rule, and the argument for
keeping both in one file is precisely that a reader comparing them should not
have to open two. `access.py` imports the Qdrant models lazily so the lexical
path does not pay for a database client it never uses.

**Store `classification` and decide access from it.** Rejected. Classification is
a label for people; entitlement is a check for code. Deriving one from the other
means every new label is a silent policy change.
