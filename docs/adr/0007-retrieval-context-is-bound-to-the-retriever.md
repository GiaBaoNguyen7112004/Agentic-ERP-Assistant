# 0007 — The retrieval context is bound to the retriever, not passed to `search`

**Status:** Accepted (2026-09-06)

## Context

`engine/ports.py` declares the shape the graph depends on:

```python
def search(self, query: str, *, limit: int) -> Sequence[EvidenceSnippet]: ...
```

There is no actor and no project on it. But retrieval has to enforce access
control — a document nobody may read must not reach a prompt — so the
authorization context has to get in somehow.

The reference specification this work follows solves it the obvious way, by
putting the context in the signature:

```python
hits = store.search(query_vector, context=RequestContext(role="member",
                    project_code="atlas"), limit=5)
```

Adopting that shape here would mean widening `DocumentRetrieverPort`, and
therefore teaching `engine/nodes.py` to build a `RetrievalContext` and pass it
on every call.

## Decision

The port is unchanged. `RetrievalService` holds the expensive shared state — the
Qdrant client, the lexical index, the embeddings client — and
`for_context(context)` returns a `HybridRetriever` bound to one actor for the
length of one turn. That object satisfies the port structurally and cannot see
anything outside its context.

There is deliberately no method anywhere in `rag/` that searches without a
context, and no way to construct a `HybridRetriever` without one: `context` is a
required field of a frozen dataclass.

## Consequences

The unsafe call does not exist. A search performed without an authorization
context is not a bug that can be introduced by forgetting an argument at a new
call site, because there is no signature that accepts the omission. That matters
more than it sounds: the call sites that will exist a year from now — a
re-indexing job, a debug endpoint, an eval harness, a second graph — are exactly
the ones nobody reviews as carefully as the first.

`engine/` needed no change at all to gain access-controlled retrieval, which is
the payoff of the port having been declared as a shape rather than transcribed
from an implementation.

Building a retriever per turn costs one small frozen object holding three
references. The indexes, the client and the connection pool are shared.

The cost is that a single object cannot serve two actors, so anything holding a
retriever must hold it for no longer than the turn it belongs to. The evaluator
makes this visible in a useful way: it builds one retriever per golden case,
which is what lets two cases ask the byte-identical question under different
entitlements and be scored as the different cases they are.

This mirrors `AgentState.scopes`, which is snapshotted when a turn begins for the
same reason — entitlements that could change midway would let one turn read
under two different permissions, with the trace showing neither.

## Alternatives considered

**Widen the port to take a context, as specified.** Rejected. It makes every
future call site responsible for a security argument, and a default value for it
would be indefensible in either direction: defaulting to "no scopes" makes the
common path silently return nothing, and defaulting to anything else grants
access from a place no reviewer looks.

**Keep the port narrow and read the context from a thread-local or a context
variable.** Rejected. It removes the argument without removing the coupling, and
it turns "which actor was this search for?" into a question answerable only by
knowing what ran before. An audit trail assembled from ambient state is not an
audit trail.

**Filter after retrieval, in the node.** Rejected on two grounds. The node would
have to know about projects and scopes to do it, which puts access policy in the
runtime; and a restricted chunk that reaches the node has already occupied a slot
in the top `k` and displaced something the reader was entitled to see, so the
result is quietly worse for the users with the fewest permissions. See ADR 0008.
