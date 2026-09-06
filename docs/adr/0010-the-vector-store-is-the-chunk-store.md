# 0010 — Qdrant holds the chunks, and the lexical index is built from it

**Status:** Accepted (2026-09-06)

## Context

Hybrid retrieval needs the same chunks in two places: as vectors in Qdrant, and
as tokenized documents in an in-process BM25 index. Nothing forces those two to
be built the same way, and the acceptance criterion the reference states —
"a chunk can never exist in one index and not the other" — is a statement about
how they are populated, not about either of them individually.

The obvious arrangement is for ingestion to write to both, and for startup to
rebuild the lexical index by re-reading the corpus. That is two write paths and
two readers of the manifest, and they agree only for as long as nobody changes
one of them. The realistic failure is not dramatic: a document is re-ingested
after an edit, the vector store is updated, the lexical index is rebuilt from a
corpus directory that has since moved on, and a passage becomes findable by a
semantic question and invisible to a keyword one. No error is logged, because
from each component's point of view nothing went wrong.

## Decision

The Qdrant point payload is the chunk itself — every field of `Chunk`, not just
metadata beside a vector. `QdrantVectorIndex.iter_chunks()` reads them back and
validates them through the same model, and `RetrievalService.load()` builds the
BM25 index from that.

There is one write path (`ingest`) and one read path (`iter_chunks`). The lexical
index describes what is stored, not what the last ingest intended to store.

The stored payload also carries each chunk's document content hash, which is what
makes an incremental re-ingest possible: the index is asked what it already
holds, and only documents whose hash has moved are embedded again.

## Consequences

The two indexes cannot disagree about which chunks exist, by construction rather
than by discipline. A test asserts the sets are identical, but the property does
not depend on the test.

Recovery is trivial and obvious: the lexical index is derived state, so losing it
costs a restart, and there is no separate lexical persistence to back up,
migrate, or leave stale.

Startup does a full scroll of the collection. At this corpus size that is a few
hundred payloads and a few milliseconds. It grows linearly, and the point where
it stops being free is the same point at which BM25 belongs in an engine that can
filter and score server-side — the design does not pretend this scales
indefinitely, and ADR 0008 names the same boundary.

Storing chunk text twice — in the payload, and nowhere else — means Qdrant is
holding the corpus, so a Qdrant backup is a corpus backup. That is a real
operational property worth stating: the volume in `docker-compose.yml` is not a
cache.

Because the payload is the store of record, `Chunk` is a validated pydantic model
rather than a plain dataclass. What comes back out of the database has been
outside this process, and a payload written by an older build must fail
validation rather than be trusted.

## Alternatives considered

**Write both indexes from the corpus at ingest, and rebuild the lexical one from
the corpus at startup.** Rejected; this is the divergence described above. It
also makes "what is actually searchable?" a question with two answers.

**Persist the lexical index to its own file.** Rejected. It adds a second
artifact to keep in step with the first, for a rebuild that costs milliseconds.

**Keep only ids and metadata in the payload, and store chunk text separately.**
Rejected. It saves storage this project does not need to save, and it makes every
search a two-step fetch — the second step being the one that fails partially and
returns a hit whose text cannot be quoted.

**Use Qdrant's own sparse-vector support for BM25 instead of an in-process
index.** Deferred, and it is the natural end state: it would put both halves
behind one filter and one engine. It is not done here because the assessed core
must contain a real BM25 implementation, and because a sparse-vector encoder
introduces a second vocabulary artifact to keep in step with the corpus.
