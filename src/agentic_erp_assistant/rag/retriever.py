"""The retriever the graph actually holds, and the service that builds one.

This is where the two halves meet and where retrieval stops being a search
problem and becomes an evidence problem. Three decisions live here.

The access context is bound, never passed
-----------------------------------------

:class:`~agentic_erp_assistant.engine.ports.DocumentRetrieverPort` declares
``search(query, *, limit)`` and carries no context, and widening it would be the
wrong fix. A retriever that *can* return restricted passages and trusts each
call site to pass the right context is one forgotten argument away from a leak,
and the forgotten one is always the interesting one.

So :class:`RetrievalService` holds the expensive shared things -- the Qdrant
client, the lexical index, the embeddings client -- and ``for_context`` returns a
cheap :class:`HybridRetriever` bound to one actor for one turn, which cannot see
anything outside it. It mirrors
:attr:`~agentic_erp_assistant.state.agent_state.AgentState.scopes`, snapshotted
when the turn begins for the same reason: entitlements that could change midway
would let one turn read under two different permissions and the trace would show
neither.

One embedding call per search
-----------------------------

The candidates were embedded once, at ingestion. A query costs exactly one
embedding call, and a test proves it by counting requests. This is the difference
between a retrieval step that costs a fraction of a cent and one that quietly
re-embeds the corpus.

Sufficiency is decided before anything is spent, and decided on the dense side
--------------------------------------------------------------------------------

An unanswerable question must produce *no* evidence rather than weak evidence,
because :meth:`~agentic_erp_assistant.engine.nodes.GraphNodes.retrieve_and_answer`
refuses on an empty result and calls the model on a non-empty one. Refusing here
costs nothing; refusing after a synthesis call costs a synthesis call.

A dense index always returns its nearest neighbours, however far away they are,
so "it returned something" is not evidence -- hence
:data:`MIN_COSINE_SIMILARITY`. And the gate is dense-only on purpose: BM25
cannot tell a match on a content word from a match on a question word, because
in a corpus of formal documents the question words are the rarer ones (measured:
"what" in 4 chunks of 64, "cutover" in 16). See
:data:`~agentic_erp_assistant.rag.lexical.COMMON_TERM_RATIO` for the numbers.

Once the gate passes, the lexical list joins the fusion in full, so a chunk the
dense half ranked poorly but that carries the exact identifier asked about can
still be lifted into the answer. Dense decides *whether*; both decide *what*.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from agentic_erp_assistant.rag.access import RetrievalContext
from agentic_erp_assistant.rag.fusion import FusedHit, RRF_K, reciprocal_rank_fusion
from agentic_erp_assistant.rag.lexical import BM25Index
from agentic_erp_assistant.rag.ports import EmbeddingsPort, VectorIndexPort
from agentic_erp_assistant.state.evidence import EvidenceSnippet

__all__ = [
    "CANDIDATE_LIMIT",
    "HybridRetriever",
    "LEXICAL",
    "MIN_COSINE_SIMILARITY",
    "RetrievalOutcome",
    "RetrievalService",
    "VECTOR",
]

logger = logging.getLogger(__name__)


VECTOR = "vector"
LEXICAL = "lexical"
"""The names the two halves appear under in a fused hit, and therefore in a
trace. Constants because a report reads them and a rename would otherwise be a
silent change to an evidence artifact."""


MIN_COSINE_SIMILARITY = 0.30
"""How close a chunk has to be for the question to count as answerable at all.

**Provisional until the evidence run.** A dense index returns its nearest
neighbours whatever the question, so without a floor "the retriever found
something" is true for every input, including questions this corpus cannot
answer -- and the refusal would then happen after a billed synthesis call
instead of before it. The number has to come from measurement rather than from
taste, so ``scripts/run_retrieval_evaluation.py`` reports the best similarity
per golden case, including for the case designed to be unanswerable, and this
constant should be set from the gap between the two groups. 0.30 is the starting
point for ``text-embedding-3-small``; it is a constructor argument, so a
deployment can move it without editing this file.
"""

CANDIDATE_LIMIT = 20
"""How many candidates each half produces before fusion.

Wider than the four the runtime asks for, because fusion can only reorder what
it is given: a chunk ranked eighth by the dense half and first by the lexical one
is exactly the hit hybrid search exists to surface, and it never gets the chance
if each half only reports its top four.
"""


@dataclass(frozen=True)
class RetrievalOutcome:
    """One search, with the numbers that explain why it came out that way.

    ``best_similarity`` is the highest cosine the dense half saw **before** the
    floor was applied, which is the one number the hits themselves cannot carry:
    a search that returned nothing returns no scores either, and "nothing was
    found" and "the closest thing was 0.29 and the floor is 0.30" call for
    completely different responses from whoever is reading the trace. It is also
    what the evaluator needs in order to set that floor from measurement instead
    of from taste.
    """

    hits: tuple[FusedHit, ...]
    best_similarity: float | None
    """``None`` only when the dense half returned nothing at all."""

    dense_candidates: int
    """How many chunks cleared the floor."""

    lexical_candidates: int
    """How many the lexical half returned. Zero when the gate refused first."""

    @property
    def gated(self) -> bool:
        """Whether the floor is what made this search empty."""
        return not self.hits and self.dense_candidates == 0


@dataclass(frozen=True)
class HybridRetriever:
    """Retrieval for one actor, on one project, for the length of one turn.

    Satisfies :class:`~agentic_erp_assistant.engine.ports.DocumentRetrieverPort`
    structurally -- it never imports ``engine`` -- so the graph depends on a
    shape and this class stays a retrieval concern.

    Frozen and cheap to build. It holds references to the shared indexes rather
    than copies, so making one per turn costs an object, which is the price of
    the context being impossible to get wrong.
    """

    vector_index: VectorIndexPort
    lexical_index: BM25Index
    embeddings: EmbeddingsPort
    context: RetrievalContext
    """The bound access context. Every search this object performs is inside it."""

    candidate_limit: int = CANDIDATE_LIMIT
    minimum_similarity: float = MIN_COSINE_SIMILARITY
    rrf_k: int = RRF_K

    def search(self, query: str, *, limit: int) -> tuple[EvidenceSnippet, ...]:
        """The port method: passages for ``query``, best first, ready to cite.

        Projects :meth:`search_ranked` down to the type the graph carries. The
        scores do not come with them, deliberately -- see
        :class:`~agentic_erp_assistant.rag.ports.ScoredChunk` and the
        ``EvidenceSnippet`` docstring: a number no branch reads, riding inside
        the object that goes into a prompt, is telemetry pretending to be
        evidence.

        Returning nothing is a normal answer and means the question could not be
        answered from these documents. The graph routes that to a refusal.
        """
        return tuple(
            hit.chunk.as_snippet() for hit in self.search_ranked(query, limit=limit)
        )

    def search_ranked(self, query: str, *, limit: int) -> tuple[FusedHit, ...]:
        """The same search, with the ranks and scores that produced it.

        For the trace, and for anything that wants to explain a result. Same
        computation as :meth:`search`, one layer earlier; see
        :meth:`search_detailed` when the diagnostics matter too.
        """
        return self.search_detailed(query, limit=limit).hits

    def search_detailed(self, query: str, *, limit: int) -> RetrievalOutcome:
        """The search, plus what the halves saw on the way.

        Args:
            query: What to look for.
            limit: How many passages to return.

        Returns:
            A :class:`RetrievalOutcome` carrying up to ``limit`` fused hits,
            best first, and the numbers behind them. No hits means the question
            had no dense support in the documents this context may read.

        Raises:
            ValueError: ``limit`` is not positive, or ``query`` is blank.
            ProviderAuthError: The embeddings provider refused definitively.
            TransientProviderError: The embeddings call failed transiently. Not
                retried here -- the runtime owns the retry budget and the trace.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if not query.strip():
            raise ValueError("query must not be blank")

        # Exactly one embedding call per search. The candidates were embedded at
        # ingestion; a retriever that embedded per candidate would re-buy the
        # corpus on every question.
        batch = self.embeddings.embed([query])
        vector_hits = self.vector_index.search(
            batch.vectors[0], context=self.context, limit=self.candidate_limit
        )

        best = max((hit.score for hit in vector_hits), default=None)
        near = tuple(
            hit for hit in vector_hits if hit.score >= self.minimum_similarity
        )
        if not near:
            logger.info(
                "closest chunk to %r for %s scored %s, under the %.2f floor; "
                "refusing before synthesis",
                query,
                self.context.actor,
                f"{best:.3f}" if best is not None else "nothing",
                self.minimum_similarity,
            )
            return RetrievalOutcome(
                hits=(),
                best_similarity=best,
                dense_candidates=0,
                lexical_candidates=0,
            )

        lexical_hits = self.lexical_index.search(
            query, context=self.context, limit=self.candidate_limit
        )

        hits = reciprocal_rank_fusion(
            {VECTOR: near, LEXICAL: lexical_hits}, k=self.rrf_k, limit=limit
        )
        logger.info(
            "%r -> %d hit(s) from %d dense and %d lexical candidate(s)",
            query,
            len(hits),
            len(near),
            len(lexical_hits),
        )
        return RetrievalOutcome(
            hits=hits,
            best_similarity=best,
            dense_candidates=len(near),
            lexical_candidates=len(lexical_hits),
        )


@dataclass
class RetrievalService:
    """The shared, expensive half of retrieval, built once per process.

    Holds the vector store, the lexical index built from it, and the embeddings
    client. Hands out a :class:`HybridRetriever` per turn.

    Usage::

        service = RetrievalService.from_env()
        retriever = service.for_context(
            RetrievalContext.for_actor(
                "priya", project_code="atlas", scopes=state.scopes
            )
        )
        runtime = WorkflowRuntime(retriever=retriever, ...)
    """

    vector_index: VectorIndexPort
    embeddings: EmbeddingsPort
    lexical_index: BM25Index
    candidate_limit: int = CANDIDATE_LIMIT
    minimum_similarity: float = MIN_COSINE_SIMILARITY
    rrf_k: int = RRF_K

    @classmethod
    def load(
        cls,
        *,
        vector_index: VectorIndexPort,
        embeddings: EmbeddingsPort,
        candidate_limit: int = CANDIDATE_LIMIT,
        minimum_similarity: float = MIN_COSINE_SIMILARITY,
        rrf_k: int = RRF_K,
    ) -> Self:
        """Build the lexical index by reading the vector store back.

        Not by re-reading the corpus. One write path means a chunk cannot exist
        in one index and not the other, and it means the lexical index describes
        what is actually stored rather than what the last ingest intended to
        store.
        """
        chunks = vector_index.iter_chunks()
        logger.info("loaded %d chunk(s) from the vector store", len(chunks))
        return cls(
            vector_index=vector_index,
            embeddings=embeddings,
            lexical_index=BM25Index(chunks),
            candidate_limit=candidate_limit,
            minimum_similarity=minimum_similarity,
            rrf_k=rrf_k,
        )

    @classmethod
    def from_env(cls, **options: object) -> Self:
        """Build everything from the environment, for a script or a server.

        Imported locally so that importing this module does not construct a
        vector-database client or read a ``.env`` -- the tests build the service
        from fakes and must not pay for either.
        """
        from agentic_erp_assistant.rag.embeddings import OpenAIEmbeddingsClient
        from agentic_erp_assistant.rag.vector_index import QdrantVectorIndex

        return cls.load(
            vector_index=QdrantVectorIndex.from_env(),
            embeddings=OpenAIEmbeddingsClient(),
            **options,  # type: ignore[arg-type]
        )

    def for_context(self, context: RetrievalContext) -> HybridRetriever:
        """A retriever bound to one actor, for one turn.

        The only way to get a retriever. There is deliberately no method here
        that searches without a context: an unbound search is the call that
        would eventually be made from somewhere that forgot to pass one.
        """
        return HybridRetriever(
            vector_index=self.vector_index,
            lexical_index=self.lexical_index,
            embeddings=self.embeddings,
            context=context,
            candidate_limit=self.candidate_limit,
            minimum_similarity=self.minimum_similarity,
            rrf_k=self.rrf_k,
        )

    def close(self) -> None:
        """Release whatever the service opened, if anything did."""
        for component in (self.embeddings, self.vector_index):
            closer = getattr(component, "close", None)
            if callable(closer):
                closer()
