"""Combining two ranked lists that do not share a scale.

Cosine similarity lives in roughly [0, 1] and clusters tightly; BM25 is
unbounded and depends on corpus statistics and query length. They are not
comparable, which is why the obvious combination is wrong:

    0.65 * dense + 0.35 * lexical

looks principled and is not. The weights imply the two numbers mean the same
thing, and they do not -- the same 0.65/0.35 split behaves completely differently
on a one-word query and a ten-word one, because only one of the two scores grew.

Reciprocal Rank Fusion sidesteps the problem by throwing the scores away and
keeping only what both lists genuinely agree on: order. A chunk contributes
``1 / (k + rank)`` from each list it appears in, and the contributions are
summed. Something ranked first by both halves beats something ranked first by
one, and something found by only one half still appears -- which is the entire
point of running two halves.

It is what OpenSearch and Elasticsearch use for hybrid search, for these
reasons, so a reviewer recognizes the formula rather than auditing an invention.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agentic_erp_assistant.rag.chunking import Chunk
from agentic_erp_assistant.rag.ports import ScoredChunk

__all__ = ["FusedHit", "RRF_K", "reciprocal_rank_fusion"]

logger = logging.getLogger(__name__)


RRF_K = 60
"""The rank offset in ``1 / (k + rank)``.

Sixty is the value from the original paper and the default in the search engines
that ship this. What it controls is how sharply the first few ranks are
preferred: at ``k = 0`` rank 1 is worth twice rank 2 and a single list can
dominate the fusion, while a large ``k`` flattens the curve until being in both
lists matters more than being first in either. Sixty sits far enough along that
agreement between the halves outweighs a narrow win inside one of them, which is
the behaviour hybrid search is for.
"""


@dataclass(frozen=True)
class FusedHit:
    """One chunk, its fused score, and where each list put it.

    The per-list ranks are kept rather than discarded because they are what
    makes a retrieval trace legible: "this came back because both halves ranked
    it second" is an explanation, and a lone fused float is not. The evaluator
    reads them too.
    """

    chunk: Chunk
    score: float
    """The summed reciprocal ranks. Comparable within one query and meaningless
    between queries -- it depends on how many lists a chunk appeared in, not on
    how good a match it was."""

    ranks: Mapping[str, int]
    """Which rank each list gave it, 1-based. A list that did not return it is
    absent, so ``"lexical" not in hit.ranks`` reads as "the keyword half never
    found this"."""

    scores: Mapping[str, float]
    """Each list's own score, kept for the trace and the eval report. Never
    compared across lists -- see the module docstring."""

    @property
    def found_by(self) -> tuple[str, ...]:
        """The lists that returned this chunk, in a stable order."""
        return tuple(sorted(self.ranks))


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ScoredChunk]],
    *,
    k: int = RRF_K,
    limit: int | None = None,
) -> tuple[FusedHit, ...]:
    """Fuse named ranked lists by rank position.

    Args:
        rankings: Named lists, each already in rank order, best first. The names
            travel through to :attr:`FusedHit.ranks`, so they should be the ones
            a reader of a trace would want to see -- ``"vector"``, ``"lexical"``.
        k: The rank offset. See :data:`RRF_K`.
        limit: How many hits to return. ``None`` returns all of them.

    Returns:
        Fused hits, best first, ties broken by chunk id so the same inputs
        always produce the same order. Retrieval is a computation, not a source
        of nondeterminism.

    Raises:
        ValueError: ``k`` is negative, or ``limit`` is not positive.
    """
    if k < 0:
        raise ValueError(f"k must not be negative, got {k}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    chunks: dict[str, Chunk] = {}
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float]] = {}
    fused: dict[str, float] = {}

    for name, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            chunks.setdefault(chunk_id, hit.chunk)
            ranks.setdefault(chunk_id, {})[name] = rank
            scores.setdefault(chunk_id, {})[name] = hit.score
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
    if limit is not None:
        ordered = ordered[:limit]

    logger.debug(
        "fused %s into %d hit(s)",
        {name: len(hits) for name, hits in rankings.items()},
        len(ordered),
    )
    return tuple(
        FusedHit(
            chunk=chunks[chunk_id],
            score=fused[chunk_id],
            ranks=dict(ranks[chunk_id]),
            scores=dict(scores[chunk_id]),
        )
        for chunk_id in ordered
    )
