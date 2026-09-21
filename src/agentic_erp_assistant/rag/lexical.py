"""The sparse half of search: real BM25, in the standard library.

Embeddings are good at "what is this about" and bad at "the exact string R-1".
A question naming a risk id, a milestone code, a date or a figure needs a term
match, and a dense index will happily return five passages about risk in general
instead. So the lexical half is a real BM25 -- the Robertson/Sparck-Jones
formula, with term frequency saturation, inverse document frequency and
document-length normalization -- and not a word-overlap ratio dressed up as a
score.

The distinction matters. Overlap counts how many query words appear; BM25 asks
how *surprising* it is that they appear here. A chunk containing "the" twenty
times and "R-17" once should rank above one containing "the" forty times, and
only the second formula can express that. It is also what OpenSearch,
Elasticsearch and most production hybrid stacks use for this half, so a reviewer
who knows BM25 recognizes it rather than having to audit an invention.

Statistics are computed over what the reader may see
----------------------------------------------------

Access control is applied before any scoring, and it is applied before the
corpus statistics are computed too. That second part is the less obvious half:
if inverse document frequency were computed over the whole corpus, a restricted
document would still influence ranking -- a term that is rare among the
documents a reader may see, but common in ones they may not, would be scored as
though it were ordinary. It would not leak the text, but "a restricted chunk
must never influence ranking" is a cleaner rule to hold than "must not influence
it very much".

The cost is that statistics depend on the asking context, so they are cached per
context. At this corpus size recomputing them is microseconds; at a scale where
it is not, this is exactly the point at which the lexical half moves into the
same engine as the dense one and inherits its filtering.

Built from the vector store, not from the corpus
------------------------------------------------

The chunks come from :meth:`~agentic_erp_assistant.rag.ports.VectorIndexPort.iter_chunks`,
so both indexes are populated from one list and a chunk cannot be in one and not
the other.
"""

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_erp_assistant.rag.access import RetrievalContext, is_authorized
from agentic_erp_assistant.rag.chunking import Chunk
from agentic_erp_assistant.rag.ports import ScoredChunk

__all__ = [
    "B",
    "BM25Index",
    "COMMON_TERM_RATIO",
    "K1",
    "MIN_CORPUS_FOR_PRUNING",
    "tokenize",
]

logger = logging.getLogger(__name__)


K1 = 1.2
"""Term-frequency saturation. The standard value.

It sets how quickly a repeated term stops adding to the score: with k1 at 1.2 a
term appearing ten times scores well under twice what it scores appearing once,
which is the behaviour that stops a long chunk winning by repetition.
"""

B = 0.75
"""Length normalization. Also the standard value.

At 1.0 a chunk is penalized fully for being longer than average, at 0.0 not at
all. Chunks here vary from a CSV row to a packed section, so some normalization
is needed; the conventional 0.75 is kept rather than tuned, because tuning it
against eight documents would be fitting a constant to a sample.
"""

COMMON_TERM_RATIO = 0.5
"""A query term appearing in more than this share of the visible corpus is
dropped from the query.

Statistical rather than a stopword list, so it adapts: in a corpus about
migrations, "migration" may well be the term carrying no information, and no
English stopword list would have known. Measured on this corpus it removes
exactly what it should -- "the" (0.95 of chunks), "and" (0.88), "is" (0.72),
"to" and "a" (0.63), "of" (0.55) -- and keeps every content term.

What this is **not** is a sufficiency test. It is tempting to think that pruning
common terms makes "no lexical hits" mean "no evidence", and the measurement
says otherwise: in this corpus "what" appears in 4 chunks of 64 and "who" in 1,
while "cutover" appears in 16 and "risk" in 14. Interrogatives are *rarer* than
content words in formal documents, so no document-frequency threshold can
separate them. A BM25 hit therefore means a term appeared, not that an answer is
present -- which is why sufficiency is decided on the dense side; see
:data:`~agentic_erp_assistant.rag.retriever.MIN_COSINE_SIMILARITY`.
"""

MIN_CORPUS_FOR_PRUNING = 8
"""Below this many visible chunks, no term is pruned.

In a five-chunk corpus a perfectly ordinary term appears in three of them, so
the ratio above is noise rather than a signal. The rule is switched off entirely
rather than scaled, because a threshold that moves with corpus size is a second
thing to explain and this one only has to be right at the size real indexes
have.
"""

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,\-/][a-z0-9]+)*")
"""What counts as one term.

Deliberately wider than ``\\w+``. This corpus is full of identifiers that carry
their punctuation -- ``r-1``, ``m2``, ``sc-2026-08-01``, ``2026-09-11``,
``515,000``, ``3.2`` -- and a tokenizer that split them would turn the most
specific term in a question into several of the least specific ones, which is
precisely backwards for the half of search that exists to match exact strings.
"""


def tokenize(text: str) -> list[str]:
    """Lowercase and split into terms.

    No stemming and no stopword list. Stemming would fold ``migrating`` onto
    ``migrate`` and also ``R-1`` onto nothing useful, and inverse document
    frequency already does what a stopword list does -- a term appearing in
    every chunk earns a near-zero weight by arithmetic rather than by a list
    someone has to maintain and defend.
    """
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class _Statistics:
    """Everything BM25 needs about one visible corpus, computed once."""

    chunks: tuple[Chunk, ...]
    frequencies: tuple[Counter[str], ...]
    lengths: tuple[int, ...]
    document_frequency: dict[str, int]
    average_length: float

    @classmethod
    def over(cls, chunks: Sequence[Chunk]) -> "_Statistics":
        frequencies = tuple(Counter(tokenize(chunk.text)) for chunk in chunks)
        lengths = tuple(sum(counter.values()) for counter in frequencies)

        document_frequency: Counter[str] = Counter()
        for counter in frequencies:
            document_frequency.update(counter.keys())

        return cls(
            chunks=tuple(chunks),
            frequencies=frequencies,
            lengths=lengths,
            document_frequency=dict(document_frequency),
            # Zero for an empty corpus, which the scorer never reaches because
            # it returns early on one.
            average_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
        )


class BM25Index:
    """A lexical index over chunks, scored with BM25.

    A plain class rather than a dataclass: it takes any sequence and stores a
    tuple, and it owns a cache that is never an argument, so the generated
    constructor would have had to be replaced anyway.

    Usage::

        index = BM25Index(vector_store.iter_chunks())
        hits = index.search("milestone M2 status", context=context, limit=5)
    """

    def __init__(
        self, chunks: Sequence[Chunk], *, k1: float = K1, b: float = B
    ) -> None:
        self.chunks: tuple[Chunk, ...] = tuple(chunks)
        """Every chunk the index knows about, before access control."""
        self._cache: dict[tuple[str, tuple[str, ...]], _Statistics] = {}
        self.k1 = k1
        self.b = b
        logger.info("lexical index built over %d chunk(s)", len(self.chunks))

    def __len__(self) -> int:
        return len(self.chunks)

    def _statistics(self, context: RetrievalContext) -> _Statistics:
        """Corpus statistics over the chunks this context may read.

        Cached per context, because the same actor asks more than one question
        and recomputing per query would make a multi-step turn pay for the
        corpus once per step.
        """
        key = (context.project_code, tuple(sorted(context.scopes)))
        statistics = self._cache.get(key)
        if statistics is None:
            visible = [
                chunk for chunk in self.chunks if is_authorized(chunk, context)
            ]
            statistics = _Statistics.over(visible)
            self._cache[key] = statistics
            logger.debug(
                "lexical statistics for %s: %d of %d chunk(s) visible",
                key,
                len(visible),
                len(self.chunks),
            )
        return statistics

    def _discriminative(
        self, terms: list[str], statistics: _Statistics
    ) -> list[str]:
        """Drop query terms that appear in most of what this reader can see.

        See :data:`COMMON_TERM_RATIO`. The corpus statistics used are the
        context's own, so what counts as common depends on what the reader may
        read -- the same rule the scoring uses, applied to the same numbers.
        """
        total = len(statistics.chunks)
        if total < MIN_CORPUS_FOR_PRUNING:
            return terms

        kept = [
            term
            for term in terms
            if statistics.document_frequency.get(term, 0) / total <= COMMON_TERM_RATIO
        ]
        if len(kept) != len(terms):
            logger.debug(
                "dropped %d common term(s) from the query",
                len(terms) - len(kept),
            )
        return kept

    def _idf(self, term: str, statistics: _Statistics) -> float:
        """Inverse document frequency, in the form that cannot go negative.

        ``ln(1 + (N - df + 0.5) / (df + 0.5))``. The classic Robertson form
        without the leading 1 turns negative for a term appearing in more than
        half the corpus, which would let a common word *subtract* from a score
        and let a chunk rank higher for not containing a query term. Lucene,
        OpenSearch and Elasticsearch all use this variant for that reason.
        """
        total = len(statistics.chunks)
        frequency = statistics.document_frequency.get(term, 0)
        return math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))

    def search(
        self, query: str, *, context: RetrievalContext, limit: int
    ) -> tuple[ScoredChunk, ...]:
        """The best-matching chunks this context may read, best first.

        Args:
            query: The words to match. Tokenized the same way the chunks were.
            context: Applied before scoring and before statistics; see the
                module docstring.
            limit: How many to return. Must be positive.

        Returns:
            Up to ``limit`` scored chunks, descending, ties broken by chunk id
            so the same query twice gives the same order. A query whose terms
            appear nowhere returns nothing rather than an arbitrary low-scoring
            result -- "no lexical evidence" is a real answer, and dressing it up
            as a weak hit is how a refusal turns into a guess.

        Raises:
            ValueError: ``limit`` is not positive.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        statistics = self._statistics(context)
        terms = self._discriminative(tokenize(query), statistics)
        if not terms or not statistics.chunks:
            return ()

        weights = {term: self._idf(term, statistics) for term in set(terms)}

        scored: list[ScoredChunk] = []
        for chunk, frequencies, length in zip(
            statistics.chunks,
            statistics.frequencies,
            statistics.lengths,
            strict=True,
        ):
            normalizer = self.k1 * (
                1.0
                - self.b
                + self.b * (length / statistics.average_length)
            )
            score = sum(
                weights[term]
                * (frequencies[term] * (self.k1 + 1.0))
                / (frequencies[term] + normalizer)
                for term in weights
                if frequencies[term]
            )
            if score > 0.0:
                scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return tuple(scored[:limit])
