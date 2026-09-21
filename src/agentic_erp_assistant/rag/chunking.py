"""Packing blocks into token-budgeted chunks that keep one address each.

A chunk is the unit that gets embedded, scored, retrieved and cited, so two
properties decide how good retrieval can possibly be: how much of one idea a
chunk contains, and whether the address it carries lands a reader on that idea.

One chunk, one locator
----------------------

A chunk never spans two sections or two pages. It could -- packing across a
heading boundary would fill the budget more evenly -- and then a citation would
have to name a range, and a reader following ``§3.1-3.3`` is back to searching
for the sentence. The whole reason the loader kept a structural address is that
the citation lands somewhere exact, and merging across addresses spends that.

The cost is real and accepted: a short section becomes a short chunk. On this
corpus that is usually the right shape anyway, because the sections were written
as units of meaning, and the eval harness is where the trade-off gets measured
rather than argued about.

Every chunk carries its own title and section
---------------------------------------------

The body is prefixed with the document title and the heading path. This is not
decoration. A passage that reads "the forecast is 515,000" is nearly
indistinguishable, to a lexical index and to an embedding model alike, from the
same sentence in a different project's report -- and this corpus deliberately
contains two status reports for two projects. The header is what separates them,
and it costs tokens that are counted against the same budget as the body.

Blocks are packed, never split, unless one block alone is too big
-----------------------------------------------------------------

Greedy packing of whole blocks means a table keeps its header and a CSV row
keeps its columns. When a single block exceeds the budget on its own -- a long
PDF paragraph, a wide table -- it falls back to a real token-level split, so no
chunk ever exceeds its budget regardless of how the source was written. The
boundary block is carried forward as overlap when it fits the overlap budget, so
a fact that spans a paragraph break survives in at least one chunk.

No token count is stored on a chunk
-----------------------------------

``context/candidate.py`` refuses to hold one and gives the reason: a number
stored beside the text it describes drifts the first time somebody edits one and
not the other, silently. The same argument applies with more force here, because
a chunk is stored in a vector database and read back later. The budget is
enforced while packing and re-measured by the tests.
"""

import logging
from collections.abc import Iterator, Sequence

from pydantic import BaseModel, ConfigDict, Field

from agentic_erp_assistant.llm.tokenizer import count_tokens, split_into_token_windows
from agentic_erp_assistant.rag.loaders import DocumentBlock
from agentic_erp_assistant.rag.manifest import SourceDocument
from agentic_erp_assistant.state.evidence import EvidenceSnippet

__all__ = [
    "CHUNK_TOKENS",
    "Chunk",
    "OVERLAP_TOKENS",
    "chunk_document",
    "chunk_documents",
]

logger = logging.getLogger(__name__)


CHUNK_TOKENS = 320
"""How large one chunk may get, in tokens of the embedding model.

Chosen against what a chunk has to do rather than against a round number. It has
to hold a whole section of one of these documents most of the time -- the
sections here run 60 to 250 tokens -- and four of them have to fit in a prompt
alongside the policy blocks without crowding out the reply
(:data:`~agentic_erp_assistant.engine.nodes.EVIDENCE_LIMIT` is 4, so this is
roughly 1,300 tokens of evidence). Much larger and a single hit starts carrying
two topics, which is how a retriever scores well and cites imprecisely.
"""

OVERLAP_TOKENS = 64
"""How much of the previous chunk a split section repeats.

Only ever spent when one section is too large for a single chunk, which on this
corpus is rare. It exists so that a fact stated across a paragraph break is
whole in at least one chunk; twenty percent of the budget is the usual band and
is cheap at this corpus size.
"""

_SEPARATOR = "\n\n"
"""What joins two blocks inside one chunk."""


class Chunk(BaseModel):
    """One passage, with its address and every policy fact needed to serve it.

    Validated rather than a plain dataclass, unlike
    :class:`~agentic_erp_assistant.rag.loaders.DocumentBlock`, because a chunk
    does cross a boundary: it is written into a vector store as a payload and
    read back out of one, and what comes back has been outside this process.
    ``project_code`` and ``required_scope`` are on it, so the thing that decides
    access is reading the chunk in front of it rather than a lookup that could
    be skipped.

    Frozen, because a chunk that could be edited after it was scored would let
    the text that was ranked differ from the text that gets quoted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    """``{document_id}#{locator}`` -- the citation tag without its brackets.

    Deliberately the same string a reader sees in the answer and in the
    retrieval trace. An opaque uuid here would mean the identifier a reviewer
    follows and the identifier the index stores are two things that have to be
    joined before anyone can check anything.
    """

    document_id: str = Field(min_length=1)
    """The document. Becomes ``EvidenceSnippet.source_id`` and the citation."""

    locator: str = Field(min_length=1)
    """Where inside it: ``§3.2``, ``p.4``, ``row R-1``, or one of those with
    ``part N`` appended when a section needed more than one chunk."""

    text: str = Field(min_length=1)
    """What gets embedded, scored, and quoted -- header line included, because
    the header is part of what makes this passage findable."""

    heading_path: tuple[str, ...] = ()
    """The headings this passage sits under, outermost first."""

    title: str = Field(min_length=1)
    """The document title, from the manifest."""

    document_type: str = Field(min_length=1)
    """The document class, from the manifest. Reported on; never used to decide
    access."""

    project_code: str = Field(min_length=1)
    """Half of the authorization check. See :mod:`agentic_erp_assistant.rag.access`."""

    required_scope: str = Field(min_length=1)
    """The other half. The entitlement a reader must hold."""

    classification: str = Field(min_length=1)
    """The sensitivity label a person sees. Never the thing that grants access."""

    content_hash: str = Field(min_length=1)
    """The hash of the document this chunk came from, so a re-ingest can tell
    which stored chunks are stale without re-reading the corpus."""

    position: int = Field(ge=0)
    """Where this chunk falls in its document. Breaks ties deterministically, so
    two chunks that score identically always come back in the same order."""

    @property
    def tag(self) -> str:
        """The exact token an answer cites, e.g. ``[status-report-2026-09#§3.2]``."""
        return f"[{self.chunk_id}]"

    def as_snippet(self) -> EvidenceSnippet:
        """Convert to the type the graph and the prompt layer carry.

        The one place the conversion happens, so a locator that would forge a
        citation tag is rejected here -- by ``EvidenceSnippet`` itself -- rather
        than at the point evidence is rendered into a prompt.
        """
        return EvidenceSnippet(
            source_id=self.document_id, locator=self.locator, text=self.text
        )


def _header(title: str, heading_path: Sequence[str]) -> str:
    """The context line every chunk carries. See the module docstring."""
    if not heading_path:
        return title
    return f"{title}\n{' > '.join(heading_path)}"


def _groups(
    blocks: Sequence[DocumentBlock],
) -> Iterator[tuple[str, tuple[str, ...], list[DocumentBlock]]]:
    """Consecutive blocks that share an address and a heading path.

    Grouped on both, not just the locator, because a PDF page can contain two
    headings: the page is one address, and the heading path has to stay true to
    the blocks it is printed above.
    """
    current: list[DocumentBlock] = []
    key: tuple[str, tuple[str, ...]] | None = None

    for block in blocks:
        block_key = (block.locator, block.heading_path)
        if key is not None and block_key != key:
            yield key[0], key[1], current
            current = []
        key = block_key
        current.append(block)

    if key is not None and current:
        yield key[0], key[1], current


def _pack(
    blocks: Sequence[DocumentBlock],
    *,
    model: str,
    budget: int,
    overlap_tokens: int,
) -> list[str]:
    """Greedily pack block texts into bodies of at most ``budget`` tokens."""
    bodies: list[str] = []
    current: list[str] = []
    carried = 0

    def flush() -> None:
        nonlocal current, carried
        if current and len(current) > carried:
            bodies.append(_SEPARATOR.join(current))
        current, carried = [], 0

    for block in blocks:
        text = block.text
        if count_tokens(text, model=model) > budget:
            # Too big to pack whole. Everything held so far is flushed first so
            # the oversized block does not get glued onto an unrelated tail.
            flush()
            windows = split_into_token_windows(
                text,
                model=model,
                window_tokens=budget,
                overlap_tokens=min(overlap_tokens, budget - 1),
            )
            bodies.extend(windows)
            continue

        if current and count_tokens(
            _SEPARATOR.join([*current, text]), model=model
        ) > budget:
            tail = current[-1]
            flush()
            # Carry the boundary block forward when it is small enough to count
            # as overlap and still leaves room for the block that displaced it.
            # Both conditions matter: a carry that does not fit would push the
            # next chunk over the budget it was just flushed to respect.
            if count_tokens(tail, model=model) <= overlap_tokens and count_tokens(
                _SEPARATOR.join([tail, text]), model=model
            ) <= budget:
                current, carried = [tail], 1

        current.append(text)

    flush()
    return bodies


def _bodies_for(
    blocks: Sequence[DocumentBlock],
    *,
    header: str,
    model: str,
    chunk_tokens: int,
    overlap_tokens: int,
    document_id: str,
    locator: str,
) -> list[str]:
    """Pack a section, then check the assembled chunk rather than trusting arithmetic.

    Subtracting the header cost from the budget is nearly right and not exactly
    right: byte-pair encoding merges across the join, so the tokens in
    ``header + separator + body`` need not equal the tokens counted in the parts.
    Modelling that would be guesswork. Measuring the thing that actually gets
    sent, and tightening the body budget by however much it overshot, is exact --
    and it converges, because the budget strictly decreases each round.
    """
    budget = chunk_tokens - count_tokens(header + _SEPARATOR, model=model)

    while budget > 0:
        bodies = _pack(
            blocks, model=model, budget=budget, overlap_tokens=overlap_tokens
        )
        overflow = max(
            (
                count_tokens(f"{header}{_SEPARATOR}{body}", model=model) - chunk_tokens
                for body in bodies
            ),
            default=0,
        )
        if overflow <= 0:
            return bodies
        logger.debug(
            "%s %s: assembled chunk ran %d token(s) over; tightening the body budget",
            document_id,
            locator,
            overflow,
        )
        budget -= overflow

    raise ValueError(
        f"{document_id}: the header for {locator} needs "
        f"{count_tokens(header + _SEPARATOR, model=model)} tokens, which leaves "
        f"no room inside a {chunk_tokens}-token chunk"
    )


def chunk_document(
    document: SourceDocument,
    *,
    model: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> tuple[Chunk, ...]:
    """Turn one document into chunks, each with one address and one budget.

    There is no injectable token counter here, unlike
    :class:`~agentic_erp_assistant.context.builder.ContextBuilder`. Counting and
    splitting have to be done on the same encoding -- a caller who could swap
    the counter but not the splitter would get chunks that exceed a budget the
    splitter believed it was honouring -- so both come from
    :mod:`agentic_erp_assistant.llm.tokenizer`, which ADR 0001 already makes the
    single authority.

    Args:
        document: A loaded document, blocks and all.
        model: Whose tokenizer decides the budget. Required, with no default,
            for the reason ``ContextBuilder`` gives: a token count means nothing
            without the model that produced it. Pass the embedding model, since
            that is what these chunks will be sent to.
        chunk_tokens: The ceiling on one chunk, header included.
        overlap_tokens: How much a split section repeats.

    Returns:
        The chunks in document order. Never empty for a loaded document.

    Raises:
        ValueError: ``overlap_tokens`` is not smaller than ``chunk_tokens``, or
            a section's header alone does not leave room for any body -- both
            configuration mistakes that would otherwise surface as chunks that
            quietly break their budget.
    """
    if overlap_tokens >= chunk_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than chunk_tokens "
            f"({chunk_tokens})"
        )

    entry = document.entry
    drafts: list[tuple[str, tuple[str, ...], str]] = []

    for locator, heading_path, blocks in _groups(document.blocks):
        header = _header(entry.title, heading_path)
        bodies = _bodies_for(
            blocks,
            header=header,
            model=model,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            document_id=entry.document_id,
            locator=locator,
        )
        for body in bodies:
            drafts.append((locator, heading_path, f"{header}{_SEPARATOR}{body}"))

    parts: dict[str, int] = {}
    for locator, _, _ in drafts:
        parts[locator] = parts.get(locator, 0) + 1

    seen: dict[str, int] = {}
    chunks: list[Chunk] = []
    for position, (locator, heading_path, text) in enumerate(drafts):
        if parts[locator] > 1:
            seen[locator] = seen.get(locator, 0) + 1
            address = f"{locator} part {seen[locator]}"
        else:
            address = locator

        chunks.append(
            Chunk(
                chunk_id=f"{entry.document_id}#{address}",
                document_id=entry.document_id,
                locator=address,
                text=text,
                heading_path=heading_path,
                title=entry.title,
                document_type=entry.document_type,
                project_code=entry.project_code,
                required_scope=entry.required_scope,
                classification=entry.classification,
                content_hash=document.content_hash,
                position=position,
            )
        )

    logger.debug("%s -> %d chunk(s)", entry.document_id, len(chunks))
    return tuple(chunks)


def chunk_documents(
    documents: Sequence[SourceDocument],
    *,
    model: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> tuple[Chunk, ...]:
    """Chunk a whole corpus, in manifest order.

    One list, because both indexes are built from it in a single pass. A chunk
    that existed in the vector store and not in the lexical one -- or the
    reverse -- would be a passage that is findable by one kind of question and
    invisible to the other, and nothing in a trace would show why.
    """
    chunks = tuple(
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            model=model,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
    )
    logger.info("chunked %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks
