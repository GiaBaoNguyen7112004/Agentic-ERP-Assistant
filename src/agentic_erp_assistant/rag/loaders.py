"""Reading a file into blocks that still know where they came from.

This is the module that decides what a citation is allowed to point at. Every
later stage inherits its answer: a chunk cannot cite a section the loader never
recorded, and no amount of care downstream can rebuild an address that was
discarded while the file was being read.

The address is the format's own, never a chunk index
----------------------------------------------------

A retrieval system that has only ever seen plain strings ends up citing
``chunk-14``, which resolves to nothing a reader can open. So each loader emits
the address its format actually has:

* Markdown and HTML -- a section, ``§3.2``. Taken from the numbering the document
  itself uses when it has one, and derived from the heading hierarchy when it
  does not. Preferring the written number matters: a reader resolving ``§3.2``
  scrolls to the heading that says ``3.2``, and a number we invented that
  disagreed with the one on the page would be worse than no number at all.
* PDF -- a page, ``p.4``. The only format here where a page is a real, stable
  address, which is why the corpus contains one.
* CSV -- a row, keyed by the row's own identifier where it has one: ``row R-1``.
  A positional ``row 7`` moves the day somebody sorts the file.

Text before the first heading gets :data:`PREAMBLE`, because "the part before
section 1" is a real place in a document and pretending it belongs to section 1
would put a citation on the wrong side of a heading.

Blocks, not paragraphs and not documents
----------------------------------------

A block is the largest run of text that shares one address: a paragraph, a whole
table, one CSV row. Chunking packs blocks and never splits them unless a single
block exceeds the budget on its own, so a table keeps its header and a row keeps
its columns. Emitting whole documents instead would leave the chunker to
rediscover the structure from punctuation; emitting individual lines would break
tables into rows that mean nothing apart.

Headings are structure, not blocks
----------------------------------

A heading is recorded as the path the following blocks sit under, not emitted as
a block of its own. A heading alone is a poor retrieval hit -- three words with
no content -- and the text is not lost: ``chunking`` prefixes each chunk with its
heading path, so the section title travels with the passage into both search
paths and into the prompt.
"""

import csv
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

__all__ = [
    "DocumentBlock",
    "PREAMBLE",
    "SUPPORTED_SUFFIXES",
    "UnsupportedFormat",
    "load_blocks",
]

logger = logging.getLogger(__name__)


PREAMBLE = "preamble"
"""The locator for text that appears before the first heading."""

_FORBIDDEN_IN_LOCATOR = re.compile(r"[\[\]#\r\n]+")
"""What :class:`~agentic_erp_assistant.state.evidence.EvidenceSnippet` refuses.

Those characters build a citation tag, so a document that could put them into a
locator would control part of a rendered reference and could manufacture one.
The check lives there; this expression is how a loader stays on the right side
of it without every caller having to know the rule.
"""

_HEADING_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(?=\S)")
"""A heading that numbers itself, e.g. ``3.2 Second integration vendor``."""

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(?:```|~~~)")

_PDF_PAGE_FOOTER = re.compile(r"^\s*Page\s+\d+\s*$")
"""A rendered page number, dropped rather than indexed.

It is furniture: it repeats on every page, so it is a term that matches
everything and distinguishes nothing, and as the first line of an extracted page
it would otherwise be glued to the front of that page's first block.
"""

_PDF_SHORT_LINE_RATIO = 0.85
"""How short a line has to be, relative to the widest line on its page, to be
read as the end of a paragraph.

Extracted PDF text has no paragraph markers -- justified body text arrives as a
run of similar-length lines -- so the break has to be inferred, and the only
signal the format leaves is that the last line of a paragraph stops early. The
cost of getting it wrong is bounded: a missed break makes a larger block, which
the chunker splits on tokens anyway.
"""


@dataclass(frozen=True)
class DocumentBlock:
    """One run of text, with the address it can be cited by.

    A frozen dataclass rather than a validated model: blocks never cross a
    process boundary, are never deserialized from anywhere, and are consumed by
    exactly one module. What has to be validated is the
    :class:`~agentic_erp_assistant.rag.chunking.Chunk` built from them, because
    that one is stored, reloaded, and cited.
    """

    locator: str
    """Where in the document this is: ``§3.2``, ``p.4``, ``row R-1``."""

    text: str
    """The block itself, with its internal line structure preserved."""

    heading_path: tuple[str, ...] = ()
    """The headings above it, outermost first. Empty in the preamble, and in
    formats with no headings to speak of."""


class UnsupportedFormat(ValueError):
    """No loader is registered for this file extension.

    Raised rather than skipped. A manifest row naming a file nobody can read is
    a corpus that silently answers fewer questions than its author believes it
    does, and the whole point of the manifest is that it lists what is indexed.
    """


def load_blocks(path: Path) -> tuple[DocumentBlock, ...]:
    """Read ``path`` into ordered blocks, dispatched on its extension.

    Args:
        path: The file to read. Its suffix picks the loader.

    Returns:
        The blocks in document order. Empty only for an empty file.

    Raises:
        UnsupportedFormat: The suffix has no loader.
        OSError: The file cannot be read.
    """
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFormat(
            f"{path.name}: no loader for {path.suffix!r}; supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    blocks = loader(path)
    logger.debug("%s -> %d block(s)", path.name, len(blocks))
    return blocks


# -- section numbering, shared by the two heading-bearing formats -------------


def _clean_locator(value: str) -> str:
    """Strip the characters a citation tag is built from."""
    return _FORBIDDEN_IN_LOCATOR.sub("-", value).strip() or PREAMBLE


@dataclass
class _Heading:
    level: int
    text: str


@dataclass
class _Sections:
    """Turns a stream of headings into section locators.

    Two rules, and the first one is the one worth defending. If a heading
    numbers itself, that number is the locator -- the document is the authority
    on its own numbering, and a positional number that disagreed with the one
    printed on the page would send a reader to the wrong place. Only when a
    document numbers nothing does this count headings itself.
    """

    base_level: int
    """The heading level at which sections start. Deeper levels nest under it."""

    counters: list[int] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    used: set[str] = field(default_factory=set)
    current: str = PREAMBLE

    def enter(self, heading: _Heading) -> None:
        """Move into a heading and recompute the current locator."""
        depth = max(heading.level - self.base_level, 0)

        del self.path[depth:]
        self.path.append(heading.text)

        match = _HEADING_NUMBER.match(heading.text)
        if match:
            number = match.group(1)
            # Keep the positional counters in step with the written numbering,
            # so a document that numbers some headings and not others does not
            # restart from 1 at the first unnumbered one.
            self.counters = [int(part) for part in number.split(".")]
        else:
            # The caller has already advanced the counter for this depth; this
            # only reads it, so numbering stays in one place.
            number = ".".join(str(part) for part in self.counters[: depth + 1])

        self.current = self._unique(f"§{number}")

    def bump(self, depth: int) -> None:
        """Advance the counter at ``depth`` for an unnumbered heading."""
        while len(self.counters) <= depth:
            self.counters.append(0)
        del self.counters[depth + 1 :]
        self.counters[depth] += 1

    def _unique(self, locator: str) -> str:
        """Keep two sections from claiming one address.

        Possible in a document that numbers some headings and not others.
        Without this, two chunks could share a citation tag and a reader
        following it would land on whichever section came first.
        """
        locator = _clean_locator(locator)
        if locator not in self.used:
            self.used.add(locator)
            return locator
        for suffix in range(2, 100):
            candidate = f"{locator}-{suffix}"
            if candidate not in self.used:
                self.used.add(candidate)
                logger.debug("duplicate section locator %s; using %s", locator, candidate)
                return candidate
        return locator


def _sections_for(headings: list[_Heading]) -> _Sections:
    """Decide where sections start, given every heading in the document.

    A single top-level heading that sits above all the others is the document
    title, not section 1 -- so numbering starts one level below it. A document
    with several level-1 headings is using them as sections, and they are
    numbered as such.
    """
    if not headings:
        return _Sections(base_level=1)

    top = min(heading.level for heading in headings)
    tops = [heading for heading in headings if heading.level == top]
    is_title = len(tops) == 1 and headings[0].level == top and len(headings) > 1
    return _Sections(base_level=top + 1 if is_title else top)


def _blocks_from_events(
    events: list[_Heading | str],
) -> tuple[DocumentBlock, ...]:
    """Turn an ordered heading/text stream into addressed blocks.

    One function for Markdown and HTML both, so the two formats cannot drift
    into numbering sections differently -- which would show up as a citation
    that resolves in one document and not in another.
    """
    headings = [event for event in events if isinstance(event, _Heading)]
    sections = _sections_for(headings)
    title_level = sections.base_level - 1

    blocks: list[DocumentBlock] = []
    for event in events:
        if isinstance(event, _Heading):
            if event.level <= title_level:
                # The document title. It names the whole file, not a section,
                # and the manifest already carries it.
                continue
            if not _HEADING_NUMBER.match(event.text):
                sections.bump(max(event.level - sections.base_level, 0))
            sections.enter(event)
            continue

        text = event.strip()
        if text:
            blocks.append(
                DocumentBlock(
                    locator=sections.current,
                    text=text,
                    heading_path=tuple(sections.path),
                )
            )
    return tuple(blocks)


# -- Markdown ----------------------------------------------------------------


def _load_markdown(path: Path) -> tuple[DocumentBlock, ...]:
    """Read Markdown: ATX headings for structure, blank lines for blocks.

    Fenced code is passed through untouched, so a ``#`` comment inside a fence
    cannot be mistaken for a heading and silently renumber every section after
    it.
    """
    events: list[_Heading | str] = []
    buffer: list[str] = []
    fenced = False

    def flush() -> None:
        if buffer:
            events.append("\n".join(buffer))
            buffer.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            buffer.append(line)
            continue

        if fenced:
            buffer.append(line)
            continue

        heading = _ATX_HEADING.match(line)
        if heading:
            flush()
            events.append(_Heading(len(heading.group(1)), heading.group(2).strip()))
            continue

        if line.strip():
            buffer.append(line.rstrip())
        else:
            flush()

    flush()
    return _blocks_from_events(events)


# -- HTML --------------------------------------------------------------------


_HTML_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_HTML_BLOCKS = {"p", "li", "blockquote", "pre", "dd", "dt", "figcaption"}
_HTML_IGNORED = {"script", "style", "head", "title", "nav", "footer"}


class _HtmlReader(HTMLParser):
    """Collects headings and block-level text in document order.

    Deliberately small. It is reading fixture documents and wiki exports, not
    arbitrary web pages, and a full DOM library would be a dependency bought to
    solve a problem this corpus does not have. Tables are gathered whole rather
    than cell by cell, because a cell without its row and header says nothing.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[_Heading | str] = []
        self._buffer: list[str] = []
        self._tag: str | None = None
        self._ignoring = 0
        self._table: list[str] | None = None
        self._row: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _HTML_IGNORED:
            self._ignoring += 1
            return
        if self._ignoring:
            return

        if tag == "table":
            self._flush_text()
            self._table = []
        elif tag == "tr":
            self._row = []
        elif tag in _HTML_HEADINGS or tag in _HTML_BLOCKS or tag in {"td", "th"}:
            self._flush_text()
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in _HTML_IGNORED:
            self._ignoring = max(self._ignoring - 1, 0)
            return
        if self._ignoring:
            return

        if tag in {"td", "th"} and self._row is not None:
            self._row.append(" ".join("".join(self._buffer).split()))
            self._buffer.clear()
            self._tag = None
        elif tag == "tr" and self._table is not None and self._row is not None:
            if any(cell for cell in self._row):
                self._table.append(" | ".join(self._row))
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.events.append("\n".join(self._table))
            self._table = None
        elif tag == self._tag:
            self._flush_text(tag)

    def handle_data(self, data: str) -> None:
        if not self._ignoring and self._tag is not None:
            self._buffer.append(data)

    def _flush_text(self, tag: str | None = None) -> None:
        text = " ".join("".join(self._buffer).split())
        self._buffer.clear()
        closing, self._tag = tag or self._tag, None
        if not text:
            return
        if closing in _HTML_HEADINGS:
            self.events.append(_Heading(int(closing[1]), text))
        else:
            self.events.append(text)

    def finish(self) -> list[_Heading | str]:
        self._flush_text()
        return self.events


def _load_html(path: Path) -> tuple[DocumentBlock, ...]:
    """Read HTML with the stdlib parser -- headings for structure, blocks for text."""
    reader = _HtmlReader()
    reader.feed(path.read_text(encoding="utf-8"))
    reader.close()
    return _blocks_from_events(reader.finish())


# -- CSV ---------------------------------------------------------------------


def _load_csv(path: Path) -> tuple[DocumentBlock, ...]:
    """Read a CSV as one block per row, addressed by the row's own identifier.

    Each block is rendered as ``column: value`` lines rather than as the raw
    comma-separated line. A retrieved passage has to be readable on its own once
    it is sitting in a prompt fifty lines away from the header row, and
    ``severity: high`` carries that context where ``high`` does not.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        key = _identifier_column(fieldnames)

        blocks: list[DocumentBlock] = []
        for position, row in enumerate(reader, start=1):
            lines = [
                f"{name}: {(row.get(name) or '').strip()}"
                for name in fieldnames
                if (row.get(name) or "").strip()
            ]
            if not lines:
                continue

            identifier = (row.get(key) or "").strip() if key else ""
            locator = _clean_locator(f"row {identifier or position}")
            blocks.append(
                DocumentBlock(
                    locator=locator,
                    text="\n".join(lines),
                    heading_path=(f"row {identifier}",) if identifier else (),
                )
            )
    return tuple(blocks)


def _identifier_column(fieldnames: list[str]) -> str | None:
    """The column whose value addresses a row, if the file has one.

    A stable key beats a position: ``row R-1`` still resolves after somebody
    sorts the register by severity, and ``row 7`` does not.
    """
    for name in fieldnames:
        lowered = name.strip().lower()
        if lowered == "id" or lowered.endswith("_id"):
            return name
    return None


# -- PDF ---------------------------------------------------------------------


def _load_pdf(path: Path) -> tuple[DocumentBlock, ...]:
    """Read a PDF page by page, one block per inferred paragraph.

    The page is the address, which is the reason a PDF is in this corpus at all.
    Paragraphs inside a page are inferred, because extracted PDF text has no
    paragraph markers -- see :data:`_PDF_SHORT_LINE_RATIO`.
    """
    from pypdf import PdfReader  # imported here: only this loader pays for it

    blocks: list[DocumentBlock] = []
    reader = PdfReader(str(path))
    for number, page in enumerate(reader.pages, start=1):
        lines = [
            line.rstrip()
            for line in (page.extract_text() or "").splitlines()
            if line.strip() and not _PDF_PAGE_FOOTER.match(line)
        ]
        if not lines:
            continue

        locator = _clean_locator(f"p.{number}")
        widest = max(len(line) for line in lines)
        heading_path: tuple[str, ...] = ()
        paragraph: list[str] = []

        def flush() -> None:
            if paragraph:
                blocks.append(
                    DocumentBlock(
                        locator=locator,
                        text="\n".join(paragraph),
                        heading_path=heading_path,
                    )
                )
                paragraph.clear()

        for line in lines:
            if _HEADING_NUMBER.match(line) and len(line) < widest * _PDF_SHORT_LINE_RATIO:
                # A numbered heading on its own line: structure, not a block,
                # the same way it is in Markdown. Chunking puts the path back in
                # front of the passage, so nothing is lost by not indexing three
                # words on their own.
                flush()
                heading_path = (line.strip(),)
                continue

            paragraph.append(line.strip())
            if len(line) < widest * _PDF_SHORT_LINE_RATIO:
                flush()

        flush()
    return tuple(blocks)


_LOADERS = {
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".html": _load_html,
    ".htm": _load_html,
    ".csv": _load_csv,
    ".pdf": _load_pdf,
}
"""Extension to loader. A table, so adding a format is a row here and a function
below it -- and so ``load_blocks`` contains no branch on file type."""

SUPPORTED_SUFFIXES = frozenset(_LOADERS)
"""What the manifest may point at. Read by tests and by the manifest loader."""
