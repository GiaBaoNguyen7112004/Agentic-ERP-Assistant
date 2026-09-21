"""Render the committed PDF fixtures from their readable Markdown sources.

The corpus in ``data/documents/`` deliberately contains one PDF, because a PDF is
the only format in this project where "page 3" is a real address a reader can
resolve -- Markdown gives sections, CSV gives rows, and a page number has to come
from somewhere. Committing a binary that nobody can diff would be the wrong way
to get one, so the text lives in ``data/documents/sources/`` and this script
renders it.

Running it is not part of ingestion and not part of the test suite. It exists so
that a reviewer who wants to change a number in the budget summary edits a
Markdown file, re-runs this, and gets a PDF whose page breaks fall wherever the
layout puts them -- which is the property that makes a page locator worth citing
in the first place.

``reportlab`` is a development dependency for this reason: it is needed to
*produce* a fixture, never to read one. The runtime reads PDFs with ``pypdf``.

    uv run python scripts/build_pdf_fixtures.py
"""

import logging
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("build_pdf_fixtures")

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "data" / "documents" / "sources"
OUTPUT_DIR = REPO_ROOT / "data" / "documents"

FIXTURES = {"budget-summary-q3.md": "budget-summary-q3.pdf"}
"""Source file -> rendered artifact. One row per committed PDF."""


def _styles() -> dict[str, ParagraphStyle]:
    """Document styles, tuned so the sample corpus runs to several pages.

    The leading and the frame margins are what decide where page breaks fall,
    and therefore what ``p.2`` means. They are set here, once, rather than left
    to library defaults, so that re-rendering after an unrelated reportlab
    upgrade produces the same pagination and does not silently invalidate a
    citation somebody already wrote down.
    """
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "DocH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "DocH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            spaceBefore=9,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "DocBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
    }


def _table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    """Render pipe rows as a bordered table, first row as the header."""
    cell = ParagraphStyle("Cell", parent=styles["body"], alignment=0, spaceAfter=0)
    data = [[Paragraph(value, cell) for value in row] for row in rows]
    table = Table(data, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _inline(text: str) -> str:
    """Convert the two inline markers the sources use into reportlab markup."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)


def _flow(source: str, styles: dict[str, ParagraphStyle]) -> list[object]:
    """Turn the Markdown subset the sources use into reportlab flowables.

    The subset is deliberately small -- headings, paragraphs, pipe tables and an
    explicit page break -- because these sources are fixtures, not a document
    format anyone else has to support.
    """
    flowables: list[object] = []
    pending_rows: list[list[str]] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            flowables.append(Paragraph(_inline(" ".join(paragraph).strip()), styles["body"]))
            paragraph.clear()

    def flush_table() -> None:
        if pending_rows:
            flowables.append(Spacer(1, 3))
            flowables.append(_table(list(pending_rows), styles))
            flowables.append(Spacer(1, 9))
            pending_rows.clear()

    for raw in source.splitlines():
        line = raw.rstrip()

        if line.startswith("|"):
            flush_paragraph()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not all(set(cell) <= {"-", ":"} for cell in cells if cell):
                pending_rows.append(cells)
            continue

        flush_table()

        if not line.strip():
            flush_paragraph()
            continue

        if line == "<!-- page-break -->":
            flush_paragraph()
            flowables.append(PageBreak())
            continue

        for prefix, style in (("### ", "h3"), ("## ", "h2"), ("# ", "title")):
            if line.startswith(prefix):
                flush_paragraph()
                heading = Paragraph(_inline(line[len(prefix) :]), styles[style])
                # A heading stranded at the foot of a page would put its section
                # number on one page and its text on the next, which is exactly
                # the citation nobody can resolve.
                flowables.append(KeepTogether([heading, Spacer(1, 2)]))
                break
        else:
            paragraph.append(line.strip())

    flush_paragraph()
    flush_table()
    return flowables


def _footer(canvas, document) -> None:
    """Stamp the page number, so a citation of ``p.3`` is checkable on the page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def render(source_path: Path, output_path: Path) -> None:
    """Render one Markdown source into one PDF fixture."""
    styles = _styles()
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=source_path.stem,
        author="Atlas programme management office",
    )
    document.build(
        _flow(source_path.read_text(encoding="utf-8"), styles),
        onFirstPage=_footer,
        onLaterPages=_footer,
    )
    logger.info("wrote %s (%d bytes)", output_path, output_path.stat().st_size)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for source_name, output_name in FIXTURES.items():
        render(SOURCE_DIR / source_name, OUTPUT_DIR / output_name)


if __name__ == "__main__":
    main()
