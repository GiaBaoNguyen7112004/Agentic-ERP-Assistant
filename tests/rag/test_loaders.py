"""What each format is allowed to be cited by.

These tests are mostly about locators rather than about text, because the text
is the easy half. A loader that returns the right words under the wrong address
produces an answer that looks grounded and sends its reader to the wrong place,
and nothing downstream can detect that.
"""

from pathlib import Path

import pytest

from agentic_erp_assistant.rag.loaders import (
    PREAMBLE,
    DocumentBlock,
    UnsupportedFormat,
    load_blocks,
)
from agentic_erp_assistant.state.evidence import EvidenceSnippet

CORPUS = Path(__file__).resolve().parents[2] / "data" / "documents"


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def locators(blocks: tuple[DocumentBlock, ...]) -> list[str]:
    return [block.locator for block in blocks]


# -- Markdown ----------------------------------------------------------------


def test_a_document_numbers_its_own_sections(tmp_path: Path) -> None:
    """The written number wins, even where it is not the positional one.

    A reader resolving §7 scrolls to the heading that says 7. A locator we
    derived positionally would have said §2 and sent them to the wrong section.
    """
    path = write(
        tmp_path,
        "doc.md",
        "# Title\n\n## 1. First\n\nalpha\n\n## 7. Seventh\n\nbeta\n\n### 7.3 Deep\n\ngamma\n",
    )
    blocks = load_blocks(path)
    assert locators(blocks) == ["§1", "§7", "§7.3"]
    assert blocks[2].heading_path == ("7. Seventh", "7.3 Deep")


def test_an_unnumbered_document_is_numbered_positionally(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "doc.md",
        "# Title\n\n## First\n\nalpha\n\n### Nested\n\nbeta\n\n## Second\n\ngamma\n",
    )
    assert locators(load_blocks(path)) == ["§1", "§1.1", "§2"]


def test_text_before_the_first_heading_is_the_preamble(tmp_path: Path) -> None:
    path = write(tmp_path, "doc.md", "# Title\n\nintro words\n\n## 1. One\n\nbody\n")
    assert locators(load_blocks(path)) == [PREAMBLE, "§1"]


def test_a_lone_top_heading_is_the_title_not_section_one(tmp_path: Path) -> None:
    """One h1 above everything names the file; several h1s are sections."""
    titled = write(tmp_path, "a.md", "# Title\n\n## One\n\nalpha\n")
    assert locators(load_blocks(titled)) == ["§1"]

    flat = write(tmp_path, "b.md", "# One\n\nalpha\n\n# Two\n\nbeta\n")
    assert locators(load_blocks(flat)) == ["§1", "§2"]


def test_a_hash_inside_a_fence_is_not_a_heading(tmp_path: Path) -> None:
    """Otherwise a code comment silently renumbers every section after it."""
    path = write(
        tmp_path,
        "doc.md",
        "# Title\n\n## 1. One\n\n```\n# not a heading\n```\n\n## 2. Two\n\nbody\n",
    )
    blocks = load_blocks(path)
    assert locators(blocks) == ["§1", "§2"]
    assert "# not a heading" in blocks[0].text


def test_a_table_stays_one_block(tmp_path: Path) -> None:
    """Split into rows, a cell would arrive in a prompt without its header."""
    path = write(
        tmp_path,
        "doc.md",
        "# Title\n\n## 1. One\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n",
    )
    (block,) = load_blocks(path)
    assert block.text.count("\n") == 3


# -- HTML --------------------------------------------------------------------


def test_html_headings_and_blocks(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "doc.html",
        "<html><head><title>T</title><style>p{color:red}</style></head><body>"
        "<h1>Title</h1><p>intro</p>"
        "<h2>1. Context</h2><p>alpha</p>"
        "<h3>1.1 Deep</h3><p>beta</p>"
        "<ul><li>gamma</li></ul>"
        "</body></html>",
    )
    blocks = load_blocks(path)
    assert locators(blocks) == [PREAMBLE, "§1", "§1.1", "§1.1"]
    assert [block.text for block in blocks] == ["intro", "alpha", "beta", "gamma"]


def test_an_html_table_is_gathered_whole(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "doc.html",
        "<body><h1>T</h1><h2>1. S</h2><table>"
        "<tr><th>Requirement</th><th>Target</th></tr>"
        "<tr><td>Availability</td><td>99.7%</td></tr>"
        "</table></body>",
    )
    (block,) = load_blocks(path)
    assert block.text == "Requirement | Target\nAvailability | 99.7%"


# -- CSV ---------------------------------------------------------------------


def test_a_csv_row_is_addressed_by_its_own_identifier(tmp_path: Path) -> None:
    """A stable key survives a re-sort of the file; a row number does not."""
    path = write(
        tmp_path,
        "doc.csv",
        "risk_id,title,severity\nR-1,Migration,high\nR-2,Vendor,medium\n",
    )
    blocks = load_blocks(path)
    assert locators(blocks) == ["row R-1", "row R-2"]
    assert blocks[0].text == "risk_id: R-1\ntitle: Migration\nseverity: high"


def test_a_csv_without_an_identifier_column_falls_back_to_position(
    tmp_path: Path,
) -> None:
    path = write(tmp_path, "doc.csv", "name,value\nalpha,1\nbeta,2\n")
    assert locators(load_blocks(path)) == ["row 1", "row 2"]


def test_a_csv_blank_field_is_not_rendered(tmp_path: Path) -> None:
    path = write(tmp_path, "doc.csv", "id,title,owner\nA,Thing,\n")
    (block,) = load_blocks(path)
    assert block.text == "id: A\ntitle: Thing"


# -- PDF, against the real committed fixture ---------------------------------


def test_the_pdf_is_addressed_by_page() -> None:
    blocks = load_blocks(CORPUS / "budget-summary-q3.pdf")
    pages = sorted({block.locator for block in blocks})
    assert pages == ["p.1", "p.2", "p.3"]


def test_the_rendered_page_number_is_not_indexed() -> None:
    """Furniture that repeats on every page matches everything and means nothing."""
    blocks = load_blocks(CORPUS / "budget-summary-q3.pdf")
    assert not any(block.text.strip().startswith("Page ") for block in blocks)


def test_the_pdf_keeps_its_numbers_where_a_reader_can_check_them() -> None:
    blocks = load_blocks(CORPUS / "budget-summary-q3.pdf")
    page_one = " ".join(b.text for b in blocks if b.locator == "p.1")
    assert "480,000" in page_one and "515,000" in page_one


# -- the rule that spans every format ----------------------------------------


def test_no_loader_can_produce_a_locator_that_forges_a_citation_tag() -> None:
    """The corpus, end to end, through the type that will carry the address.

    ``EvidenceSnippet`` rejects the characters a tag is built from. Running the
    real documents through it here means a loader cannot introduce a locator
    that only fails much later, at the point evidence enters a prompt.
    """
    for path in sorted(CORPUS.iterdir()):
        if not path.is_file() or path.suffix == ".json":
            continue
        blocks = load_blocks(path)
        assert blocks, f"{path.name} produced no blocks"
        for block in blocks:
            snippet = EvidenceSnippet(
                source_id=path.stem, locator=block.locator, text=block.text
            )
            assert snippet.tag.startswith(f"[{path.stem}#")


def test_an_unknown_format_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    path = write(tmp_path, "doc.rtf", "whatever")
    with pytest.raises(UnsupportedFormat):
        load_blocks(path)
