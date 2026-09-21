# 0009 — The locator is the format's own address, and the chunk id is the citation tag

**Status:** Accepted (2026-09-06)

## Context

The project rule is that every factual answer about a project document carries
citations, and that a citation without a resolvable reference is a defect. That
puts a requirement on retrieval that is easy to satisfy formally and hard to
satisfy usefully.

The formal version is what most pipelines do: chunk the text, number the chunks,
cite `doc-12#chunk-7`. Every claim then has a reference, and a reader who follows
one arrives at an identifier that means nothing outside the index. It cannot be
checked against the document, it changes the next time the chunker is retuned,
and it is unusable to anyone who was not given the index.

The reference specification this work follows specifies exactly that shape:
`chunk.chunk_id == f"{document.document_id}#chunk-{n}"`.

Two facts about this project make a better answer available. The corpus is
deliberately multi-format, and every one of those formats already has an address
its readers use — a numbered section in Markdown and HTML, a page in a PDF, a row
identifier in a CSV. And `EvidenceSnippet` already carries `source_id` and an
opaque `locator`, rendering `[source_id#locator]` as the tag the model is told to
cite.

## Decision

Loaders emit blocks carrying the address the format actually has, and a chunk
inherits it:

| Format | Locator | Where it comes from |
|---|---|---|
| Markdown, HTML | `§3.2` | The number the heading writes, when it has one; the heading hierarchy when it does not |
| PDF | `p.4` | The page the text was extracted from |
| CSV | `row R-1` | The row identifier column, falling back to position |
| Any, before the first heading | `preamble` | — |

A chunk never spans two addresses. When one section needs several chunks they
become `§3.2 part 1`, `§3.2 part 2`.

The chunk id is `{document_id}#{locator}` — the citation tag without its
brackets. The same string identifies the chunk in the index, appears in the
retrieval trace, appears in the evidence block of the prompt, and appears in the
answer.

A document that numbers its own headings wins over any numbering we would derive.

## Consequences

A citation is checkable by a person holding the document and nothing else.
`[budget-summary-q3#p.2]` is an instruction to open page 2; `[status-report-2026-09#§3.2]`
is an instruction to scroll to the heading that says 3.2. Neither requires the
index, and neither becomes wrong when the chunker is retuned — re-chunking a
section changes how many parts it has, not where the section is.

There is one identifier, not two. Nothing has to join a storage id to a citation
id, and a reviewer reading `retrieval-report.json` and a reviewer reading an
answer are looking at the same strings.

The chunker gives up some packing efficiency for this. A short section becomes a
short chunk, because merging it with its neighbour would force the citation to
name a range, and a reader following `§3.1-3.3` is back to searching for the
sentence. Measured on this corpus the chunks run 21 to 199 tokens against a
320-token budget — smaller than a tuned pipeline would produce, and the eval
harness is where that trade-off gets measured rather than argued about.

Preferring the document's own numbering means the derived numbering is only a
fallback, and the two can collide in a document that numbers some headings and
not others. The loader detects that and disambiguates, rather than letting two
sections claim one address.

Qdrant point ids are a UUIDv5 of the chunk id, because Qdrant accepts only a uuid
or an integer. That is a storage detail with no reader: the chunk id travels in
the payload and is what everything else uses.

## Alternatives considered

**`chunk-{n}`, as specified.** Rejected. It satisfies the letter of the citation
rule and none of its purpose. The rule exists so a reader can check a claim, and
an index-local counter is not checkable.

**Character offsets (`chars 1200-1520`).** Rejected. Precise, stable under
re-chunking, and useless to a human: nobody counts characters into a PDF.

**Both — a structural locator for display and a chunk index for identity.**
Rejected. Two identifiers for one thing is a join, and the day they disagree is
the day a citation points somewhere the retrieval trace does not.

**Allow a chunk to span sections and cite a range.** Rejected; see Consequences.
Packing efficiency is worth less than a citation landing on the passage.
