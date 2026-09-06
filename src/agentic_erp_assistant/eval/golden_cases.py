"""The questions retrieval is held to, written down before the numbers are.

A golden set is only worth anything if it was written to be failed. These cases
are chosen so that each one breaks for a different, nameable reason -- a format
whose locator stopped working, a scope that stopped being enforced, a floor set
too low -- rather than nine variations of "does search work".

Four of the nine expect something to be *absent*, which is deliberate and is
where most of the value is. Getting a topical query right is the easy half of
retrieval; the half that goes wrong quietly is returning a document the reader
was not entitled to, or answering a question the corpus cannot support. Those
failures do not look like failures downstream: they look like confident,
well-cited answers.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from agentic_erp_assistant.rag.access import RetrievalContext

__all__ = ["DEFAULT_CASES", "Expectation", "GoldenCase", "READER_SCOPES", "FINANCE_SCOPES"]


Expectation = Literal["present", "absent", "empty"]
"""What a case is asserting about the top ``k``.

``present`` -- at least one named document is there. The set names the documents
that could legitimately answer, and one of them is enough, because an answer
needs one source and naming several would fail a retriever for picking the other
good one.

``absent`` -- none of the named documents is there. Other results are fine and
expected: a reader denied the budget PDF should still get the status report.

``empty`` -- nothing came back at all. The strongest assertion, and the one that
proves the pipeline refuses before it pays for a synthesis call.
"""


READER_SCOPES = frozenset({"project.docs.read"})
"""What an ordinary project member holds."""

FINANCE_SCOPES = frozenset({"project.docs.read", "project.docs.finance.read"})
"""What somebody in the finance group holds as well."""


@dataclass(frozen=True)
class GoldenCase:
    """One question, one asking context, and what must come back."""

    case_id: str
    """Stable name. Appears in the evidence report, so it does not change once a
    number has been published against it."""

    query: str
    """Asked exactly as written."""

    actor: str
    project_code: str
    scopes: frozenset[str]

    expectation: Expectation
    documents: frozenset[str] = frozenset()
    """The documents the expectation is about. Empty only for ``empty``."""

    why: str = ""
    """What this case is protecting. Read by whoever has to fix it when it goes
    red, which is the moment the reason is least obvious and most needed."""

    def __post_init__(self) -> None:
        if self.expectation == "empty" and self.documents:
            raise ValueError(
                f"{self.case_id}: an 'empty' case asserts that nothing came "
                f"back, so naming documents would be describing a different "
                f"assertion"
            )
        if self.expectation != "empty" and not self.documents:
            raise ValueError(
                f"{self.case_id}: a {self.expectation!r} case has to name the "
                f"documents it is about"
            )

    @property
    def context(self) -> RetrievalContext:
        return RetrievalContext(
            actor=self.actor, project_code=self.project_code, scopes=self.scopes
        )

    def scores(self, retrieved_documents: Iterable[str]) -> bool:
        """Whether this case passed, given the documents in the top ``k``."""
        found = set(retrieved_documents)
        if self.expectation == "empty":
            return not found
        if self.expectation == "absent":
            return not (found & self.documents)
        return bool(found & self.documents)


def _atlas(
    case_id: str,
    query: str,
    expectation: Expectation,
    documents: Iterable[str] = (),
    *,
    actor: str = "priya",
    scopes: frozenset[str] = READER_SCOPES,
    why: str = "",
) -> GoldenCase:
    return GoldenCase(
        case_id=case_id,
        query=query,
        actor=actor,
        project_code="atlas",
        scopes=scopes,
        expectation=expectation,
        documents=frozenset(documents),
        why=why,
    )


DEFAULT_CASES: tuple[GoldenCase, ...] = (
    # -- ordinary topical retrieval, one per format ------------------------
    _atlas(
        "m2-slip-cause",
        "Why is milestone M2 late and by how much?",
        "present",
        ["status-report-2026-09", "steering-minutes-2026-08"],
        why=(
            "The plain case: a question answered in prose, in Markdown, by more "
            "than one document. Either source is a correct answer."
        ),
    ),
    _atlas(
        "sprint-13-progress",
        "How many story points has sprint 13 completed against its commitment?",
        "present",
        ["sprint-13-report"],
        why=(
            "Numbers in a table-free narrative, in a document whose subject "
            "overlaps heavily with the status report. Tests that the chunk "
            "header keeps the two apart."
        ),
    ),
    _atlas(
        "risk-r2-owner",
        "Who owns risk R-2 and what is the mitigation?",
        "present",
        ["risk-register"],
        why=(
            "A CSV row addressed by its own identifier. This is the case the "
            "lexical half exists for -- an exact string a dense index blurs into "
            "five passages about risk in general."
        ),
    ),
    _atlas(
        "vendor-escalation-rule",
        "How many unanswered written requests justify contractual escalation to a vendor?",
        "present",
        ["support-policy"],
        why=(
            "A policy document, a different content class from the delivery "
            "reporting that dominates the corpus. If a routing or weighting "
            "change ever buries policy under status reports, this is what says so."
        ),
    ),
    _atlas(
        "payments-interface-realtime",
        "Which inbound interface is a synchronous API rather than a batch file?",
        "present",
        ["architecture-notes"],
        why=(
            "HTML, and a question whose answer sits under a heading two levels "
            "deep. Proves the HTML loader still produces section locators."
        ),
    ),
    # -- the PDF, which is the only source of a page locator ---------------
    _atlas(
        "forecast-overrun-driver",
        "What is the forecast at completion, and which cost category drives the overrun?",
        "present",
        ["budget-summary-q3"],
        actor="wei",
        scopes=FINANCE_SCOPES,
        why=(
            "The only document that can answer this, and the only PDF. A hit "
            "here is also the evidence that a citation can carry a page number "
            "rather than a chunk index."
        ),
    ),
    # -- refusals, which is where the value is ------------------------------
    _atlas(
        "forecast-overrun-denied",
        "What is the forecast at completion, and which cost category drives the overrun?",
        "absent",
        ["budget-summary-q3"],
        why=(
            "The same question as forecast-overrun-driver, asked by somebody "
            "without the finance scope. The identical query is the point: only "
            "the entitlement differs, so a pass here cannot be luck."
        ),
    ),
    _atlas(
        "cross-project-orion",
        "What is the status of the Orion CRM migration and its budget?",
        "absent",
        ["orion-status-report-2026-09"],
        why=(
            "A valid scope on the wrong project. The Orion report is the best "
            "textual match in the whole corpus for this query, so the only thing "
            "that can keep it out is the project half of the access rule."
        ),
    ),
    _atlas(
        "unanswerable",
        "What is the weather forecast in Hanoi next week?",
        "empty",
        why=(
            "Nothing in the corpus answers this. The pipeline has to return "
            "nothing rather than the four nearest passages, because the graph "
            "calls the model on a non-empty result -- so this case is the "
            "difference between refusing for free and paying to be told there "
            "is no evidence."
        ),
    ),
)
"""The nine cases. Five topical across four formats, three access refusals, one
unanswerable question."""
