"""The five rows the dev database actually held on 2026-09-14 must never be
written again, and the rows that were right must survive the change.

Judged by :func:`~agentic_erp_assistant.memory.policy.decide` directly, with
the turn's own words attached to the candidate (``request_text``/
``response_text``) the way ``LLMMemoryProposer`` carries them. Each junk row
asserts *which* rule refuses it, because a rule that refuses for the wrong
reason is a rule that will stop refusing the day the wrong reason moves.
"""

import pytest

from agentic_erp_assistant.memory.models import MemoryCandidate
from agentic_erp_assistant.memory.policy import decide

from tests.memory.builders import make_scope
from tests.memory.junk_fixtures import GOOD_ROWS, JUNK_ROWS

# Which not_established rule each junk row falls to. Rows 1-3 are absence
# claims (row 1 is also a self-description; the absence pattern is checked
# first, so it wins the reason). Row 4 restates the turn's own reply -- the
# reply-restatement rule, not the volatile check: 3b runs before step 5, so
# "days remaining" is the second line of defence, not the first. Row 5 is a
# preference the user never stated.
JUNK_RULES = {
    "user_name_unavailable": "absence",
    "project_atlas_sprints_unavailable": "absence",
    "sprint_information_unavailable": "absence",
    "sprint_13_progress": "restated reply",
    "clarity_in_requests": "unstated preference",
}


def make_candidate(row: dict) -> MemoryCandidate:
    return MemoryCandidate(
        kind=row["kind"],
        key=row["key"],
        statement=row["statement"],
        confidence=row["confidence"],
        required_scope="project.docs.read",
        request_text=row["request"],
        response_text=row["response"],
    )


@pytest.mark.parametrize("row", JUNK_ROWS, ids=lambda row: row["key"])
def test_a_junk_row_is_refused_as_not_established(row: dict) -> None:
    verdict = decide(make_candidate(row), scope=make_scope())

    assert verdict.rejection == "not_established"
    assert verdict.reason  # and the audit row can say why in one line


@pytest.mark.parametrize(
    ("row_key", "rule"), sorted(JUNK_RULES.items()), ids=lambda value: value[0]
)
def test_each_row_falls_to_the_rule_the_plan_names(row_key: str, rule: str) -> None:
    """The reason text carries the tell the rule matched -- absence wording for
    rows 1-3, reply restatement for row 4, request ownership for row 5."""
    row = next(item for item in JUNK_ROWS if item["key"] == row_key)
    verdict = decide(make_candidate(row), scope=make_scope())

    if rule == "absence":
        assert "not found" in verdict.reason or "absence" in verdict.reason
    elif rule == "restated reply":
        assert "restates this turn's own reply" in verdict.reason
    else:
        assert "stated by the user" in verdict.reason


@pytest.mark.parametrize("row", GOOD_ROWS, ids=lambda row: row["key"])
def test_a_good_row_is_still_stored(row: dict) -> None:
    verdict = decide(make_candidate(row), scope=make_scope())

    assert verdict.stores is True