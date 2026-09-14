"""The five rows the dev database actually wrote must never be written again,
and the rows that were right must survive the change.

Judged by :func:`~agentic_erp_assistant.memory.policy.decide` directly, with
the turn's own words attached to the candidate (``request_text``/
``response_text``) the way ``LLMMemoryProposer`` will carry them once the
refactor lands. The module is marked ``xfail(strict=True)`` until then: strict
means the day Phase 4 removes the marker, a row that starts passing silently
fails the suite instead of quietly becoming XPASS.
"""

import pytest

from agentic_erp_assistant.memory.models import MemoryCandidate
from agentic_erp_assistant.memory.policy import decide

from tests.memory.builders import make_scope
from tests.memory.junk_fixtures import GOOD_ROWS, JUNK_ROWS

pytestmark = pytest.mark.xfail(
    strict=True, reason="the not_established checks arrive with the memory refactor"
)


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
def test_a_junk_row_is_refused(row: dict) -> None:
    verdict = decide(make_candidate(row), scope=make_scope())

    assert verdict.stores is False


@pytest.mark.parametrize("row", GOOD_ROWS, ids=lambda row: row["key"])
def test_a_good_row_is_still_stored(row: dict) -> None:
    verdict = decide(make_candidate(row), scope=make_scope())

    assert verdict.stores is True