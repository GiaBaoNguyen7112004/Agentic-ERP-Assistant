"""The window is a promise ("every turn inside HISTORY_TURN_LIMIT fits") backed
by arithmetic, not by hoping the budget is big enough on the day it matters."""

from datetime import UTC, datetime, timedelta

import tiktoken

from agentic_erp_assistant.context.history_injection import (
    CLIP_MARKER,
    HISTORY_BUDGET_TOKENS,
    HISTORY_TURN_LIMIT,
    REQUEST_MAX_TOKENS,
    RESPONSE_MAX_TOKENS,
    TURN_FRAMING_TOKENS,
    clip_to_tokens,
    select_history,
    strip_citations,
)
from agentic_erp_assistant.engine.nodes import _with_sources
from agentic_erp_assistant.state.conversation import ConversationTurn

MODEL = "gpt-4o"
START = datetime(2026, 9, 8, 9, 0, tzinfo=UTC)


def tokens(text: str, *, model: str = MODEL) -> int:
    return len(tiktoken.encoding_for_model(model).encode(text))


def turn(
    trace_id: str = "run-1",
    *,
    request: str = "How is M2 tracking?",
    response: str | None = "On track.",
    started_at: datetime = START,
    **overrides: object,
) -> ConversationTurn:
    fields: dict[str, object] = {
        "trace_id": trace_id,
        "session_id": "sess-1",
        "actor": "bao",
        "request": request,
        "response": response,
        "route": "answer",
        "started_at": started_at,
        "finished_at": started_at,
    }
    fields.update(overrides)
    return ConversationTurn(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The arithmetic invariant: the window always fits the budget
# --------------------------------------------------------------------------


def test_the_window_always_fits_the_budget() -> None:
    assert (
        HISTORY_TURN_LIMIT * (REQUEST_MAX_TOKENS + RESPONSE_MAX_TOKENS + TURN_FRAMING_TOKENS)
        <= HISTORY_BUDGET_TOKENS
    )


# --------------------------------------------------------------------------
# strip_citations
# --------------------------------------------------------------------------


def test_strip_undoes_with_sources() -> None:
    body = _with_sources("The vendor slipped.", ["[doc-1#3.2]"])

    assert strip_citations(body) == "The vendor slipped."


def test_no_tag_survives_even_inline() -> None:
    text = "see [doc-1#p.4] and [r#R-1] for details"

    stripped = strip_citations(text)

    assert "[" not in stripped
    assert "#" not in stripped
    assert "]" not in stripped


def test_a_locator_with_a_space_is_stripped_too() -> None:
    """CSV rows and split sections produce locators with spaces (ADR 0009);
    the old pattern refused whitespace and let a real tag survive inline."""
    text = "see [risk-register#row R-2] and [status-report-2026-09#§3.2 part 1] for details"

    stripped = strip_citations(text)

    assert "[" not in stripped
    assert "#" not in stripped
    assert "]" not in stripped
    assert "risk-register" not in stripped
    assert "row R-2" not in stripped


def test_strip_runs_before_clip() -> None:
    """A trailer cut mid-tag by the clip would leave a stray '[doc#' fragment;
    stripping first means the clip never sees a tag at all."""
    long_body = ("word " * 500).strip()
    body = _with_sources(long_body, ["[sprint-12-report.md#3.2]"])

    clipped = clip_to_tokens(
        strip_citations(body), model=MODEL, max_tokens=RESPONSE_MAX_TOKENS
    )

    assert "[" not in clipped
    assert "#" not in clipped


# --------------------------------------------------------------------------
# clip_to_tokens
# --------------------------------------------------------------------------


def test_clip_is_by_tokens_not_chars() -> None:
    """A Vietnamese sentence costs more tokens per character than an English
    one; a character budget would clip it far short of its token cost."""
    vietnamese = "Ngân sách của dự án atlas đang được theo dõi rất sát sao. " * 10

    clipped = clip_to_tokens(vietnamese, model=MODEL, max_tokens=20)

    assert clipped.endswith(CLIP_MARKER)
    assert tokens(clipped[: -len(CLIP_MARKER)]) <= 20


def test_text_that_already_fits_is_returned_unchanged() -> None:
    short = "On track."

    assert clip_to_tokens(short, model=MODEL, max_tokens=RESPONSE_MAX_TOKENS) == short


def test_empty_text_is_returned_unchanged() -> None:
    assert clip_to_tokens("", model=MODEL, max_tokens=10) == ""


# --------------------------------------------------------------------------
# select_history
# --------------------------------------------------------------------------


def test_newest_turns_win_and_order_is_oldest_first() -> None:
    turns = [
        turn(f"run-{i}", request=f"question {i}", started_at=START + timedelta(minutes=i))
        for i in range(HISTORY_TURN_LIMIT + 2)
    ]

    selection = select_history(turns, budget_tokens=HISTORY_BUDGET_TOKENS, model=MODEL)

    kept_ids = [t.trace_id for t in selection.selected]
    assert kept_ids == [f"run-{i}" for i in range(2, HISTORY_TURN_LIMIT + 2)]


def test_more_than_the_limit_keeps_the_newest() -> None:
    turns = [
        turn(f"run-{i}", started_at=START + timedelta(minutes=i))
        for i in range(HISTORY_TURN_LIMIT * 3)
    ]

    selection = select_history(turns, budget_tokens=HISTORY_BUDGET_TOKENS, model=MODEL)

    assert len(selection.selected) == HISTORY_TURN_LIMIT
    assert selection.selected[-1].trace_id == f"run-{HISTORY_TURN_LIMIT * 3 - 1}"


def test_within_the_limit_nothing_is_dropped_for_budget() -> None:
    turns = [
        turn(f"run-{i}", started_at=START + timedelta(minutes=i))
        for i in range(HISTORY_TURN_LIMIT)
    ]

    selection = select_history(turns, budget_tokens=HISTORY_BUDGET_TOKENS, model=MODEL)

    assert len(selection.selected) == HISTORY_TURN_LIMIT
    assert selection.dropped == ()


def test_selection_clips_and_strips_the_turns_it_keeps() -> None:
    poisoned = turn(
        response=_with_sources("Sprint 12 closes on 30 September.", ["[doc-1#3.2]"])
    )

    selection = select_history(
        [poisoned], budget_tokens=HISTORY_BUDGET_TOKENS, model=MODEL
    )

    assert selection.selected[0].response == "Sprint 12 closes on 30 September."


def test_selected_copies_are_what_the_prompt_renders() -> None:
    from agentic_erp_assistant.llm.prompts import _render_history

    turns = [turn(f"run-{i}", started_at=START + timedelta(minutes=i)) for i in range(2)]

    selection = select_history(turns, budget_tokens=HISTORY_BUDGET_TOKENS, model=MODEL)
    block = _render_history(selection.selected)

    for one in selection.selected:
        assert one.request in block
        assert one.response in block
