"""The counter is a budget guard, so the tests pin the numbers, not the shape.

Two things matter here. The count must be tiktoken's own count, with no fudge
factor of ours layered on top; and an unrecognized model name must still produce
a number, because tiktoken's per-model table lags releases and a tokenizer that
raises on a current model takes the whole runtime down with it.
"""

import pytest
import tiktoken

from agentic_erp_assistant.llm.prompts import DEVELOPER_CONTRACT, build_messages
from agentic_erp_assistant.llm.schemas import EvidenceSnippet
from agentic_erp_assistant.llm.tokenizer import (
    FALLBACK_ENCODING,
    TiktokenCounter,
    TokenCounter,
    count_message_tokens,
    count_tokens,
    split_into_token_windows,
)

MODEL = "gpt-4o"
UNKNOWN_MODEL = "gpt-9-imaginary-2099"

# Both from the module under test, so the assertions below stay arithmetic
# rather than restating magic numbers.
TOKENS_PER_MESSAGE = 3
TOKENS_FOR_REPLY = 3


def test_a_known_string_matches_what_tiktoken_itself_reports() -> None:
    """The named acceptance criterion, asserted two ways.

    Against tiktoken directly, which proves no fudge factor of ours is layered
    on top; and against the literal count, which fails loudly if the encoding
    changes underneath us instead of quietly agreeing with itself.
    """
    expected = len(tiktoken.encoding_for_model(MODEL).encode("hello world"))

    assert count_tokens("hello world", model=MODEL) == expected
    assert count_tokens("hello world", model=MODEL) == 2


def test_a_model_tiktoken_has_never_heard_of_still_returns_a_count() -> None:
    """The named acceptance criterion: an unknown model must not raise."""
    counted = count_tokens("hello world", model=UNKNOWN_MODEL)

    assert isinstance(counted, int)
    assert counted > 0


def test_the_fallback_is_the_encoding_we_said_it_was() -> None:
    """Pins which encoding we fell back to, not merely that we survived."""
    expected = len(tiktoken.get_encoding(FALLBACK_ENCODING).encode("hello world"))

    assert count_tokens("hello world", model=UNKNOWN_MODEL) == expected


def test_a_message_costs_its_content_plus_its_framing_and_role() -> None:
    encoding = tiktoken.encoding_for_model(MODEL)
    content = "Sprint 12 closes on 30 September."
    messages = [{"role": "user", "content": content}]

    counted = count_message_tokens(messages, model=MODEL)

    assert counted == (
        TOKENS_FOR_REPLY
        + TOKENS_PER_MESSAGE
        + len(encoding.encode("user"))
        + len(encoding.encode(content))
    )


def test_reply_priming_is_charged_once_for_the_request_not_once_per_message() -> None:
    encoding = tiktoken.encoding_for_model(MODEL)
    first = {"role": "user", "content": "one"}
    second = {"role": "assistant", "content": "two"}

    delta = count_message_tokens([first, second], model=MODEL) - count_message_tokens(
        [first], model=MODEL
    )

    assert delta == (
        TOKENS_PER_MESSAGE
        + len(encoding.encode("assistant"))
        + len(encoding.encode("two"))
    )


def test_an_empty_request_costs_only_the_reply_priming() -> None:
    assert count_tokens("", model=MODEL) == 0
    assert count_message_tokens([], model=MODEL) == TOKENS_FOR_REPLY


def test_a_real_prompt_counts() -> None:
    """Ties the counter to the thing it exists to measure."""
    messages = build_messages(
        "When does sprint 12 close?",
        [
            EvidenceSnippet(
                source_id="doc-12",
                locator="3.2",
                text="Sprint 12 closes on 30 September.",
            )
        ],
    )

    counted = count_message_tokens(messages, model=MODEL)

    assert counted > count_tokens(DEVELOPER_CONTRACT, model=MODEL)


def test_the_tiktoken_counter_satisfies_the_port() -> None:
    assert isinstance(TiktokenCounter(), TokenCounter)


def test_a_class_missing_a_method_does_not_satisfy_the_port() -> None:
    class NotACounter:
        def count_tokens(self, text: str, *, model: str) -> int:
            return 0

    assert not isinstance(NotACounter(), TokenCounter)


# -- splitting ---------------------------------------------------------------
#
# The splitter lives beside the counter so both read the same encoding. These
# tests assert the property that matters downstream: a window never exceeds the
# budget it was cut to, whatever the text is.


def test_text_that_already_fits_comes_back_as_one_window() -> None:
    assert split_into_token_windows(
        "a short passage", model=MODEL, window_tokens=100
    ) == ["a short passage"]


def test_no_window_exceeds_the_budget() -> None:
    text = "migration exceptions against a tolerance of ten. " * 60
    windows = split_into_token_windows(text, model=MODEL, window_tokens=40)

    assert len(windows) > 1
    assert all(count_tokens(window, model=MODEL) <= 40 for window in windows)


def test_windows_overlap_by_the_requested_amount() -> None:
    """A sentence spanning a cut has to survive whole in at least one window."""
    text = " ".join(f"word{index}" for index in range(200))
    windows = split_into_token_windows(
        text, model=MODEL, window_tokens=50, overlap_tokens=10
    )

    assert len(windows) > 1
    assert windows[0][-20:] in windows[1]


def test_the_split_loses_nothing_when_there_is_no_overlap() -> None:
    text = " ".join(f"word{index}" for index in range(200))
    windows = split_into_token_windows(text, model=MODEL, window_tokens=50)

    assert "".join(windows) == text


def test_empty_text_produces_no_windows() -> None:
    assert split_into_token_windows("", model=MODEL, window_tokens=10) == []


def test_an_overlap_that_fills_the_window_is_refused() -> None:
    """It would never advance, so the split would not terminate."""
    with pytest.raises(ValueError, match="never advances"):
        split_into_token_windows(
            "text", model=MODEL, window_tokens=10, overlap_tokens=10
        )


def test_a_non_positive_window_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        split_into_token_windows("text", model=MODEL, window_tokens=0)


def test_an_unknown_model_still_splits() -> None:
    windows = split_into_token_windows(
        "a passage " * 40, model=UNKNOWN_MODEL, window_tokens=20
    )
    assert windows
    assert all(
        count_tokens(window, model=UNKNOWN_MODEL) <= 20 for window in windows
    )
