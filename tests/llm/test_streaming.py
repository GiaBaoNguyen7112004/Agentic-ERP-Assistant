"""JsonStringFieldExtractor: extracting one string field out of a JSON
document that arrives a fragment at a time, split anywhere -- including
inside an escape.

The load-bearing property, proved for every sample below: however the
document is cut into two pieces, feeding them in order produces the same
decoded text as feeding the whole thing at once.
"""

import json

import pytest

from agentic_erp_assistant.llm.streaming import AnswerStreamSink, JsonStringFieldExtractor


def whole(field: str, document: str) -> str:
    """The reference decode: feed the entire document in one call."""
    extractor = JsonStringFieldExtractor(field)
    return extractor.feed(document)


# --------------------------------------------------------------------------
# Every split index, on realistic documents
# --------------------------------------------------------------------------

SIMPLE = json.dumps(
    {
        "answer": "The vendor slipped\nby two days, cost \"a lot\" \\ literally.",
        "citations": [{"source_id": "risk-1", "locator": "row R-1"}],
        "grounded": True,
        "confidence": 0.9,
    }
)

NESTED_ANSWER_KEY = json.dumps(
    {
        "answer": "M2 slipped two weeks after a vendor delay.",
        "citations": [
            {"source_id": "m2-status.md", "locator": "p.2", "answer": "not this one"}
        ],
        "grounded": True,
        "confidence": 0.85,
    }
)

ANSWER_AFTER_CITATIONS = json.dumps(
    {
        "citations": [{"source_id": "x", "locator": "p.1"}],
        "answer": "Order does not matter.",
        "grounded": True,
        "confidence": 0.5,
    }
)

NO_ANSWER_FIELD = json.dumps({"citations": [], "grounded": False})

EMOJI = json.dumps({"answer": "Shipped \U0001F600 today."})


@pytest.mark.parametrize(
    "document,expected",
    [
        (SIMPLE, "The vendor slipped\nby two days, cost \"a lot\" \\ literally."),
        (NESTED_ANSWER_KEY, "M2 slipped two weeks after a vendor delay."),
        (ANSWER_AFTER_CITATIONS, "Order does not matter."),
        (EMOJI, "Shipped \U0001F600 today."),
    ],
)
def test_the_whole_document_at_once_decodes_correctly(
    document: str, expected: str
) -> None:
    assert whole("answer", document) == expected


@pytest.mark.parametrize(
    "document,expected",
    [
        (SIMPLE, "The vendor slipped\nby two days, cost \"a lot\" \\ literally."),
        (NESTED_ANSWER_KEY, "M2 slipped two weeks after a vendor delay."),
        (ANSWER_AFTER_CITATIONS, "Order does not matter."),
        (EMOJI, "Shipped \U0001F600 today."),
    ],
)
def test_every_two_way_split_produces_the_same_text(
    document: str, expected: str
) -> None:
    for k in range(len(document) + 1):
        extractor = JsonStringFieldExtractor("answer")
        got = extractor.feed(document[:k]) + extractor.feed(document[k:])
        assert got == expected, f"split at {k} in {document!r}"
        assert extractor.complete


@pytest.mark.parametrize(
    "document,expected",
    [(SIMPLE, "The vendor slipped\nby two days, cost \"a lot\" \\ literally.")],
)
def test_every_three_way_split_produces_the_same_text(
    document: str, expected: str
) -> None:
    n = len(document)
    for i in range(0, n + 1, 3):
        for j in range(i, n + 1, 5):
            extractor = JsonStringFieldExtractor("answer")
            got = (
                extractor.feed(document[:i])
                + extractor.feed(document[i:j])
                + extractor.feed(document[j:])
            )
            assert got == expected, f"split at {i},{j}"


def test_no_matching_field_yields_nothing() -> None:
    extractor = JsonStringFieldExtractor("answer")

    assert extractor.feed(NO_ANSWER_FIELD) == ""
    assert not extractor.complete


def test_complete_is_false_until_the_closing_quote_arrives() -> None:
    extractor = JsonStringFieldExtractor("answer")

    extractor.feed('{"answer": "still going')
    assert not extractor.complete

    extractor.feed('"}')
    assert extractor.complete


def test_feed_after_complete_always_returns_empty() -> None:
    extractor = JsonStringFieldExtractor("answer")
    extractor.feed(SIMPLE)

    assert extractor.feed('{"answer": "should never appear"}') == ""


def test_an_emoji_split_between_its_two_surrogate_escapes_decodes_to_one_character() -> (
    None
):
    # 😀 is the UTF-16 surrogate pair json.dumps produces for the
    # emoji above; find the midpoint of that escape sequence and split there.
    split_at = EMOJI.index("\\ude00") + 3  # inside the second \u escape

    extractor = JsonStringFieldExtractor("answer")
    got = extractor.feed(EMOJI[:split_at]) + extractor.feed(EMOJI[split_at:])

    assert got == "Shipped \U0001F600 today."


def test_a_field_name_other_than_answer_is_honoured() -> None:
    document = json.dumps({"summary": "custom field", "answer": "not this one"})

    extractor = JsonStringFieldExtractor("summary")

    assert extractor.feed(document) == "custom field"


# --------------------------------------------------------------------------
# AnswerStreamSink: a two-method Protocol
# --------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.resets = 0

    def delta(self, text: str) -> None:
        self.deltas.append(text)

    def reset(self) -> None:
        self.resets += 1
        self.deltas.clear()


def test_a_plain_class_satisfies_the_sink_protocol_structurally() -> None:
    assert isinstance(RecordingSink(), AnswerStreamSink)
