"""What it takes to show a reply as it arrives, one provider fragment at a time.

Two pieces, and they solve different problems:

:class:`AnswerStreamSink` is the destination -- a screen, or anything that
wants text in order. It is a two-method Protocol on purpose: ``delta`` for
more text, ``reset`` for "forget everything you were shown so far", because
the gateway retries a transient failure by replaying the call from the start,
and a sink that was not told would show the reply twice.

:class:`JsonStringFieldExtractor` is the translation. The answering call
returns structured output -- one JSON object, :class:`~agentic_erp_assistant.
llm.schemas.GroundedAnswer` -- so "stream the answer" cannot mean "forward
provider fragments verbatim": half of a citation object arriving is not
something a chat bubble should render. It means extracting the decoded text
of the ``answer`` field alone, as the surrounding JSON document arrives one
fragment at a time, in whatever order and at whatever split points the
provider happens to flush them. Nothing else in the object is ever emitted.

Why a character state machine and not "try to json.loads() every fragment"
--------------------------------------------------------------------------

A prefix of a JSON document is not JSON: ``{"answer": "The vendor sl`` fails
to parse, and it fails the same way whether one character or one thousand are
still to come. Parsing has to happen incrementally, against a document that
is not yet valid, which is what a hand-rolled state machine is for and a
general-purpose JSON parser is not.
"""

from typing import Protocol, runtime_checkable

__all__ = ["AnswerStreamSink", "JsonStringFieldExtractor"]


@runtime_checkable
class AnswerStreamSink(Protocol):
    """Where reply text goes while it is still arriving.

    Bound at the gateway (:attr:`~agentic_erp_assistant.llm.gateway.
    LLMGateway.stream`), not passed through the engine's ports: the planner
    and the composer stay ignorant of streaming, and a gateway built for
    memory work is built with no sink at all, so a memory proposal can never
    stream into the chat.
    """

    def delta(self, text: str) -> None:
        """More of the reply has arrived, in order. May be called zero or
        more times per attempt."""
        ...

    def reset(self) -> None:
        """Forget everything streamed so far for the call in progress.

        Called before a retried attempt's first delta: a stream that died at
        token 40 is replayed from token 0 on the next attempt, and a sink
        that was not told to reset would show the reply twice.
        """
        ...


_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
"""The single-character JSON escapes. ``\\u`` is handled separately -- it is
the only one that consumes more than the one character after the backslash."""


class JsonStringFieldExtractor:
    """Feed a JSON document a fragment at a time; get back the decoded text of
    one top-level string field as it becomes available.

    Only the string value of ``field`` at object depth 1 is captured. A key
    with the same name nested inside, say, ``citations[...]`` is ignored --
    depth is tracked by counting ``{``/``[`` and ``}``/``]`` outside of
    strings, and only a candidate key read while that count is exactly 1 is
    ever compared to ``field``. Handles fragments that split anywhere:
    mid-escape, mid-key, between a key and its colon, inside a ``\\uXXXX``
    escape, even between the two halves of a surrogate pair. Escapes are
    decoded (``\\n \\" \\\\ \\/ \\b \\f \\r \\t \\uXXXX``); an incomplete
    escape is held back until it completes.
    """

    def __init__(self, field: str = "answer") -> None:
        self.field = field

        self._depth = 0
        self._complete = False

        # Whichever string is open right now, if any: the target field's
        # value (_capturing) or anything else (_in_string). Never both.
        self._capturing = False
        self._in_string = False

        # Bookkeeping for a non-captured string at depth 1: its raw text, so
        # it can be compared to `field` once we learn whether it was a key.
        self._string_is_candidate = False
        self._candidate_buffer = ""
        self._pending_key_text: str | None = None
        self._await_colon = False
        self._expect_value_string = False
        self._generic_escape = False

        # Bookkeeping for the captured string's own escape decoding.
        self._capture_escape = False
        self._capture_unicode_digits: str | None = None
        self._pending_high_surrogate: int | None = None

    @property
    def complete(self) -> bool:
        """Whether the target field's string value has been fully read.

        Once true, :meth:`feed` returns ``""`` forever -- there is nothing
        left this extractor is looking for, in this document or the next
        fragment of it.
        """
        return self._complete

    def feed(self, fragment: str) -> str:
        """Advance the state machine by ``fragment``; return newly decoded
        field text, or ``""`` if none arrived (or the field is already
        :attr:`complete`)."""
        if self._complete:
            return ""
        produced: list[str] = []
        for character in fragment:
            piece = self._consume(character)
            if piece is not None:
                produced.append(piece)
            if self._complete:
                break
        return "".join(produced)

    # -- the state machine, one character at a time -------------------------

    def _consume(self, ch: str) -> str | None:
        if self._capturing:
            return self._consume_capturing(ch)
        if self._in_string:
            self._consume_generic_string(ch)
            return None
        return self._consume_structural(ch)

    def _consume_structural(self, ch: str) -> None:
        """Outside any string: track depth, and watch for ``"field":``."""
        if ch == '"':
            if self._expect_value_string:
                self._expect_value_string = False
                self._capturing = True
            else:
                self._in_string = True
                self._string_is_candidate = self._depth == 1
                self._candidate_buffer = ""
            return None
        if ch in "{[":
            self._depth += 1
            self._await_colon = False
            return None
        if ch in "}]":
            self._depth -= 1
            self._await_colon = False
            return None
        if ch == ":" and self._await_colon:
            self._await_colon = False
            if self._pending_key_text == self.field:
                self._expect_value_string = True
            return None
        if not ch.isspace():
            # A comma, a digit, a bare word (true/false/null) -- none of
            # them can be the colon this key was waiting for.
            self._await_colon = False
        return None

    def _consume_generic_string(self, ch: str) -> None:
        """Inside a string that is not the target's value: track its raw
        text (if it might be a depth-1 key) and find its unescaped close."""
        if self._generic_escape:
            self._generic_escape = False
            if self._string_is_candidate:
                self._candidate_buffer += ch
            return
        if ch == "\\":
            self._generic_escape = True
            return
        if ch == '"':
            self._in_string = False
            if self._string_is_candidate:
                self._pending_key_text = self._candidate_buffer
                self._await_colon = True
            return
        if self._string_is_candidate:
            self._candidate_buffer += ch

    def _consume_capturing(self, ch: str) -> str | None:
        """Inside the target field's own string value: decode escapes as
        they complete, and end capture at the unescaped closing quote."""
        if self._capture_unicode_digits is not None:
            self._capture_unicode_digits += ch
            if len(self._capture_unicode_digits) < 4:
                return None
            code = int(self._capture_unicode_digits, 16)
            self._capture_unicode_digits = None
            if 0xD800 <= code <= 0xDBFF:
                self._pending_high_surrogate = code
                return None
            if self._pending_high_surrogate is not None and 0xDC00 <= code <= 0xDFFF:
                high = self._pending_high_surrogate
                self._pending_high_surrogate = None
                combined = 0x10000 + (high - 0xD800) * 0x400 + (code - 0xDC00)
                return chr(combined)
            self._pending_high_surrogate = None
            return chr(code)

        if self._capture_escape:
            self._capture_escape = False
            if ch == "u":
                self._capture_unicode_digits = ""
                return None
            return _ESCAPES.get(ch, ch)

        if ch == "\\":
            self._capture_escape = True
            return None
        if ch == '"':
            self._capturing = False
            self._complete = True
            return None
        return ch
