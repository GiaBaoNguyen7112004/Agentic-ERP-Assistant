"""Token counting with the tokenizer the API actually bills on.

The runtime has to bound a request before it sends one -- does this prompt plus
its evidence fit the context window, and what will it cost? The cheap answers to
that question (``len(text) / 4``, word counts) are wrong by the most on exactly
the inputs that matter, code and non-English text, and they are wrong silently.

So this module uses ``tiktoken``, the byte-pair encoding the provider bills with,
and the documented chat-format overhead on top of it. What it produces is still a
*pre-call estimate*: it counts the port's message list, and ``developer`` and
``evidence`` are roles no vendor accepts on the wire, so the adapter collapses
them (see
:meth:`~agentic_erp_assistant.llm.ports.LargeLanguageModelClient.complete`) and
the shipped message count can differ by a message or two. This is the budget
guard. ``CompletionResponse.usage`` is the invoice.

Operational note: ``tiktoken`` downloads its encoding file on first use and
caches it. On an offline machine that download is the failure, not this code;
point ``TIKTOKEN_CACHE_DIR`` at a pre-warmed cache.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, runtime_checkable

import tiktoken

from agentic_erp_assistant.llm.ports import Message

__all__ = [
    "count_message_tokens",
    "count_tokens",
    "FALLBACK_ENCODING",
    "TiktokenCounter",
    "TokenCounter",
]

logger = logging.getLogger(__name__)


FALLBACK_ENCODING = "o200k_base"
"""The encoding used when the model name is not in tiktoken's table."""

# The chat format's own overhead, quoted rather than guessed. Every message is
# wrapped in <|start|>role<|message|>...<|end|>, and every request is primed for
# the reply with a further <|start|>assistant<|message|>.
_TOKENS_PER_MESSAGE = 3
_TOKENS_FOR_REPLY = 3


@runtime_checkable
class TokenCounter(Protocol):
    """What the runtime is allowed to assume about counting tokens.

    One implementation today. It is a protocol anyway because the alternative --
    a provider-side counter that trades a round trip for an exact number -- is a
    thing the budget layer may want later, and swapping it should be a change at
    one construction site rather than at every call site.
    """

    def count_tokens(self, text: str, *, model: str) -> int:
        """Count the tokens in a bare string for ``model``."""
        ...

    def count_message_tokens(
        self,
        messages: Sequence[Message],
        *,
        model: str,
    ) -> int:
        """Count a whole request, chat-format overhead included."""
        ...


@lru_cache(maxsize=None)
def _encoding_for(model: str) -> tiktoken.Encoding:
    """Resolve the encoding for a model, falling back rather than raising.

    tiktoken's per-model table lags new releases, so a very recently released --
    but entirely current -- model name will not be in it. A tokenizer that raises
    on such a name is an outage waiting for the next model launch, so an
    unrecognized name falls back to the current default encoding and says so in
    the log rather than doing it invisibly.

    Only ``KeyError`` is caught: that is tiktoken's "no such model", and it is
    the one failure a fallback is the right answer to. Anything else -- a missing
    encoding file, a failed download -- is a real problem and propagates.

    Cached because building an encoding is expensive and, the first time, does
    network I/O.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug(
            "no tiktoken encoding registered for model %r; falling back to %s",
            model,
            FALLBACK_ENCODING,
        )
        return tiktoken.get_encoding(FALLBACK_ENCODING)


class TiktokenCounter:
    """Counts with the provider's own byte-pair encoding."""

    def count_tokens(self, text: str, *, model: str) -> int:
        """Count the tokens in a bare string -- content only, no framing."""
        return len(_encoding_for(model).encode(text))

    def count_message_tokens(
        self,
        messages: Sequence[Message],
        *,
        model: str,
    ) -> int:
        """Count a request the way the chat format charges for it.

        Per message: the framing overhead, plus the encoded role, plus the
        encoded content. The role string is counted because the format really
        does send it -- which means this project's longer role names
        (``developer``, ``evidence``) cost what they actually cost. Reply priming
        is added once for the request, not once per message.
        """
        encoding = _encoding_for(model)
        total = _TOKENS_FOR_REPLY
        for message in messages:
            total += _TOKENS_PER_MESSAGE
            total += len(encoding.encode(message["role"]))
            total += len(encoding.encode(message["content"]))
        return total


_DEFAULT_COUNTER = TiktokenCounter()


def count_tokens(text: str, *, model: str) -> int:
    """Count the tokens in ``text`` for ``model``.

    Any model name works: one tiktoken does not recognize falls back to
    :data:`FALLBACK_ENCODING` instead of raising.
    """
    return _DEFAULT_COUNTER.count_tokens(text, model=model)


def count_message_tokens(messages: Sequence[Message], *, model: str) -> int:
    """Count a whole request for ``model``, chat-format overhead included."""
    return _DEFAULT_COUNTER.count_message_tokens(messages, model=model)
