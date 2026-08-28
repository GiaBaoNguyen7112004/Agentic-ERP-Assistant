"""The typed provider port: the contract every real LLM call goes through.

Nothing in this module may import a provider SDK. Adapters (``llm/anthropic.py``
and friends) translate their vendor's request/response/exception shapes into the
types declared here, so the runtime depends on this file and never on a vendor.

Two things live here:

* :class:`LargeLanguageModelClient` -- the callable surface, plus the message and
  response shapes that cross it.
* Three failure types that the caller is expected to tell apart. They are
  siblings, never subclasses of one another, so ``except TransientProviderError``
  cannot silently swallow an auth or configuration failure and retry it forever.
"""

from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict, runtime_checkable

__all__ = [
    "ClientConfigurationError",
    "CompletionResponse",
    "LargeLanguageModelClient",
    "LLMClientError",
    "Message",
    "ProviderAuthError",
    "Role",
    "TransientProviderError",
    "Usage",
]


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """One turn of the conversation handed to the provider."""

    role: Role
    content: str


class Usage(TypedDict):
    """Token accounting for a single completion, for trace and budget records."""

    input_tokens: int
    output_tokens: int


class CompletionResponse(TypedDict):
    """The normalized result of one completion.

    ``model`` is the model the provider actually served (which may differ from
    the requested alias). ``stop_reason`` is the provider's raw stop label, kept
    as an opaque string so the port does not have to enumerate every vendor's
    vocabulary; ``None`` means the adapter had nothing to report.
    """

    text: str
    model: str
    stop_reason: str | None
    usage: Usage


@runtime_checkable
class LargeLanguageModelClient(Protocol):
    """What the runtime is allowed to assume about any LLM provider.

    Implementations are synchronous and stateless per call: the same client may
    be reused across requests, and ``complete`` must not retain the messages it
    was given. Retry is a runtime concern and lives above this port -- an adapter
    reports a failure with the right type and returns.
    """

    model_name: str
    """The model this client is bound to. Recorded in the trace for every call."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
    ) -> CompletionResponse:
        """Run one completion and return its normalized result.

        Raises:
            ClientConfigurationError: Required local configuration is missing, so
                no request was attempted.
            ProviderAuthError: The provider rejected the credential or the
                request itself; retrying it unchanged cannot succeed.
            TransientProviderError: The call failed in a way that a later attempt
                may survive (timeout, connection reset, 429, 5xx).
        """
        ...


class LLMClientError(Exception):
    """Base for every failure raised through the provider port.

    Present so a caller that only wants to know "the model call failed" has one
    thing to catch. Callers that act on the difference must catch the concrete
    types below, which are siblings of each other, never ancestors.
    """


class TransientProviderError(LLMClientError):
    """A transport-level failure that a later identical attempt may survive.

    Timeouts, connection resets, HTTP 429, HTTP 5xx. This is the only one of the
    three the retry engine is allowed to act on.
    """


class ProviderAuthError(LLMClientError):
    """The provider reached us and refused: bad or revoked key, malformed request.

    A network round trip happened and produced a definitive rejection, so
    retrying the same call unchanged only burns budget.
    """


class ClientConfigurationError(LLMClientError):
    """Required local configuration is missing before any network call is made.

    A missing API key or an unset model is not the provider rejecting anything --
    nothing was ever sent -- so it is deliberately not a
    :class:`ProviderAuthError`. It is a startup bug, and the fix is local.
    """
