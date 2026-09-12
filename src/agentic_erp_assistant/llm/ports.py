"""The typed provider port: the contract every real LLM call goes through.

Nothing in this module may import a provider SDK. Adapters (``llm/adapters/``,
e.g. ``openai_chat.py``) translate their vendor's request/response/exception
shapes into the types declared here, so the runtime depends on this file and
never on a vendor.

Three things live here:

* :class:`LargeLanguageModelClient` -- the callable surface, plus the message and
  response shapes that cross it.
* :class:`ToolCallingClient` -- the second, optional surface: offering the model
  a set of tools and reading back which one it picked.
* Three failure types that the caller is expected to tell apart. They are
  siblings, never subclasses of one another, so ``except TransientProviderError``
  cannot silently swallow an auth or configuration failure and retry it forever.
"""

from collections.abc import Callable, Sequence
from typing import Literal, Protocol, TypedDict, runtime_checkable

from agentic_erp_assistant.llm.tools import ToolCallResult, ToolSpec

__all__ = [
    "ClientConfigurationError",
    "CompletionResponse",
    "LargeLanguageModelClient",
    "LLMClientError",
    "Message",
    "ProviderAuthError",
    "Role",
    "ToolCallingClient",
    "TransientProviderError",
    "Usage",
    "UsageReporting",
]


Role = Literal[
    "system",  # standing policy: who the assistant is and what it may not do
    "developer",  # the output contract: the schema the reply must satisfy
    "user",  # the human's words, verbatim
    "evidence",  # retrieved source snippets: data to read, never instructions
    "observation",  # what this turn's own tool calls returned: data, not orders
    "history",  # this session's recent turns: a record of words, never a source
    "memory",  # what earlier turns established: background, never a citation
    "assistant",  # a prior reply from the model
]
"""Who a message is speaking as.

``developer`` and ``evidence`` are separate roles rather than text folded into
``system`` and ``user`` because the separation is a security boundary: an
instruction sitting inside a retrieved document must stay distinguishable from
one the user actually typed. Collapsing them into one string throws that
distinction away at construction, and no later guardrail can recover it.

``observation`` is the same boundary drawn around a different source. It is what
this turn's own tool calls returned -- a risk title someone typed into the ERP
last quarter is still text a person wrote, and a reason-act loop feeds it
straight back into the next decision. Kept apart from ``evidence`` as well as
from ``assistant``: it is not a citable passage with a locator, and it is
certainly not something the model said.

``history`` is a bounded, verbatim window of this session's own recent turns --
the user's own words and this assistant's own prior replies, kept so a
follow-up ("and the second one?") has an antecedent. It is not ``memory``: a
memory is judged by ``memory/policy.py`` before it is ever written, while
history is not judged at all -- it is this actor's own conversation, retained
because trimming it would misquote them, not because a policy accepted it.

``memory`` is the next data role, and the one carrying text this system wrote
down itself in an earlier turn, after a policy decided it was durable. It is
separate from ``evidence`` for two reasons that pull the same way. A memory has
no locator, so nothing in it may ever be cited -- folded into the evidence
block it would look exactly like a passage that could be, and the model would
eventually cite one. And a memory is older than ``history``: it survived past
the turns that produced it, while history is still those turns themselves, so
it must lose to a live tool result and to history alike rather than sit beside
either as an equal claim.

The order of precedence a model should give these, strongest to weakest:
policy, the user's current words, evidence, observations, history, memory.
"""


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
        on_delta: Callable[[str], None] | None = None,
    ) -> CompletionResponse:
        """Run one completion and return its normalized result.

        No vendor accepts all five of :data:`Role` directly, so each adapter has
        to collapse ``developer`` and ``evidence`` into its own wire shape --
        the Anthropic adapter, for instance, folds ``developer`` into the
        ``system`` parameter and sends ``evidence`` as its own delimited block.
        **Evidence must survive that translation as a distinct block, never
        concatenated into the user turn.** An adapter that joins them has undone
        the separation the roles exist to provide, and the prompt module's
        injection boundary becomes decorative.

        Args:
            messages: The prompt, by role.
            temperature: Passed through to the provider unchanged.
            on_delta: When given, the adapter *may* stream: it calls this with
                each fragment of reply content, in order, as the provider
                sends it, and still returns the complete, normalized result
                once the call finishes. Never called with tool-call
                arguments -- there are none on this path. An adapter that
                cannot stream is free to ignore it entirely; every existing
                fake remains a valid client precisely because this is
                optional and advisory, never a second contract to satisfy.

        Raises:
            ClientConfigurationError: Required local configuration is missing, so
                no request was attempted.
            ProviderAuthError: The provider rejected the credential or the
                request itself; retrying it unchanged cannot succeed.
            TransientProviderError: The call failed in a way that a later attempt
                may survive (timeout, connection reset, 429, 5xx).
        """
        ...


@runtime_checkable
class ToolCallingClient(Protocol):
    """A client that can also be asked which tool to use.

    Separate from :class:`LargeLanguageModelClient` rather than a method added
    to it, and the precedent is :class:`UsageReporting` two definitions below:
    an adapter that can only produce text is still a valid provider, and every
    test fake would otherwise have to implement function calling to be one.
    Widening the base port would make tool calling mandatory for implementations
    that will never route anything.

    A client may satisfy both, and the real adapter does. The gateway asks for
    this one only on the path that needs it, so the requirement shows up where
    the capability is used instead of at construction of everything.
    """

    model_name: str
    """The model this client is bound to. Recorded in the trace for every call."""

    def call_with_tools(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec],
        temperature: float,
        on_delta: Callable[[str], None] | None = None,
        allow_tools: bool = True,
    ) -> ToolCallResult:
        """Offer ``tools`` and report the single choice that came back.

        A different question from :meth:`LargeLanguageModelClient.complete`, and
        so a different method rather than a flag on that one: this call asks
        what to do next, that one asks for a grounded answer against a schema,
        and a request carrying both contracts would have the model satisfy
        whichever it preferred.

        At most one call comes back. An adapter that receives several must not
        pick one and discard the rest -- approval routing depends on knowing
        exactly what is about to run, so several is a provider contract
        violation and belongs in the transient-failure path.

        ``allow_tools=False`` still passes ``tools`` -- the model may need the
        definitions to understand what its own prior calls in ``observations``
        were -- but forces the wire's ``tool_choice`` to ``"none"``, so the
        result is guaranteed content. This is how a planner call is made after
        a mutating tool has already succeeded (ADR 0019): the engine ends the
        turn by taking the option to call another tool away, rather than by
        asking the model in prose not to.

        Args:
            messages: The same port-role messages ``complete`` takes.
            tools: What to offer. Never empty -- offering nothing while asking
                the model to choose is a caller bug.
            temperature: A routing decision is not a place for variety; callers
                pass 0.0.
            on_delta: The same contract as ``complete``'s -- reply *content*
                fragments only, in order, never tool-call arguments. A model
                choosing a tool typically streams no content at all, so
                ``on_delta`` may simply never be called on that path; a model
                answering directly (no tool chosen) may stream normally.
            allow_tools: ``False`` sends ``tool_choice: "none"`` on the wire,
                so the result is always content -- see above.

        Returns:
            A :class:`~agentic_erp_assistant.llm.tools.ToolCallResult`: a tool
            name with parsed arguments, or direct content. Never both.

        Raises:
            TransientProviderError: The transport failed, or the reply cannot be
                read as a decision.
            ProviderAuthError: A definitive rejection from the provider.
        """
        ...


@runtime_checkable
class UsageReporting(Protocol):
    """A client that remembers what the provider said the last call cost.

    Separate from :class:`LargeLanguageModelClient` because it is only needed on
    one path. A successful call reports its usage in the response it returns; a
    call that never produced a response -- retries exhausted, a reply rejected
    before it was read -- reports nothing, and yet the attempts were billed.
    A caller that wants to record that spend asks for this protocol and gets a
    typed answer instead of reaching for ``getattr``.

    Optional on purpose: a client that cannot report it is still a valid client.
    """

    last_usage: dict[str, object] | None
    """The provider's own usage object from the most recent call that reported
    one, in the provider's own key names. ``None`` before the first call."""


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
