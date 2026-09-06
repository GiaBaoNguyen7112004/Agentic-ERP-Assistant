"""The one place a real call to an embeddings API is made.

The sibling of :mod:`agentic_erp_assistant.llm.adapters.openai_chat`, written the
same way and for the same reasons: plain ``httpx`` rather than the vendor SDK, so
the wire shape stays visible in this file and a test can inject
``httpx.MockTransport`` and walk the entire failure matrix without a key, a
network, or a bill; no retry here, because a retry needs a budget and a trace
entry and both live above this layer; and the same three port errors, so
:func:`~agentic_erp_assistant.llm.retry.retry_with_backoff` already knows which
failures are worth repeating.

Two decisions specific to embeddings
------------------------------------

**The response is sorted by ``index`` before the vectors are returned.** The API
is not contractually required to return them in input order, and the caller pairs
vectors with chunks positionally. A silent reordering would attach every passage
to the wrong vector -- and nothing downstream could catch it, because a
mis-paired vector is still a perfectly valid vector. It would surface as
retrieval that is subtly, unexplainably bad.

**There is no default model.** ``OPENAI_EMBEDDING_MODEL`` has to be set, and a
missing one raises :class:`ClientConfigurationError` rather than picking
something. Choosing an embedding model for the operator would bill their account
for a decision they never made, and worse, the choice is baked into every stored
vector: a collection embedded with one model cannot be searched with another, so
a default here would be a default nobody could change later without re-embedding
the corpus.
"""

import logging
import os
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import httpx
from dotenv import load_dotenv

from agentic_erp_assistant.llm.ports import (
    ClientConfigurationError,
    ProviderAuthError,
    TransientProviderError,
)
from agentic_erp_assistant.rag.ports import EmbeddingBatch

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "MAX_INPUTS_PER_REQUEST",
    "OpenAIEmbeddingsClient",
]

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://api.openai.com/v1"
"""Constructor default, not an environment variable -- see the chat adapter."""

DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
"""Longer on read than the chat adapter, short on connect for the same reason.

An embeddings request carries a whole batch of chunks, so it legitimately takes
longer than a single completion; failing to reach the host at all is still
decided quickly, because that failure will not resolve itself in two minutes.
"""

MAX_INPUTS_PER_REQUEST = 2048
"""The documented ceiling on inputs in one embeddings request.

Enforced here so an oversized batch fails locally with a message naming the
problem, instead of as a 400 that the error taxonomy would classify as a
definitive rejection -- true, but unhelpful. Splitting a corpus into batches is
the caller's decision, because the caller is the one that knows what a batch
costs; see :mod:`agentic_erp_assistant.rag.ingest`.
"""

_ERROR_BODY_EXCERPT = 300


def _require_env(name: str, *, missing: type[Exception], why: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise missing(f"{name} is not set. {why}")
    return value


class OpenAIEmbeddingsClient:
    """An :class:`~agentic_erp_assistant.rag.ports.EmbeddingsPort` over OpenAI.

    Structural conformance only -- it does not inherit from the protocol, and a
    test asserts ``isinstance`` against the runtime-checkable Protocol instead,
    so the adapter knows about the port and the port knows nothing about OpenAI.

    Configuration is read at construction, so a misconfigured deployment fails at
    startup rather than on the first question. The two missing-configuration
    cases raise different types, exactly as the chat adapter does: no key is a
    credential problem (:class:`ProviderAuthError`), no model is an incomplete
    local file (:class:`ClientConfigurationError`).

    Usage::

        with OpenAIEmbeddingsClient() as client:
            batch = client.embed([chunk.text for chunk in chunks])
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Bind the client to a key and an embedding model.

        Args:
            api_key: Overrides ``OPENAI_API_KEY``.
            model: Overrides ``OPENAI_EMBEDDING_MODEL``.
            base_url: The API root.
            timeout: Passed straight to ``httpx``.
            transport: A transport to run requests through --
                ``httpx.MockTransport`` in tests. The client is still built here,
                so closing it is still this object's job.
            http_client: A client owned by the caller, and not closed by
                :meth:`close`.

        Raises:
            ProviderAuthError: No API key, in the argument or the environment.
            ClientConfigurationError: No embedding model.
            ValueError: Both ``transport`` and ``http_client`` were given.
        """
        if transport is not None and http_client is not None:
            raise ValueError(
                "pass transport or http_client, not both: the transport would "
                "be ignored"
            )

        # Only touched when something has to come from the environment, so a
        # fully-injected client never reads the developer's real .env -- which is
        # what makes the failure-path tests trustworthy.
        if api_key is None or model is None:
            load_dotenv(override=False)

        self._api_key = api_key or _require_env(
            "OPENAI_API_KEY",
            missing=ProviderAuthError,
            why="Put it in .env (see .env.example); it is never read from code.",
        )
        self.model_name = model or _require_env(
            "OPENAI_EMBEDDING_MODEL",
            missing=ClientConfigurationError,
            why=(
                "This project chooses no embedding model for you, and the choice "
                "is baked into every stored vector: a collection embedded with "
                "one model cannot be searched with another."
            ),
        )

        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=base_url, timeout=timeout, transport=transport
        )

        self.last_usage: dict[str, Any] | None = None
        """The provider's own ``usage`` block from the most recent call that
        reported one, verbatim. ``None`` before the first call."""

    def __enter__(self) -> "OpenAIEmbeddingsClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the connection pool, if this object opened it."""
        if self._owns_client:
            self._http.close()

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed ``texts`` in a single request.

        One request, always. Batching is a real cost and latency decision -- one
        round trip and one rate-limit slot for N chunks instead of N of each --
        and it is made by the caller, which is why this method does not silently
        split an oversized list into several billable requests.

        Args:
            texts: What to embed, in the order the caller will pair them back up.
                An empty sequence returns an empty batch without calling the API;
                paying for a request that asks nothing is the one avoidable cost
                in this file.

        Returns:
            An :class:`EmbeddingBatch` whose vectors are in input order.

        Raises:
            ValueError: A text is blank, or there are more than
                :data:`MAX_INPUTS_PER_REQUEST` of them.
            TransientProviderError: Timeout, transport failure, 429, 5xx, or a
                reply this code cannot read as an embeddings response.
            ProviderAuthError: 401/403, or any other 4xx.
        """
        if not texts:
            return EmbeddingBatch(vectors=(), model=self.model_name, prompt_tokens=0)

        if len(texts) > MAX_INPUTS_PER_REQUEST:
            raise ValueError(
                f"{len(texts)} inputs exceeds the {MAX_INPUTS_PER_REQUEST} an "
                f"embeddings request may carry; split the batch at the call site, "
                f"where what a batch costs is known"
            )

        blank = [index for index, text in enumerate(texts) if not text.strip()]
        if blank:
            raise ValueError(
                f"inputs at {blank} are blank; the API rejects an empty string, "
                f"and a chunk with no text should never have been created"
            )

        body = self._send({"model": self.model_name, "input": list(texts)})
        vectors = self._read_vectors(body, expected=len(texts))
        return EmbeddingBatch(
            vectors=vectors,
            model=body.get("model") or self.model_name,
            prompt_tokens=self._read_prompt_tokens(body),
        )

    def _send(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Do the round trip: send, classify the status, read the envelope."""
        logger.debug(
            "embeddings request: model=%s inputs=%d",
            self.model_name,
            len(payload["input"]),
        )

        try:
            response = self._http.post(
                "/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException as error:
            raise TransientProviderError(
                f"timed out embedding with {self.model_name}: {error!r}"
            ) from error
        except httpx.TransportError as error:
            raise TransientProviderError(
                f"transport failure embedding with {self.model_name}: {error!r}"
            ) from error

        self._raise_for_status(response)

        try:
            body = response.json()
        except ValueError as error:
            raise TransientProviderError(
                f"unreadable body from {self.model_name}: {error!r}"
            ) from error

        if not isinstance(body, dict):
            raise TransientProviderError(
                f"expected a JSON object from {self.model_name}, got "
                f"{type(body).__name__}"
            )
        return body

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Turn an HTTP error status into the right port error.

        The split is retryability, not severity, exactly as in the chat adapter:
        429 and 5xx may work later; every other 4xx will not, and the retry
        budget must not be spent proving it.
        """
        status = response.status_code
        if status < 400:
            return

        request_id = response.headers.get("x-request-id", "none")
        detail = (
            f"{status} from {self.model_name} (request id: {request_id}): "
            f"{response.text[:_ERROR_BODY_EXCERPT].strip()}"
        )
        if status == 429 or status >= 500:
            raise TransientProviderError(detail)
        raise ProviderAuthError(detail)

    def _read_vectors(
        self, body: Mapping[str, Any], *, expected: int
    ) -> tuple[tuple[float, ...], ...]:
        """Pull the vectors out, in input order, and check they are usable.

        Sorted by the ``index`` the API reports rather than trusted in arrival
        order -- see the module docstring for why that is the difference between
        working retrieval and retrieval that is quietly wrong.
        """
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise TransientProviderError(
                f"{self.model_name} returned "
                f"{len(data) if isinstance(data, list) else type(data).__name__} "
                f"embeddings for {expected} inputs"
            )

        indexed: list[tuple[int, tuple[float, ...]]] = []
        for item in data:
            if not isinstance(item, dict):
                raise TransientProviderError(
                    f"unreadable embedding from {self.model_name}: {item!r}"
                )
            index, vector = item.get("index"), item.get("embedding")
            if not isinstance(index, int) or not isinstance(vector, list) or not vector:
                raise TransientProviderError(
                    f"embedding from {self.model_name} has no usable index or "
                    f"vector: {item!r}"
                )
            indexed.append((index, tuple(float(value) for value in vector)))

        indexed.sort(key=lambda pair: pair[0])
        if [index for index, _ in indexed] != list(range(expected)):
            raise TransientProviderError(
                f"{self.model_name} returned embedding indices "
                f"{[index for index, _ in indexed]}, expected 0..{expected - 1}"
            )

        vectors = tuple(vector for _, vector in indexed)
        widths = {len(vector) for vector in vectors}
        if len(widths) != 1:
            # Vectors of two widths cannot go into one collection, and a
            # similarity computed across them would be meaningless rather than
            # merely wrong.
            raise TransientProviderError(
                f"{self.model_name} returned vectors of {len(widths)} different "
                f"widths in one batch: {sorted(widths)}"
            )
        return vectors

    def _read_prompt_tokens(self, body: Mapping[str, Any]) -> int:
        """What the provider says it billed.

        A missing usage block is a failure rather than a zero, for the reason the
        chat adapter gives: zero cost is a number the budget layer would believe,
        and believing it turns an unmeasured run into a free-looking one.
        """
        usage = body.get("usage")
        if not isinstance(usage, dict):
            raise TransientProviderError(
                f"no usage block in the reply from {self.model_name}; refusing to "
                f"record an estimate as though it were billed"
            )

        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int):
            raise TransientProviderError(
                f"unusable usage block from {self.model_name}: {usage!r}"
            )

        self.last_usage = dict(usage)
        return prompt_tokens
