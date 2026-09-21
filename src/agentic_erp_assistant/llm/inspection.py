"""What one model call actually sent and received, bounded for a live screen.

``telemetry.py`` records what a call cost; this module is the sibling that
records what it *said* -- and the two are deliberately kept apart. A
``ModelCallRecord`` is persisted, exported, and read as audited cost
evidence; the snapshots here are never persisted at all, reach a screen only
when a developer has explicitly turned prompt capture on
(``DEV_TRACE_MODEL_IO``, ``composition/settings.py``), and exist purely so
"what did the agent actually receive and produce?" (docs/trace-inspector-
plan.md's own framing) has an answer beyond a one-line ``detail`` string.

Why a second hook beside ``TelemetrySink``, not a wider ``ModelCallRecord``
--------------------------------------------------------------------------

Widening the persisted record to carry prompt text would mean every stored
run -- exported as evidence, read back by ``EvidenceQueries`` -- also carries
the evidence and history text that produced it, restated. ``TraceEvent``'s
own 280-character cap exists for exactly this reason: free text that grows
without bound turns a log into an unaudited prompt dump. This module's
snapshots travel only from a gateway call to whatever ``LLMGateway.inspector``
is bound to for the length of one request; nothing here is written down.

Why the text is clipped here, independent of the web layer's own bound
------------------------------------------------------------------------

``web/protocol.py``'s ``TEXT_MAX_CHARS``/``clip_text`` bound what reaches the
browser. This module clips independently, at a smaller boundary, because
``llm/`` must not import ``web/`` (the dependency arrow points inward,
per CLAUDE.md) and because a snapshot is worth bounding even for a caller
that never puts it on a wire at all -- ``scripts/run_turn.py``'s console
stream, say, or a future test harness holding these in memory.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from agentic_erp_assistant.llm.ports import Message
from agentic_erp_assistant.llm.telemetry import ModelCallRecord

__all__ = [
    "ModelCallInspector",
    "ModelRequestSnapshot",
    "ModelResponseSnapshot",
    "SNAPSHOT_TEXT_LIMIT",
    "snapshot_request",
    "snapshot_response",
]


SNAPSHOT_TEXT_LIMIT = 4_000
"""How much of one message's or one reply's text a snapshot may carry.
See the module docstring for why this is not ``web/protocol.py``'s own cap."""


def _clip(text: str, limit: int = SNAPSHOT_TEXT_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit]


@dataclass(frozen=True)
class ModelRequestSnapshot:
    """What one call sent -- the role blocks, what was offered, and how.

    Frozen, like every other record of something that already happened in
    this project: a snapshot edited after the call it describes would be
    evidence about a request nobody actually sent.
    """

    kind: Literal["answer", "tools"]
    """``"answer"`` for :meth:`~agentic_erp_assistant.llm.gateway.
    LLMGateway.answer`'s structured-output call (no tools offered);
    ``"tools"`` for :meth:`~agentic_erp_assistant.llm.gateway.LLMGateway.
    call_tools` (routing, a declaration, a memory proposal)."""

    messages: tuple[tuple[str, str], ...]
    """``(role, content)`` per role block, in order, content clipped to
    :data:`SNAPSHOT_TEXT_LIMIT`."""

    message_chars: tuple[int, ...]
    """Each message's real length, before clipping -- positionally aligned
    with :attr:`messages`, the same "say what was cut" rule
    ``web/protocol.py::TextOut`` follows."""

    tools: tuple[str, ...]
    """The offered tools' names. Empty for ``kind="answer"``. Names only:
    the schemas are static code, not something a live snapshot needs to
    repeat."""

    tool_choice: str | None
    """The wire's own ``tool_choice`` value (``"auto"``/``"none"``/
    ``"required"``) for a tools call, or ``None`` for ``kind="answer"``,
    which has no such parameter."""

    temperature: float


@dataclass(frozen=True)
class ModelResponseSnapshot:
    """What one call received back -- content, or a tool call, never both
    (the same rule :class:`~agentic_erp_assistant.llm.tools.ToolCallResult`
    enforces on the parsed choice itself)."""

    content: str | None
    """The raw reply text -- the structured-output JSON string for
    ``answer()``, or the model's prose for a ``tools`` call that answered
    without calling anything. Clipped to :data:`SNAPSHOT_TEXT_LIMIT`.
    ``None`` when the call named a tool instead, or never produced a reply
    (a failure -- see :func:`snapshot_response`'s callers)."""

    content_chars: int | None
    """:attr:`content`'s real length before clipping, or ``None`` to match
    a ``None`` content."""

    tool_name: str | None
    arguments: Mapping[str, Any] | None
    """Parsed, not re-validated -- the same "transport-shaped" reading
    :class:`~agentic_erp_assistant.llm.tools.ToolCallResult` gives these
    two fields."""

    stop_reason: str | None
    """The provider's own stop label, when the client reports one --
    ``answer()`` only; a tool-call result carries no such field."""


def snapshot_request(
    kind: Literal["answer", "tools"],
    messages: Sequence[Message],
    tools: Sequence[str] = (),
    tool_choice: str | None = None,
    temperature: float = 0.0,
) -> ModelRequestSnapshot:
    """Bound one call's outgoing messages for a live screen."""
    return ModelRequestSnapshot(
        kind=kind,
        messages=tuple((message["role"], _clip(message["content"])) for message in messages),
        message_chars=tuple(len(message["content"]) for message in messages),
        tools=tuple(tools),
        tool_choice=tool_choice,
        temperature=temperature,
    )


def snapshot_response(
    *,
    content: str | None = None,
    tool_name: str | None = None,
    arguments: Mapping[str, Any] | None = None,
    stop_reason: str | None = None,
) -> ModelResponseSnapshot:
    """Bound one call's reply for a live screen. Every field defaults to
    absent, since a failed call's response is built from whatever survived
    -- usually nothing."""
    return ModelResponseSnapshot(
        content=_clip(content) if content is not None else None,
        content_chars=len(content) if content is not None else None,
        tool_name=tool_name,
        arguments=dict(arguments) if arguments is not None else None,
        stop_reason=stop_reason,
    )


@runtime_checkable
class ModelCallInspector(Protocol):
    """Where a call's live I/O goes, beside its cost record.

    A protocol with one method, the same shape
    :class:`~agentic_erp_assistant.llm.telemetry.TelemetrySink` takes and for
    the same reason: the real implementation is
    ``web/stream.py::TurnStream``, and this module must not import it -- the
    dependency would point outward from the core.
    """

    def model_call(
        self,
        record: ModelCallRecord,
        request: ModelRequestSnapshot | None,
        response: ModelResponseSnapshot | None,
    ) -> None:
        """Note one call's cost record and, when the caller asked for it
        (``LLMGateway.inspect_io``), what it sent and received.

        Args:
            record: The same record :class:`~agentic_erp_assistant.llm.
                telemetry.TelemetrySink.record` was, or will be, given --
                repeated here so an inspector never has to correlate two
                separate calls to learn one call's full story.
            request: ``None`` when ``inspect_io`` is off, or when the call
                never reached the point of having a request to describe.
            response: ``None`` when ``inspect_io`` is off, or the call
                failed before anything came back.

        Must not raise: telemetry never fails a request, the same contract
        every sink in this project keeps.
        """
        ...
