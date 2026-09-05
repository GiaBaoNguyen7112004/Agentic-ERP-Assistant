"""The only two things outside itself the graph is allowed to depend on.

A retriever, and a way to run a tool. Everything else a turn needs is already
in :class:`~agentic_erp_assistant.state.agent_state.AgentState`. Keeping both
declarations in one file is the point of the file: the workflow's entire
external surface is two protocols on one screen, so "what does the graph
depend on?" is answered by reading rather than by grepping imports across a
package.

Protocols, so implementations never import this module
------------------------------------------------------

Both are :class:`typing.Protocol`. Conformance is structural -- a retriever
satisfies :class:`DocumentRetrieverPort` by having the method, not by
inheriting from it -- which means ``rag/`` and ``tools/`` will implement these
without importing anything from ``runtime/``. That is what keeps the dependency
direction pointing inward: the graph depends on a shape, the shape depends on
nothing, and no outer layer is dragged in behind it.

The consequence for tests is the one that matters day to day: a fake retriever
is a ten-line class with a ``search`` method, and no node has ever seen a
concrete retriever class to be coupled to.

The ports lead; the implementations follow
------------------------------------------

These signatures are not transcribed from an existing retriever or gateway.
They say what the runtime needs, and an implementation that does not match
gets an adapter -- the same arrangement
:mod:`agentic_erp_assistant.llm.ports` already has with its vendor adapters.
Writing the port to match whatever a backend happens to expose puts that
backend's shape into the core, which is the coupling the port exists to avoid.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_erp_assistant.state.agent_state import ApprovalDecision
from agentic_erp_assistant.state.evidence import EvidenceSnippet

__all__ = ["DocumentRetrieverPort", "ToolGatewayPort", "ToolOutcome"]


@runtime_checkable
class DocumentRetrieverPort(Protocol):
    """What the graph assumes about any way of finding project passages.

    Lexical, vector or hybrid: the runtime does not know and must not care.
    It asks for passages and gets back objects that carry their own citation.
    """

    def search(self, query: str, *, limit: int) -> Sequence[EvidenceSnippet]:
        """Return the passages most worth reading for ``query``, best first.

        ``limit`` is keyword-only and has no default, deliberately. How much
        evidence to pull is a context-budget decision the runtime makes and
        records; a default living inside a retriever would change the size of
        every prompt from a place no reviewer thinks to look, and swapping the
        backend would silently change it again.

        Ordering is part of the contract -- the caller truncates from the end
        when the budget is tight, so a retriever that returns its results
        unranked has made that truncation arbitrary. It is also why no score
        comes back: rank order is the only thing the runtime reads, and a
        number nothing branches on would be telemetry riding inside the object
        that goes into the prompt.

        Returning fewer than ``limit`` results, including none, is a normal
        answer and not an error: "nothing was found" is a routed outcome
        (``insufficient_evidence``), and an exception would turn it into a
        crash the trace records as a fault.
        """
        ...


class ToolOutcome(BaseModel):
    """What running one tool produced -- success or failure, both as data.

    A failed tool call is not an exception here. A tool that was denied,
    rejected its arguments or timed out is something the turn has to record and
    then route on, and an exception thrown through the graph would leave the
    state that describes the attempt unwritten. So the gateway returns this
    either way, and the node decides what it means.

    Frozen and ``extra="forbid"``, like every other object that crosses a node
    boundary: what the trace says the tool returned has to be what the model
    was then shown.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1)
    """Which tool ran. Repeated here rather than inferred from context so a
    trace entry stands on its own."""

    ok: bool
    """Whether the call did what it was asked to do."""

    output: str = ""
    """What to feed back to the model. Empty is legitimate for a failure."""

    error: str | None = None
    """Why it failed, in one line, for a person reading the trace."""

    @model_validator(mode="after")
    def _an_outcome_reports_exactly_one_thing(self) -> "ToolOutcome":
        if self.ok and self.error is not None:
            raise ValueError(
                "error: present on a successful call -- a reader cannot tell "
                "whether the call worked"
            )
        if not self.ok and not (self.error or "").strip():
            raise ValueError(
                "error: a failed call must say why, or the trace records that "
                "something went wrong and nothing about what"
            )
        return self


@runtime_checkable
class ToolGatewayPort(Protocol):
    """What the graph assumes about the boundary where a tool actually runs.

    The gateway is the last thing between a decision and a change to ERP data,
    which is why :meth:`execute` is told about the approval instead of trusting
    that someone upstream checked. A boundary that cannot see whether a call
    was approved has to trust every caller that will ever exist, and the one
    that forgets is the one that matters.
    """

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        actor: str,
        approval: ApprovalDecision,
    ) -> ToolOutcome:
        """Run one tool call and report what happened.

        ``actor`` and ``approval`` are keyword-only and required. They are not
        bookkeeping the gateway passes through to a log: a mutating tool must
        refuse to run unless ``approval`` is ``"approved"``, and ``actor``
        names who the write is performed on behalf of. Whether a tool mutates
        is read from its own
        :attr:`~agentic_erp_assistant.llm.tools.ToolSpec.mutating` flag, never
        guessed from the name here.

        A refusal comes back as a :class:`ToolOutcome` with ``ok=False``, not
        as an exception -- see that class for why.

        Args:
            tool_name: The tool to run, as declared in the registry.
            arguments: Parsed arguments. The gateway validates them against the
                tool's own schema; a caller cannot pre-approve a shape.
            actor: Who the call is made on behalf of.
            approval: Where the call stands with its approver.

        Returns:
            A :class:`ToolOutcome` describing success or failure.
        """
        ...
