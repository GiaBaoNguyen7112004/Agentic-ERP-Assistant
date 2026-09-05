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

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.state.tool_request import ToolRequest

__all__ = ["DocumentRetrieverPort", "ToolGatewayPort", "ToolOutcome", "ToolRequest"]

# ToolRequest and ToolOutcome are re-exported, not defined here. They are the
# two halves of execute()'s signature, so this module has to name them -- but
# the tool layer has to build both, and definitions living in the port would
# drag runtime/ into every implementation. Declaring the gateway as a Protocol
# exists precisely to stop that, so the types sit in state/, which both sides
# may depend on.


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


@runtime_checkable
class ToolGatewayPort(Protocol):
    """What the graph assumes about the boundary where a tool actually runs.

    The gateway is the last thing between a decision and a change to ERP data,
    which is why the request carries the approval and the actor's scopes
    instead of the gateway trusting that someone upstream checked. A boundary
    that cannot see whether a call was permitted and approved has to trust
    every caller that will ever exist, and the one that forgets is the one that
    matters.
    """

    def execute(self, request: ToolRequest) -> ToolOutcome:
        """Run one tool call and report what happened.

        One object in, one object out. The request carries the actor, the
        scopes they hold and where the call stands with an approver, and none
        of that is optional: a mutating tool must refuse to run unless
        ``request.approval`` is ``"approved"``, and a call must be refused
        outright unless the actor holds the tool's declared scope. Whether a
        tool mutates is read from its own registry entry, never guessed from
        the name here.

        Loose arguments were the alternative, and they are worse in a specific
        way: the audit facts would each be one more parameter a caller could
        omit, and the one that gets omitted is a scope.

        A refusal comes back as a :class:`ToolOutcome` carrying ``"denied"``,
        and a call nobody has approved yet as ``"approval_required"`` -- not as
        exceptions; see that class for why.

        Args:
            request: The call to make, with everything a check will consult.

        Returns:
            A :class:`ToolOutcome` describing success or failure.
        """
        ...
