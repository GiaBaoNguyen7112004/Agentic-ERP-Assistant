"""The only four things outside itself the graph is allowed to depend on.

A retriever, a way to run a tool, something that decides what to do next, and
something that turns evidence into a grounded answer. Everything else a turn
needs is already in
:class:`~agentic_erp_assistant.state.agent_state.AgentState`. Keeping all four
declarations in one file is the point of the file: the workflow's entire
external surface is four protocols on one screen, so "what does the graph
depend on?" is answered by reading rather than by grepping imports across a
package.

Two of them are the model, and they are separate for a reason
-------------------------------------------------------------

:class:`PlannerPort` decides; :class:`AnswerComposerPort` writes the grounded
reply. One object may satisfy both, and in this project the second is satisfied
by :class:`~agentic_erp_assistant.llm.gateway.LLMGateway` exactly as it already
stands. They are declared apart because they are implemented apart -- the
planner is a policy layer over a tool-calling call, the composer is a
schema-validated answering call -- and a node that only answers should not have
to be handed something that can also route.

Protocols, so implementations never import this module
------------------------------------------------------

Both are :class:`typing.Protocol`. Conformance is structural -- a retriever
satisfies :class:`DocumentRetrieverPort` by having the method, not by
inheriting from it -- which means ``rag/`` and ``tools/`` will implement these
without importing anything from ``engine/``. That is what keeps the dependency
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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentic_erp_assistant.reasoning.decision import ReasoningDecision
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.state.tool_request import ToolRequest

if TYPE_CHECKING:  # pragma: no cover - names a contract without depending on it
    # The answer contract is declared in llm/schemas.py, and the composer port
    # has to name it in a signature. Imported under TYPE_CHECKING so a type
    # checker sees the real class while this package still pulls in no part of
    # llm/ at import time -- importing it for real would drag the gateway, the
    # tokenizer and an HTTP client behind it into every node and every runtime
    # test, for a name that is only ever read off the returned object.
    from agentic_erp_assistant.llm.schemas import GroundedAnswer

__all__ = [
    "AnswerComposerPort",
    "DocumentRetrieverPort",
    "PlannerPort",
    "ToolGatewayPort",
    "ToolOutcome",
    "ToolRequest",
]

# ToolRequest and ToolOutcome are re-exported, not defined here. They are the
# two halves of execute()'s signature, so this module has to name them -- but
# the tool layer has to build both, and definitions living in the port would
# drag engine/ into every implementation. Declaring the gateway as a Protocol
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

    def preflight(self, request: ToolRequest) -> ToolOutcome:
        """Every check that precedes execution, and nothing that is execution.

        Exists so a write can be put to a human only after the checks that
        would refuse it anyway have already passed (ADR 0016): the tool
        exists, its arguments are valid, the actor holds its scope, any
        project the call names matches the actor's, and there is budget left
        -- with nothing counted and no handler reached. The answer a caller
        wants is the status: ``"approval_required"`` means the call may be
        put to a human; anything else is the refusal that human would
        otherwise have been asked to rule on.

        Args:
            request: The call as it would be made, with the actor's scopes
                and project -- ``approval`` is expected not yet
                ``"approved"``, since this is called before a human has seen
                the call.

        Returns:
            A :class:`ToolOutcome`: a refusal (``"failed"``,
            ``"invalid_arguments"``, ``"denied"``, ``"rate_limited"``) or
            ``"approval_required"`` when every check passes.

        Raises:
            ValueError: ``request`` names a tool that needs no approval, or
                one that is already approved -- neither has an honest
                outcome for this method to return.
        """
        ...


@runtime_checkable
class PlannerPort(Protocol):
    """What the graph assumes about whatever decides the next action.

    One method, and it returns a
    :class:`~agentic_erp_assistant.reasoning.decision.ReasoningDecision` rather
    than a provider reply. That boundary is the reason this port exists at all:
    reading a tool call and deciding that a mutating tool means
    ``request_approval`` is routing policy, and policy inside a graph node is
    policy nobody can test without a fake provider speaking wire JSON. Here, a
    routing test scripts one typed decision.

    The runtime therefore never sees a prompt, a tool schema or a token count.
    It sees a route, and a rationale it writes into the trace.
    """

    def plan(self, state: AgentState) -> ReasoningDecision:
        """Decide what this turn should do next, from everything it knows.

        Takes the whole state, not a question plus a history. The planner reads
        the request, the evidence gathered so far and the observations already
        produced, and a signature that passed those separately would be one a
        caller could assemble inconsistently -- which is how a loop ends up
        deciding on a stale view and repeating a call it has already made.

        Returns:
            A decision the runtime will still check against the transition
            table before acting on it. A planner is not trusted to know the
            graph.

        Raises:
            Exception: Whatever the underlying model call raises. The runtime
                turns a failure here into a routed, traced ``fail``; a planner
                is not asked to classify its own outage.
        """
        ...


@runtime_checkable
class AnswerComposerPort(Protocol):
    """What the graph assumes about writing a grounded reply.

    Named here rather than typed as the concrete gateway, so ``engine/`` keeps
    depending on shapes. It adds no mechanism: the implementation this project
    ships is
    :meth:`~agentic_erp_assistant.llm.gateway.LLMGateway.answer`, unchanged and
    satisfying this structurally.
    """

    def answer(
        self,
        question: str,
        evidence: Sequence[EvidenceSnippet],
        memories: Sequence[MemoryRecord] = (),
        history: Sequence[ConversationTurn] = (),
    ) -> "GroundedAnswer":
        """Answer ``question`` from ``evidence``, or refuse in a typed way.

        The return type is the part that matters: an answer either carries
        citations or says why it refused. The node that calls this still checks
        every citation against the evidence actually retrieved before the reply
        reaches anyone -- a composer is trusted to write, not to have cited
        something real.

        ``memories`` and ``history`` shape *how* the reply reads and never what
        it asserts. Both are safe to hand over for a structural reason rather
        than a hopeful one: neither carries a locator, so there is nothing in
        either a citation could be built from, and the caller's check against
        the retrieved evidence catches one that was invented anyway.

        Raises:
            Exception: Provider failures and contract violations propagate. The
                node turns them into a routed ``fail`` with the reason traced.
        """
        ...
