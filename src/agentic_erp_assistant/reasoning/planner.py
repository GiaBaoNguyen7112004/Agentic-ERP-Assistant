"""One function call in, one typed decision out.

This is the whole of the reason-act loop's "reason" step, and it is deliberately
thin. The model is offered every action the turn could take as a function
(:data:`~agentic_erp_assistant.llm.tools.PLANNING_TOOLS`) and calls exactly one;
this module reads that call and says which route it is. Nothing here talks to a
provider, and nothing here decides *what* to offer -- the gateway owns the call,
the tool list owns the offering, and what is left is the mapping.

Why the mapping is not in the gateway or in a node
--------------------------------------------------

It is policy. "A tool whose ``mutating`` flag is set routes to
``request_approval`` rather than ``call_tool``" is the approval rule stated once,
and it has to sit somewhere a test can reach without a provider and somewhere a
node cannot quietly disagree with it. In the gateway it would be routing decided
in the transport layer; in a node it would be untestable without a fake speaking
wire JSON. Here it is a pure function of a
:class:`~agentic_erp_assistant.llm.tools.ToolCallResult`.

What the rationale is, and what it is not
-----------------------------------------

The rationale this module writes states what was chosen -- "called list_risks"
-- and is not a thought the model reported having. That follows the rule
:mod:`agentic_erp_assistant.reasoning.decision` already sets: each route is
produced by an independently checkable mechanism, never by a self-report. A
model asked to narrate its reasoning produces text that reads like evidence of a
decision process and is not, and putting that in the trace would make the audit
worse while making it look better.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from agentic_erp_assistant.llm.ports import ToolChoice
from agentic_erp_assistant.llm.tools import (
    PLANNING_TOOLS,
    ToolCallResult,
    ToolSpec,
)
from agentic_erp_assistant.reasoning.decision import DecisionRoute, ReasoningDecision
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.memory import MemoryRecord
from agentic_erp_assistant.state.tool_outcome import ToolOutcome

__all__ = [
    "BLANK_REQUEST_QUESTION",
    "CONTROL_ROUTES",
    "DecisionModel",
    "Planner",
    "UNSCORED_CONFIDENCE",
]


CONTROL_ROUTES: Mapping[str, DecisionRoute] = {
    "search_project_documents": "retrieve_project_documents",
    "ask_clarification": "clarify",
    "refuse": "refuse",
}
"""The three offered functions the graph carries out itself, and their routes.

A table rather than a chain of ``if name ==`` branches, for the reason the
transition table is one: adding a control function is then an entry, and a
function offered without an entry falls through to the tool path, where the
registry has no handler for it and the failure is loud.
"""


UNSCORED_CONFIDENCE = 0.5
"""What goes in ``confidence`` when nothing measured it.

Chat Completions returns no calibrated probability for a function choice, so any
number here is invented. Rather than invent a confident-looking one, every
model-made decision records the same value, and the field means "unscored" for
this planner. Nothing branches on it today; when something does it will need a
real signal -- an agreement check, a logprob, a second sample -- and this
constant is where that absence stays visible instead of hiding behind a
plausible 0.9.
"""


BLANK_REQUEST_QUESTION = "What would you like to know about the project?"
"""What an empty request is answered with, without spending a model call."""


@runtime_checkable
class DecisionModel(Protocol):
    """The one thing a planner needs from the model layer.

    Satisfied by :meth:`~agentic_erp_assistant.llm.gateway.LLMGateway.decide`.
    Narrower than that class on purpose: a planner has no business answering,
    pricing or counting tokens, and a port that named the whole gateway would
    let it do all three.
    """

    def decide(
        self,
        question: str,
        evidence: Sequence[EvidenceSnippet] = (),
        observations: Sequence[ToolOutcome] = (),
        memories: Sequence[MemoryRecord] = (),
        history: Sequence[ConversationTurn] = (),
        *,
        tools: Sequence[ToolSpec] = ...,
        tool_choice: ToolChoice = "auto",
    ) -> ToolCallResult:
        """Offer ``tools`` for this turn's state and return the one choice made.

        ``tool_choice="none"`` forces the wire's ``tool_choice`` to ``"none"``
        (see
        :meth:`~agentic_erp_assistant.llm.ports.ToolCallingClient.call_with_tools`),
        so the result is guaranteed content; ``"required"`` forces a call
        back. Both are the mechanism :meth:`Planner.plan`'s ``tool_choice``
        parameter drives.
        """
        ...


@dataclass(frozen=True)
class Planner:
    """Turns the model's chosen function into the route the graph will take.

    Satisfies :class:`~agentic_erp_assistant.engine.ports.PlannerPort`
    structurally, without importing it -- which is what lets ``engine/`` depend
    on the protocol while this module stays in the decision layer.
    """

    model: DecisionModel
    """Where the choice comes from."""

    tools: tuple[ToolSpec, ...] = PLANNING_TOOLS
    """What to offer. Injectable so a test can narrow the menu, and so the
    offered set stays data rather than a constant read at the call site."""

    _by_name: dict[str, ToolSpec] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", {spec.name: spec for spec in self.tools})

    def plan(
        self,
        state: AgentState,
        *,
        tool_choice: ToolChoice = "auto",
        withhold: frozenset[str] = frozenset(),
    ) -> ReasoningDecision:
        """Decide the next action for ``state``.

        Args:
            state: The turn as it stands. Its request, evidence, observations,
                recalled memories and recent history are what the model is
                shown -- all five read off the one object, so a routing
                decision can never be made against a view somebody assembled
                inconsistently.
            tool_choice: ``"auto"`` (default) lets the model pick a tool or
                answer. ``"none"`` still shows the model every offered tool's
                definition (so it can still make sense of what its own prior
                calls in ``observations`` returned) but forces the call to
                come back as content. Used by ``engine/nodes.py::think`` (ADR
                0019) to end a turn's planning loop after a mutating tool has
                already succeeded, or after a call has already been repeated
                once -- by taking the option to call another tool away, not
                by asking in prose. ``"required"`` forces a call back; used
                by the same node (ADR 0021) to force a choice among what
                remains offered after ``withhold`` removes a tool the
                declared reply contract has already been satisfied without.
            withhold: Tool names to leave out of what is offered this call --
                distinct from ``tool_choice="none"``, which still offers
                everything and only changes what the model may *do* with it.
                A name here is never sent to the model, so a call naming one
                anyway is a provider contract violation the real API cannot
                produce.

        Returns:
            A :class:`ReasoningDecision`. Every path returns one -- a choice
            that cannot be acted on becomes a ``fail`` decision rather than an
            exception, because "the model named a tool that does not exist" is a
            turn that has to be reported, not a crash. A tool call surviving
            ``tool_choice="none"``, prose surviving ``tool_choice="required"``,
            and a call naming a withheld tool are all exactly as unreadable:
            the real provider cannot produce any of the three, so seeing one
            here means whatever is standing in for it did not honor the
            request.
        """
        if not state.request.strip():
            # Short-circuited before the model is called. A blank question
            # cannot be routed, and asking a provider what to do with one spends
            # money to be told so.
            return ReasoningDecision(
                route="clarify",
                confidence=UNSCORED_CONFIDENCE,
                message=BLANK_REQUEST_QUESTION,
                rationale="the request was empty, so nothing was sent",
            )

        offered = tuple(spec for spec in self.tools if spec.name not in withhold)

        result = self.model.decide(
            state.request,
            state.evidence,
            state.observations,
            state.memories,
            state.history,
            tools=offered,
            tool_choice=tool_choice,
        )

        if result.tool_name is None:
            if tool_choice == "required":
                return self._unreadable(
                    "the model answered in prose after a call was required"
                )
            # No call: the model elected to answer. ToolCallResult guarantees
            # content is present in that case, so there is nothing to check.
            return ReasoningDecision(
                route="answer",
                confidence=UNSCORED_CONFIDENCE,
                message=result.content,
                rationale="answered without calling a tool",
            )

        if tool_choice == "none":
            return self._unreadable(
                f"the model called {result.tool_name!r} after tools were "
                f"withheld for this call"
            )

        if result.tool_name in withhold:
            return self._unreadable(
                f"the model called {result.tool_name!r}, which was withheld "
                f"for this call"
            )

        spec = self._by_name.get(result.tool_name)
        if spec is None:
            return self._unreadable(
                f"the model called {result.tool_name!r}, which was not offered"
            )

        if spec.name in CONTROL_ROUTES:
            return self._control(spec, result.arguments or {})
        return self._tool_call(spec, result.arguments or {})

    # -- the three shapes a choice can take --------------------------------

    def _control(
        self, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> ReasoningDecision:
        """A function the graph carries out itself: search, clarify, refuse.

        These arguments are validated here, unlike a tool call's. The graph
        reads them directly -- they become the query it searches for and the
        words the user is shown -- so a missing one would surface as an
        attribute error inside a node. A tool call's arguments are left alone
        because the tool gateway validates them against the same spec and
        records ``invalid_arguments`` in the audit, which is the better place
        for it.
        """
        try:
            validated = spec.validate_arguments(arguments)
        except ValidationError as error:
            return self._unreadable(
                f"{spec.name} was called with arguments it does not accept: "
                f"{error.error_count()} problem(s)"
            )

        route = CONTROL_ROUTES[spec.name]
        if route == "retrieve_project_documents":
            return ReasoningDecision(
                route=route,
                confidence=UNSCORED_CONFIDENCE,
                search_query=validated.query,
                rationale=f"called {spec.name}",
            )
        return ReasoningDecision(
            route=route,
            confidence=UNSCORED_CONFIDENCE,
            message=validated.question if route == "clarify" else validated.reason,
            rationale=f"called {spec.name}",
        )

    def _tool_call(
        self, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> ReasoningDecision:
        """A tool the gateway will run -- once, and only after any approval.

        The approval rule, stated once: a tool whose own ``mutating`` flag is
        set never routes straight to execution. The flag is read off the
        declaration, never guessed from the name and never taken from the model,
        which is the difference between an approval gate and a suggestion.

        Policy can also escalate a read to an approver, and that decision lives
        in the tool registry, which this layer does not import. Such a call
        routes to ``call_tool`` here and is refused at the gateway with
        ``approval_required``; the node reads that status and re-routes to a
        human. The gate holds either way, and the registry stays the single
        authority on it.
        """
        return ReasoningDecision(
            route="request_approval" if spec.mutating else "call_tool",
            confidence=UNSCORED_CONFIDENCE,
            required_tool=spec.name,
            tool_arguments=arguments,
            mutating=spec.mutating,
            approval_required=spec.mutating,
            rationale=f"called {spec.name}",
        )

    @staticmethod
    def _unreadable(why: str) -> ReasoningDecision:
        """A choice that cannot be acted on: routed to fail, with the reason.

        Not raised. The turn still has to end somewhere the trace can name, and
        a broken decision is exactly what ``fail`` is for -- see
        :data:`~agentic_erp_assistant.reasoning.decision.DecisionRoute`.
        """
        return ReasoningDecision(
            route="fail",
            confidence=UNSCORED_CONFIDENCE,
            rationale=why,
        )
