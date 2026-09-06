"""What the runtime decided to do next, in fields a reviewer can audit.

A routing decision has to be checkable without reading prose. That is the whole
purpose of this module: every part of the decision that changes what happens
next lives in a typed field, and the free-text ``rationale`` is a capped summary
that no branch is allowed to read. If a reviewer has to parse English to find out
whether a call needed approval, the approval rule is not enforced anywhere -- it
is merely described.

Where each route actually comes from
-------------------------------------

:data:`DecisionRoute` is not filled in from a model self-report of intent --
there is no upstream classification step to trust for it. Each value is
produced by a different, independently-checkable mechanism:

``call_tool`` and ``request_approval``
    Read off a live decision: whether
    :class:`~agentic_erp_assistant.llm.tools.ToolCallResult` names a tool, and
    whether that tool's :attr:`~agentic_erp_assistant.llm.tools.ToolSpec.mutating`
    flag is set. The model choosing to call a tool is itself the routing
    signal (native function calling); ``mutating`` -- declared on the tool,
    never guessed by the model -- is what decides which of the two routes it
    becomes. Collapsing them would cost the distinction that matters most
    here: under one label, a mutating call could no longer tell "this needs a
    human" apart from "a human already approved it, now run the call".

``retrieve_project_documents``
    Read off the same live decision as ``call_tool``: retrieval is offered to
    the model as a tool like any other, so the model choosing it is the routing
    signal. It is a separate route rather than a tool the gateway runs because
    its observation is a different type -- typed passages carrying locators,
    which have to reach the prompt intact, not a
    :class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome` summary
    bound for an audit row. Routes are execution shapes; the decision is
    uniform either way.

``think``
    Not chosen by anyone. It is where a turn goes to decide, so the runtime
    moves onto it whenever an action has produced an observation and the next
    action is not yet known. Having it as a route rather than as an implicit
    return to ``None`` is what lets the reason-act cycle repeat while staying
    inside the transition table: every re-plan is a declared edge, and the step
    budget is what stops the cycle from being unbounded.

``clarify`` and ``refuse``
    Guardrail outcomes: low confidence, budget overflow, insufficient
    evidence -- the same signals :func:`classify_failure` takes as input.
    Neither corresponds to a tool the model could have called, which is
    exactly why they carry no ``required_tool``.

``answer``
    Everything the reply needs is already in state, so no preparatory step is
    left to take. Read off the same live decision as ``call_tool``: a
    :class:`~agentic_erp_assistant.llm.tools.ToolCallResult` that names no tool
    is the model electing to answer.

``fail``
    Not a choice anyone made: something broke, and the turn still has to end
    somewhere the trace can name. Produced together with a
    :data:`FailureMode` from :func:`classify_failure` -- the route says the
    turn ended in failure, the mode says which failure. Deliberately not
    folded into ``refuse``: a refusal is policy working as designed and a
    failure is the system not working, so a reviewer counting refusals must
    not be counting outages.

The set splits three ways. ``retrieve_project_documents``, ``call_tool`` and
``request_approval`` name a *preparatory* action -- something to do before a
reply can exist. The other four end the turn.

``answer`` is a member rather than an implied ``None`` on purpose. A ``None``
route would give "the runtime decided to reply" and "nothing has been decided
yet" the same value, and the graph has to tell those apart: the first is a node
to dispatch, the second is a turn that never routed. It and ``fail`` were both
added at the point a graph node existed to dispatch them -- a label with no
branch behind it is a lie the type system helps tell.

What this module deliberately does not check
--------------------------------------------

``required_tool`` is not validated against the tool registry. That would couple
the decision layer to :mod:`agentic_erp_assistant.llm.tools` and would turn "the
model named a tool that does not exist" into an object that cannot be
constructed -- when it should be a routed, traced, reportable failure. The
router owns that check, against ``DEFAULT_TOOLS``, where the failure has
somewhere to be recorded.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

__all__ = [
    "classify_failure",
    "DecisionRoute",
    "FailureMode",
    "RATIONALE_MAX_CHARS",
    "ReasoningDecision",
]


DecisionRoute = Literal[
    "think",                       # decide what to do next, from what is known
    "retrieve_project_documents",  # go to RAG before answering
    "call_tool",                   # a tool call that may run unattended
    "request_approval",            # a tool call that must stop for a human
    "answer",                      # nothing left to prepare; reply from state
    "clarify",                     # the request is under-specified; ask back
    "refuse",                      # do not proceed at all
    "fail",                        # the turn broke; end and report the failure
]
"""The actions the runtime knows how to take next.

A closed set: an action nobody can dispatch has to fail here, at the
decision, rather than later as a missing branch.
"""


FailureMode = Literal[
    "context_budget_exceeded",
    "provider_failure",
    "insufficient_evidence",
    "tool_failure",
    "max_steps_exceeded",
    "none",
]
"""Why a turn could not produce a grounded answer -- including "it could".

The last two are not produced by :func:`classify_failure`, and cannot be: that
function reads three signals from one attempted answer, while ``tool_failure``
is assigned by the node that watched a call come back unusable and
``max_steps_exceeded`` by the loop guard, which is not answering anything at
all. Both are still :data:`FailureMode` members rather than free text in
``error_detail`` -- a reviewer counting how runs end has to be counting typed
values, and the loop guard firing is exactly the outcome nobody wants to
discover by grepping prose.

``"none"`` is the string, not ``None``. A ``None`` return invites callers to
write ``if failure:`` and quietly collapse four outcomes into two; a Literal
that is always a member forces a caller to match a branch.
"""


RATIONALE_MAX_CHARS = 280
"""How much free text a decision may carry.

A cap rather than a style note. Without one, ``rationale`` becomes the place a
model's full chain of thought is stored, shipped into the trace, and then read
by a reviewer as if it were audited reasoning. It is a one-line summary of a
decision whose substance is already in the typed fields above it.
"""


_TOOL_ROUTES = frozenset({"call_tool", "request_approval"})
"""The routes for which naming a tool is meaningful -- in both directions."""


_RETRIEVAL_ROUTE = "retrieve_project_documents"
"""The one route that searches, and so the only one a query belongs on."""


_MESSAGE_ROUTES = frozenset({"answer", "clarify", "refuse"})
"""The routes that end the turn by saying something to the user."""


_MESSAGE_REQUIRED_ROUTES = frozenset({"clarify", "refuse"})
"""The routes that cannot be carried out without those words.

``answer`` is absent: an answer may still have to be composed from evidence
after this decision is made, so the decision is allowed not to carry one. A
clarification or a refusal has no later step to fill it in -- the decision is
the whole of it.
"""


class ReasoningDecision(BaseModel):
    """One routing decision, with its invariants enforced at construction.

    Frozen, because a decision that has been made and then handed to a
    dispatcher must not be edited on the way: an approval gate that reads
    ``approval_required`` has to be reading the value the decision was audited
    with. ``extra="forbid"`` because an unmodelled field is a routing input the
    type system never saw.

    The cross-field rules live in the validator below rather than in a check the
    dispatcher runs. A downstream check can be skipped by a second caller; a
    constructor cannot, so the invalid states simply have no instances.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: DecisionRoute
    """What to do next. Constrained to :data:`DecisionRoute`, never a string."""

    confidence: float = Field(ge=0.0, le=1.0)
    """0.0-1.0. A low value is a reason to ``clarify``, not to guess louder."""

    required_evidence: tuple[str, ...] = ()
    """The sources this decision expects to need, by identifier.

    A tuple rather than a list: a frozen model holding a mutable list is not
    frozen, it merely looks it.
    """

    required_tool: str | None = Field(default=None, min_length=1)
    """The tool to call, for the two routes where that means anything."""

    tool_arguments: Mapping[str, Any] | None = None
    """The arguments the model called the tool with, exactly as they arrived.

    Parsed, not validated. Validating them here would move the check away from
    the boundary that has to make it anyway and would turn "the model invented
    an argument" into an object that cannot be constructed, when it should be a
    routed, audited ``invalid_arguments`` outcome. The tool gateway checks them
    against the tool's own model; this field just carries them there.

    Required on the two tool routes and rejected on the rest: a call that does
    not say what it is calling with cannot be executed, and cannot be shown to
    an approver either -- which is the more important half.
    """

    approval_required: bool = False
    """Whether a human decision must be recorded before anything executes."""

    mutating: bool = False
    """Whether the named tool changes ERP data.

    Copied off the tool's own
    :attr:`~agentic_erp_assistant.llm.tools.ToolSpec.mutating` flag by whoever
    looked the tool up, and carried here because the runtime is not allowed to
    look it up itself:
    :func:`~agentic_erp_assistant.runtime.transitions.assert_transition`
    requires the answer at every transition that touches execution, and
    :mod:`agentic_erp_assistant.runtime.transitions` deliberately does not
    import the tool registry.

    Separate from ``approval_required``, and not merely a synonym for it. This
    one is about what the tool does; that one is about what policy demands --
    a read-only tool can be escalated to an approver without ever becoming a
    write. The implication only runs one way, and the validator enforces it: a
    write always needs approval.
    """

    search_query: str | None = Field(default=None, min_length=1)
    """What to search for, on the one route that searches.

    The model picks retrieval by calling a function with a query argument, and
    that query is usually a better one than the raw request -- it is the part
    of the question that has to be looked up. Carried in its own field rather
    than in ``required_tool``/arguments because retrieval is not executed by
    the tool gateway: it has no registry entry, and a name plus a loose
    argument bag here would imply it did.

    Required on ``retrieve_project_documents`` and rejected everywhere else, so
    a search can never be routed without saying what it searches for.
    """

    message: str | None = Field(default=None, min_length=1)
    """The words the user gets, on the routes where the decision is the reply.

    Kept apart from ``rationale`` on purpose. This is addressed to the person
    who asked; ``rationale`` is addressed to whoever audits the run, is capped,
    and is never shown. One field serving both would mean either shipping audit
    notes to users or truncating answers at the audit cap.
    """

    rationale: str = Field(default="", max_length=RATIONALE_MAX_CHARS)
    """A one-line summary for a reader. Never parsed, never dispatched on."""

    @field_validator("tool_arguments")
    @classmethod
    def _freeze_arguments(
        cls, value: Mapping[str, Any] | None
    ) -> Mapping[str, Any] | None:
        """Take a copy behind a read-only view, as every carried call is."""
        return None if value is None else MappingProxyType(dict(value))

    @field_serializer("tool_arguments")
    def _unwrap_arguments(
        self, value: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        return None if value is None else dict(value)

    @model_validator(mode="after")
    def _fields_must_match_the_route(self) -> "ReasoningDecision":
        names_tool = self.route in _TOOL_ROUTES

        if names_tool and self.required_tool is None:
            raise ValueError(
                f"required_tool: route {self.route!r} is a call and must name "
                f"the tool it calls"
            )
        if not names_tool and self.required_tool is not None:
            raise ValueError(
                f"required_tool: route {self.route!r} calls no tool, so a tool "
                f"name here is an input no branch will ever read"
            )
        if self.required_tool is not None and not self.required_tool.strip():
            raise ValueError("required_tool: must not be blank")

        if names_tool and self.tool_arguments is None:
            raise ValueError(
                f"tool_arguments: route {self.route!r} calls a tool, and an "
                f"approver cannot be shown a call whose arguments are missing"
            )
        if not names_tool and self.tool_arguments is not None:
            raise ValueError(
                f"tool_arguments: route {self.route!r} calls nothing, so "
                f"arguments here are an input no branch will ever read"
            )

        if self.mutating and not names_tool:
            raise ValueError(
                f"mutating: route {self.route!r} executes no tool, so there is "
                f"nothing here that could change ERP data"
            )
        if self.mutating and not self.approval_required:
            raise ValueError(
                "mutating: a call that changes ERP data always needs a recorded "
                "approval; a decision that says otherwise would route a write "
                "straight past the gate"
            )

        if self.route == _RETRIEVAL_ROUTE and self.search_query is None:
            raise ValueError(
                "search_query: a retrieval decision must say what it searches "
                "for; the route alone leaves the query to be invented later"
            )
        if self.search_query is not None and self.route != _RETRIEVAL_ROUTE:
            raise ValueError(
                f"search_query: route {self.route!r} searches nothing, so a "
                f"query here is an input no branch will ever read"
            )

        if self.route in _MESSAGE_REQUIRED_ROUTES and self.message is None:
            raise ValueError(
                f"message: route {self.route!r} ends the turn by saying "
                f"something, and no later step exists to supply the words"
            )
        if self.message is not None and self.route not in _MESSAGE_ROUTES:
            raise ValueError(
                f"message: route {self.route!r} says nothing to the user, so "
                f"text here is a reply that would never be delivered"
            )

        if self.approval_required and not names_tool:
            raise ValueError(
                f"approval_required: nothing executes on route {self.route!r}, "
                f"so there is nothing for an approver to approve"
            )

        if any(not source.strip() for source in self.required_evidence):
            raise ValueError("required_evidence: source ids must not be blank")

        return self


def classify_failure(
    *,
    evidence_count: int,
    budget_overflow: bool,
    provider_error: bool,
) -> FailureMode:
    """Name the single failure that best explains a turn, or ``"none"``.

    Keyword-only on purpose: two of the three arguments are booleans, and a
    positional swap between them would be silent and would misclassify every
    failure it touched.

    The precedence is the design. ``budget_overflow`` wins over everything
    because it is the only condition detectable *before* the provider is
    touched: when a request was too large to send, a provider error and an empty
    evidence set are symptoms of a call that should never have been made, and
    reporting either of them sends a reader to the wrong layer. Provider failure
    then outranks thin evidence for the same reason -- a call that never
    returned explains an empty result, and the reverse is not true.

    Args:
        evidence_count: How many snippets retrieval actually supplied.
        budget_overflow: Whether the estimate exceeded the context budget.
        provider_error: Whether the provider call ultimately failed.

    Returns:
        The dominant :data:`FailureMode`, or ``"none"`` when the turn was fine.

    Raises:
        ValueError: ``evidence_count`` is negative, which is not a state
            retrieval can produce and so is a bug in the caller.
    """
    if evidence_count < 0:
        raise ValueError(f"evidence_count: cannot be negative, got {evidence_count}")

    if budget_overflow:
        return "context_budget_exceeded"
    if provider_error:
        return "provider_failure"
    if evidence_count == 0:
        return "insufficient_evidence"
    return "none"
