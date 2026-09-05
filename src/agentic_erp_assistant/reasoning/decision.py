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
    Decided by context construction -- the question needs grounding -- and is
    independent of any tool call.

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

Five of the six name a *preparatory* action; ``answer`` names the absence of
one, and it is a member of the set rather than an implied ``None`` on purpose.
A ``None`` route would give "the runtime decided to reply" and "nothing has
been decided yet" the same value, and the graph has to tell those apart: the
first is a node to dispatch, the second is a turn that never routed. It is
added now, and was not present before, because only now is there a graph node
that can dispatch it -- a label with no branch behind it is a lie the type
system helps tell.

What this module deliberately does not check
--------------------------------------------

``required_tool`` is not validated against the tool registry. That would couple
the decision layer to :mod:`agentic_erp_assistant.llm.tools` and would turn "the
model named a tool that does not exist" into an object that cannot be
constructed -- when it should be a routed, traced, reportable failure. The
router owns that check, against ``DEFAULT_TOOLS``, where the failure has
somewhere to be recorded.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "classify_failure",
    "DecisionRoute",
    "FailureMode",
    "RATIONALE_MAX_CHARS",
    "ReasoningDecision",
]


DecisionRoute = Literal[
    "retrieve_project_documents",  # go to RAG before answering
    "call_tool",                   # a tool call that may run unattended
    "request_approval",            # a tool call that must stop for a human
    "answer",                      # nothing left to prepare; reply from state
    "clarify",                     # the request is under-specified; ask back
    "refuse",                      # do not proceed at all
]
"""The actions the runtime knows how to take next.

A closed set: an action nobody can dispatch has to fail here, at the
decision, rather than later as a missing branch.
"""


FailureMode = Literal[
    "context_budget_exceeded",
    "provider_failure",
    "insufficient_evidence",
    "none",
]
"""Why a turn could not produce a grounded answer -- including "it could".

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

    approval_required: bool = False
    """Whether a human decision must be recorded before anything executes."""

    rationale: str = Field(default="", max_length=RATIONALE_MAX_CHARS)
    """A one-line summary for a reader. Never parsed, never dispatched on."""

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
