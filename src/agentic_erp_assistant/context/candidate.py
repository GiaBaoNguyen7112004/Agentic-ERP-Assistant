"""One thing that wants to be in the prompt, described in fields.

Context construction is a competition: memory, retrieved passages, tool results
and standing policy all want room in a window that cannot hold them, and
something has to lose. This module is the entry form for that competition. It
says what a contender must declare about itself before it is allowed to compete
-- and, just as deliberately, what it is not allowed to declare.

The field that is missing is the important one
----------------------------------------------

There is no ``token_count`` here, and no ``estimated_tokens``. A caller cannot
tell the builder how large its candidate is; the builder measures
:attr:`ContextCandidate.text` itself. That is the whole reason the field does not
exist rather than merely being documented as untrusted: a number sitting beside
the text it describes drifts from that text the first time someone edits one and
forgets the other, and the drift is silent. It shows up much later as a request
that was waved through the budget check and rejected by the provider.

A count that cannot be supplied cannot be wrong. See
:mod:`agentic_erp_assistant.context.builder` for the measurement itself.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["CandidateKind", "ContextCandidate"]


CandidateKind = Literal[
    "policy",  # standing instructions: what the assistant is and may not do
    "evidence",  # a retrieved project passage, on its way into the prompt
    "memory",  # something remembered from an earlier turn
    "tool_result",  # what a tool returned, being fed back to the model
    "history",  # a prior turn of this conversation
]
"""What sort of thing a candidate is.

A closed set, for the same reason
:data:`~agentic_erp_assistant.reasoning.decision.DecisionRoute` is one: these are
the buckets a trace groups by and a policy layer filters on, so a kind nobody
budgets for has to fail here, at construction, rather than downstream as a
category that quietly appeared in a report.

The kind is descriptive, never a priority in disguise. Whether policy outranks
memory is a decision the caller makes in :attr:`ContextCandidate.priority`, where
a reviewer can see the number.
"""


class ContextCandidate(BaseModel):
    """A single contender for space in the prompt.

    Frozen, because the builder sorts these and then reports the winners: a
    candidate edited after it was measured would make the plan describe something
    that was never assembled. ``extra="forbid"`` because an unmodelled field here
    is an input to context construction that no reviewer ever agreed to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    """How the plan refers to this candidate afterwards.

    Must be unique within one :meth:`~agentic_erp_assistant.context.builder.ContextBuilder.build`
    call -- the builder enforces that. It also breaks ties in the sort, so it has
    to be stable across runs: a positional index or an object id would make an
    identical input produce a different plan on a different day.
    """

    kind: CandidateKind
    """What sort of thing this is. See :data:`CandidateKind`."""

    text: str
    """The exact text that would go into the prompt.

    The single source of truth for this candidate's size. Empty is allowed and
    costs nothing; it is not this module's job to decide that a caller's empty
    passage is a mistake.
    """

    priority: int
    """Who wins when the budget runs out. Higher is kept first.

    Plain integers, not an enum, because priority is the caller's policy and this
    module has no business fixing the order in which a project's own memory,
    evidence and history compete.
    """

    allowed: bool = True
    """Whether policy permits this text in the prompt at all.

    ``False`` is a refusal, not a preference, and the builder honors it before it
    looks at the budget -- so a denied candidate is never reported as something
    that merely did not fit. Defaults to ``True`` because a candidate arriving
    without a policy verdict has not been denied; a layer that filters is
    expected to set it explicitly.
    """
