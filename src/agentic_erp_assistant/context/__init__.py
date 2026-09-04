"""Context construction: what gets into the prompt, and why the rest did not.

The one thing this package refuses to own is a tokenizer. Counting lives in
:mod:`agentic_erp_assistant.llm.tokenizer` and is used here through its
``TokenCounter`` port, so the budget the builder plans against and the budget the
gateway checks before sending are the same arithmetic. A second counter here
would let those two disagree about the same text, and a trace could no longer
say which of them refused a request.
"""

from agentic_erp_assistant.context.builder import (
    ContextBuilder,
    ContextPlan,
    ExcludedCandidate,
    ExclusionReason,
)
from agentic_erp_assistant.context.candidate import CandidateKind, ContextCandidate

__all__ = [
    "CandidateKind",
    "ContextBuilder",
    "ContextCandidate",
    "ContextPlan",
    "ExcludedCandidate",
    "ExclusionReason",
]
