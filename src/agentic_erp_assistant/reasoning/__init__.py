"""The decision layer: what the runtime does next, and why a turn failed.

Provider-neutral and dependency-light on purpose -- importing this package pulls
in no HTTP client, no adapter, and nothing from :mod:`agentic_erp_assistant.llm`.
A routing decision is a fact about the run, not about who served it.
"""

from agentic_erp_assistant.reasoning.decision import (
    classify_failure,
    DecisionRoute,
    FailureMode,
    RATIONALE_MAX_CHARS,
    ReasoningDecision,
)

__all__ = [
    "classify_failure",
    "DecisionRoute",
    "FailureMode",
    "RATIONALE_MAX_CHARS",
    "ReasoningDecision",
]
