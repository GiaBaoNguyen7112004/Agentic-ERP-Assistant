"""The decision layer: what the runtime does next, and why a turn failed.

Provider-neutral and dependency-light on purpose -- importing this package pulls
in no HTTP client, no adapter, and nothing from :mod:`agentic_erp_assistant.llm`.
A routing decision is a fact about the run, not about who served it.

``reasoning.planner`` is the exception, and is deliberately not re-exported
here. It has to name what a tool is, so importing it reaches into
``llm.tools``; keeping it out of this file means the light import stays
light and the cost is paid only by whoever actually builds a planner.

``reasoning.completeness`` is a second exception, for a sharper reason than
weight: it imports :mod:`agentic_erp_assistant.state.agent_state`, which
imports :mod:`agentic_erp_assistant.reasoning.decision` -- and *that* import
is what actually loads this package's own ``__init__``. Re-exporting
``completeness`` here would make finishing this file's own import depend on
this file having already finished importing, which Python quite reasonably
refuses. Import it directly:
``from agentic_erp_assistant.reasoning.completeness import assess``.
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
