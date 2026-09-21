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
from agentic_erp_assistant.context.catalogue import (
    build_catalogue,
    CatalogueEntry,
    DocumentCatalogue,
)
from agentic_erp_assistant.context.compact import (
    CompactedConversation,
    compact_conversation,
    PRESERVED_FIELDS,
    structural_summary,
    SUMMARY_MAX_CHARS,
    SUMMARY_UNAVAILABLE,
    Summarizer,
)
from agentic_erp_assistant.context.history_injection import (
    CLIP_MARKER,
    HISTORY_BUDGET_TOKENS,
    HISTORY_TURN_LIMIT,
    HistorySelection,
    clip_to_tokens,
    select_history,
    strip_citations,
)
from agentic_erp_assistant.context.memory_injection import (
    KIND_PRIORITY,
    MemorySelection,
    PINNED_KINDS,
    SkipReason,
    SkippedMemory,
    select_memories,
)

__all__ = [
    "build_catalogue",
    "CandidateKind",
    "CatalogueEntry",
    "CLIP_MARKER",
    "CompactedConversation",
    "compact_conversation",
    "ContextBuilder",
    "ContextCandidate",
    "ContextPlan",
    "DocumentCatalogue",
    "ExcludedCandidate",
    "ExclusionReason",
    "HISTORY_BUDGET_TOKENS",
    "HISTORY_TURN_LIMIT",
    "HistorySelection",
    "clip_to_tokens",
    "KIND_PRIORITY",
    "MemorySelection",
    "PINNED_KINDS",
    "PRESERVED_FIELDS",
    "select_history",
    "select_memories",
    "SkippedMemory",
    "SkipReason",
    "strip_citations",
    "structural_summary",
    "SUMMARY_MAX_CHARS",
    "SUMMARY_UNAVAILABLE",
    "Summarizer",
]
