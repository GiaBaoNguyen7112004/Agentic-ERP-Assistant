"""The reasoning state: everything one turn knows, in typed, frozen objects.

The innermost layer. It imports the decision vocabulary and nothing else -- no
provider, no retriever, no gateway, no web framework -- so a state can be built,
serialized and asserted on in a test that touches no I/O at all. Layers above
depend on this one; it depends on none of them.
"""

from agentic_erp_assistant.state.agent_state import (
    AgentState,
    ApprovalDecision,
    ERROR_DETAIL_MAX_CHARS,
    STATE_VERSION,
)
from agentic_erp_assistant.state.conversation import ConversationTurn
from agentic_erp_assistant.state.events import (
    EVENT_DETAIL_MAX_CHARS,
    EventKind,
    TraceEvent,
)
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.memory import (
    KEY_MAX_CHARS,
    MemoryKind,
    MemoryRecord,
    STATEMENT_MAX_CHARS,
)
from agentic_erp_assistant.state.reply_contract import (
    EMPTY_CONTRACT,
    ReplyContract,
    ReplyNeed,
)
from agentic_erp_assistant.state.tool_outcome import ToolOutcome, ToolStatus
from agentic_erp_assistant.state.tool_request import ToolRequest

__all__ = [
    "AgentState",
    "ApprovalDecision",
    "ConversationTurn",
    "EMPTY_CONTRACT",
    "ERROR_DETAIL_MAX_CHARS",
    "EVENT_DETAIL_MAX_CHARS",
    "EventKind",
    "EvidenceSnippet",
    "KEY_MAX_CHARS",
    "MemoryKind",
    "MemoryRecord",
    "ReplyContract",
    "ReplyNeed",
    "STATEMENT_MAX_CHARS",
    "STATE_VERSION",
    "ToolOutcome",
    "ToolRequest",
    "ToolStatus",
    "TraceEvent",
]
