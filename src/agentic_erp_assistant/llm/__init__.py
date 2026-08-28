"""LLM access: the typed provider port, its adapters, and its response contracts.

Import the port, the schemas and the prompt builder from here; provider SDKs stay
behind the adapter modules.
"""

from agentic_erp_assistant.llm.ports import (
    ClientConfigurationError,
    CompletionResponse,
    LargeLanguageModelClient,
    LLMClientError,
    Message,
    ProviderAuthError,
    Role,
    TransientProviderError,
    Usage,
)
from agentic_erp_assistant.llm.prompts import (
    DEVELOPER_CONTRACT,
    NO_EVIDENCE,
    SYSTEM_POLICY,
    build_messages,
)
from agentic_erp_assistant.llm.schemas import (
    ApprovalRequest,
    Citation,
    ClassifiedIntent,
    EvidenceSnippet,
    GroundedAnswer,
    Route,
)

__all__ = [
    "ApprovalRequest",
    "build_messages",
    "Citation",
    "ClassifiedIntent",
    "ClientConfigurationError",
    "CompletionResponse",
    "DEVELOPER_CONTRACT",
    "EvidenceSnippet",
    "GroundedAnswer",
    "LargeLanguageModelClient",
    "LLMClientError",
    "Message",
    "NO_EVIDENCE",
    "ProviderAuthError",
    "Role",
    "Route",
    "SYSTEM_POLICY",
    "TransientProviderError",
    "Usage",
]
