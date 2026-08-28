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
from agentic_erp_assistant.llm.pricing import (
    MODEL_RATES,
    ModelRate,
    PRICING_CHECKED_ON,
    PRICING_SOURCE,
    UnknownModelRateError,
    estimate_cost_usd,
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
from agentic_erp_assistant.llm.tokenizer import (
    FALLBACK_ENCODING,
    TiktokenCounter,
    TokenCounter,
    count_message_tokens,
    count_tokens,
)

__all__ = [
    "ApprovalRequest",
    "build_messages",
    "Citation",
    "ClassifiedIntent",
    "ClientConfigurationError",
    "CompletionResponse",
    "count_message_tokens",
    "count_tokens",
    "DEVELOPER_CONTRACT",
    "estimate_cost_usd",
    "EvidenceSnippet",
    "FALLBACK_ENCODING",
    "GroundedAnswer",
    "LargeLanguageModelClient",
    "LLMClientError",
    "Message",
    "MODEL_RATES",
    "ModelRate",
    "NO_EVIDENCE",
    "PRICING_CHECKED_ON",
    "PRICING_SOURCE",
    "ProviderAuthError",
    "Role",
    "Route",
    "SYSTEM_POLICY",
    "TiktokenCounter",
    "TokenCounter",
    "TransientProviderError",
    "UnknownModelRateError",
    "Usage",
]
