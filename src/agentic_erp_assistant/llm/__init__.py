"""LLM access: the typed provider port, its adapters, and its response contracts.

Import the port, the schemas and the prompt builder from here; provider SDKs stay
behind the adapter modules.
"""

from agentic_erp_assistant.llm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    EVIDENCE_PREAMBLE,
    OpenAIChatClient,
    RESPONSE_FORMAT_NAME,
)
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
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "DEVELOPER_CONTRACT",
    "estimate_cost_usd",
    "EVIDENCE_PREAMBLE",
    "EvidenceSnippet",
    "FALLBACK_ENCODING",
    "GroundedAnswer",
    "LargeLanguageModelClient",
    "LLMClientError",
    "Message",
    "MODEL_RATES",
    "ModelRate",
    "NO_EVIDENCE",
    "OpenAIChatClient",
    "PRICING_CHECKED_ON",
    "PRICING_SOURCE",
    "ProviderAuthError",
    "RESPONSE_FORMAT_NAME",
    "Role",
    "Route",
    "SYSTEM_POLICY",
    "TiktokenCounter",
    "TokenCounter",
    "TransientProviderError",
    "UnknownModelRateError",
    "Usage",
]
