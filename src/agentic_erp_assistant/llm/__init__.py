"""LLM access: the typed provider port, its response contracts, and its policies.

Everything exported here is provider-neutral, and importing this package pulls in
no HTTP client and no vendor dependency. The adapters live one level down, in
``llm.adapters``, and are imported by name at the construction site -- see
:mod:`agentic_erp_assistant.llm.adapters` for why they are not re-exported here.
"""

from agentic_erp_assistant.llm.gateway import (
    ContextWindowExceeded,
    Evidence,
    LLMGateway,
    WHOLE_DOCUMENT,
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
from agentic_erp_assistant.llm.retry import retry_with_backoff
from agentic_erp_assistant.llm.schemas import (
    ApprovalRequest,
    Citation,
    ClassifiedIntent,
    EvidenceSnippet,
    GroundedAnswer,
    Route,
)
from agentic_erp_assistant.llm.telemetry import (
    InMemoryTelemetry,
    ModelCallRecord,
    Outcome,
    TelemetrySink,
)
from agentic_erp_assistant.llm.tokenizer import (
    FALLBACK_ENCODING,
    TiktokenCounter,
    TokenCounter,
    count_message_tokens,
    count_tokens,
)
from agentic_erp_assistant.llm.tools import (
    DEFAULT_TOOLS,
    GET_PROJECT_STATUS_TOOL,
    ProjectStatusArguments,
    StrictArguments,
    ToolCallResult,
    ToolSpec,
)

__all__ = [
    "ApprovalRequest",
    "build_messages",
    "Citation",
    "ClassifiedIntent",
    "ClientConfigurationError",
    "CompletionResponse",
    "ContextWindowExceeded",
    "count_message_tokens",
    "count_tokens",
    "DEFAULT_TOOLS",
    "DEVELOPER_CONTRACT",
    "estimate_cost_usd",
    "Evidence",
    "EvidenceSnippet",
    "FALLBACK_ENCODING",
    "GET_PROJECT_STATUS_TOOL",
    "GroundedAnswer",
    "InMemoryTelemetry",
    "LargeLanguageModelClient",
    "LLMGateway",
    "LLMClientError",
    "Message",
    "MODEL_RATES",
    "ModelCallRecord",
    "ModelRate",
    "NO_EVIDENCE",
    "Outcome",
    "PRICING_CHECKED_ON",
    "PRICING_SOURCE",
    "ProjectStatusArguments",
    "ProviderAuthError",
    "retry_with_backoff",
    "Role",
    "Route",
    "StrictArguments",
    "SYSTEM_POLICY",
    "TiktokenCounter",
    "TokenCounter",
    "TelemetrySink",
    "ToolCallResult",
    "ToolSpec",
    "TransientProviderError",
    "UnknownModelRateError",
    "UsageReporting",
    "Usage",
    "WHOLE_DOCUMENT",
]
