"""The tool boundary: what a call looks like, and what running one produced.

``models.py`` is the whole contract in one file, ``registry.py`` is the control
plane that declares every tool's policy, ``handlers.py`` is the code each tool
runs, and ``gateway.py`` is the single ordered path from a request to an
outcome -- validate, permit, rate-limit, approve, audit, execute, trace, in
that order. ``limits.py`` holds the counter that order consults.

Nothing in ``engine/`` imports this package -- the graph depends on
:class:`~agentic_erp_assistant.engine.ports.ToolGatewayPort`, and an
implementation here satisfies that protocol structurally.
"""

from agentic_erp_assistant.tools.audit import AuditSink, InMemoryAuditLog
from agentic_erp_assistant.tools.gateway import GATEWAY_NODE, ToolGateway
from agentic_erp_assistant.tools.handlers import Handler, HandlerResult
from agentic_erp_assistant.tools.limits import InMemoryRateLimiter, RateLimiter
from agentic_erp_assistant.tools.models import (
    ApprovalDecision,
    ARGUMENTS_SUMMARY_MAX_CHARS,
    AuditRow,
    ToolError,
    ToolOutcome,
    ToolRequest,
    ToolStatus,
    TransientToolError,
)
from agentic_erp_assistant.tools.registry import (
    build_default_registry,
    DEFAULT_RATE_LIMIT,
    NO_RETRY,
    RateLimitPolicy,
    RetryPolicy,
    SideEffect,
    ToolDefinition,
    ToolRegistry,
    UnknownTool,
    WRITE_RATE_LIMIT,
)

__all__ = [
    "ApprovalDecision",
    "ARGUMENTS_SUMMARY_MAX_CHARS",
    "AuditRow",
    "AuditSink",
    "build_default_registry",
    "DEFAULT_RATE_LIMIT",
    "GATEWAY_NODE",
    "Handler",
    "HandlerResult",
    "InMemoryAuditLog",
    "InMemoryRateLimiter",
    "NO_RETRY",
    "RateLimiter",
    "RateLimitPolicy",
    "RetryPolicy",
    "SideEffect",
    "ToolDefinition",
    "ToolError",
    "ToolGateway",
    "ToolOutcome",
    "ToolRegistry",
    "ToolRequest",
    "ToolStatus",
    "TransientToolError",
    "UnknownTool",
    "WRITE_RATE_LIMIT",
]
