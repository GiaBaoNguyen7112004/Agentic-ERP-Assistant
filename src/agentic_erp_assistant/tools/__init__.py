"""The tool boundary: what a call looks like, and what running one produced.

``models.py`` is the whole contract in one file, ``registry.py`` is the control
plane that declares every tool's policy, ``handlers.py`` is the code each tool
runs, and ``gateway.py`` is the single ordered path from a request to an
outcome -- validate, permit, approve, audit, execute, trace, in that order.

Nothing in ``runtime/`` imports this package -- the graph depends on
:class:`~agentic_erp_assistant.runtime.ports.ToolGatewayPort`, and an
implementation here satisfies that protocol structurally.
"""

from agentic_erp_assistant.tools.audit import AuditSink, InMemoryAuditLog
from agentic_erp_assistant.tools.gateway import GATEWAY_NODE, ToolGateway
from agentic_erp_assistant.tools.handlers import Handler, HandlerResult
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
    NO_RETRY,
    RetryPolicy,
    SideEffect,
    ToolDefinition,
    ToolRegistry,
    UnknownTool,
)

__all__ = [
    "ApprovalDecision",
    "ARGUMENTS_SUMMARY_MAX_CHARS",
    "AuditRow",
    "AuditSink",
    "build_default_registry",
    "GATEWAY_NODE",
    "Handler",
    "HandlerResult",
    "InMemoryAuditLog",
    "NO_RETRY",
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
]
