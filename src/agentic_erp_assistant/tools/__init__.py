"""The tool boundary: what a call looks like, and what running one produced.

``models.py`` is the whole contract in one file, ``registry.py`` is the control
plane that declares every tool's policy, and ``handlers.py`` is the code each
tool runs. The gateway that orders them lands beside these. Nothing in ``runtime/`` imports this package --
the graph depends on
:class:`~agentic_erp_assistant.runtime.ports.ToolGatewayPort`, and an
implementation here satisfies that protocol structurally.
"""

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
    "build_default_registry",
    "Handler",
    "HandlerResult",
    "NO_RETRY",
    "RetryPolicy",
    "SideEffect",
    "ToolDefinition",
    "ToolError",
    "ToolOutcome",
    "ToolRegistry",
    "ToolRequest",
    "ToolStatus",
    "TransientToolError",
    "UnknownTool",
]
