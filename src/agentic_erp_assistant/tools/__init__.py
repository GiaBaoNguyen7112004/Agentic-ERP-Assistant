"""The tool boundary: what a call looks like, and what running one produced.

``models.py`` is the whole contract in one file; the router, the approval gate
and the handlers land beside it. Nothing in ``runtime/`` imports this package --
the graph depends on
:class:`~agentic_erp_assistant.runtime.ports.ToolGatewayPort`, and an
implementation here satisfies that protocol structurally.
"""

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

__all__ = [
    "ApprovalDecision",
    "ARGUMENTS_SUMMARY_MAX_CHARS",
    "AuditRow",
    "ToolError",
    "ToolOutcome",
    "ToolRequest",
    "ToolStatus",
    "TransientToolError",
]
