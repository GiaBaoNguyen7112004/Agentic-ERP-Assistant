"""Where a mutating call stands with its approver.

A leaf module, deliberately: :class:`~agentic_erp_assistant.state.agent_state.AgentState`
and :class:`~agentic_erp_assistant.state.conversation.ConversationTurn` both need this
type, and ``AgentState`` also needs ``ConversationTurn`` -- so the type they share
has to sit below both of them, or the two would import each other.
"""

from typing import Literal

__all__ = ["ApprovalDecision"]


ApprovalDecision = Literal[
    "not_required",  # nothing mutating is pending
    "pending",       # a human has been asked and has not answered
    "approved",      # a human said yes, and it is recorded here
    "denied",        # a human said no
]
"""Where a mutating call stands with its approver.

Four members, and ``"not_required"`` is a string rather than ``None`` for the
reason ``FailureMode`` uses ``"none"``: an optional field invites
``if state.approval:`` at the gate, which collapses "nobody needs to approve
this" and "we asked and were refused" into the same falsy value. The one place
that must never be ambiguous is the gate in front of a write.
"""
