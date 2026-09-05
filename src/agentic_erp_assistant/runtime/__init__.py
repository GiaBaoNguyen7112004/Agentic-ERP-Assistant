"""The graph engine and the contracts its nodes run against.

It holds the four ports the workflow depends on, the table of moves the graph
is allowed to make, the three nodes that make them, and the engine that applies
nodes until a turn ends, pauses on an approval, or exhausts its step budget.

The import rule for everything in this package: it may depend on
:mod:`agentic_erp_assistant.state` and on the protocols declared here, and on
nothing that names a concrete backend. A node that imports a retriever, a
gateway, a provider or the web layer has coupled the runtime to a
replaceable part, and ``tests/runtime/test_ports.py`` checks that it has not.
"""

from agentic_erp_assistant.runtime.ports import (
    AnswerComposerPort,
    DocumentRetrieverPort,
    PlannerPort,
    ToolGatewayPort,
    ToolOutcome,
)
from agentic_erp_assistant.runtime.nodes import (
    EVIDENCE_LIMIT,
    GraphNodes,
    Node,
    NodeTable,
)
from agentic_erp_assistant.runtime.transitions import (
    ALLOWED,
    advance,
    assert_transition,
    IllegalTransition,
    TERMINAL_ROUTES,
)
from agentic_erp_assistant.runtime.workflow import (
    MAX_STEPS,
    NotPaused,
    UnroutableState,
    WorkflowRuntime,
    is_paused,
)

__all__ = [
    "ALLOWED",
    "AnswerComposerPort",
    "advance",
    "assert_transition",
    "DocumentRetrieverPort",
    "EVIDENCE_LIMIT",
    "GraphNodes",
    "IllegalTransition",
    "is_paused",
    "MAX_STEPS",
    "Node",
    "NodeTable",
    "NotPaused",
    "PlannerPort",
    "TERMINAL_ROUTES",
    "ToolGatewayPort",
    "ToolOutcome",
    "UnroutableState",
    "WorkflowRuntime",
]
