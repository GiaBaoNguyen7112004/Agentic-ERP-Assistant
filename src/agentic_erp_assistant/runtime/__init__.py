"""The graph engine and the contracts its nodes run against.

Today it holds the two ports the workflow depends on and the table of moves
the graph is allowed to make; the node contracts, retry budget and the engine
itself land here as they are built.

The import rule for everything in this package: it may depend on
:mod:`agentic_erp_assistant.state` and on the protocols declared here, and on
nothing that names a concrete backend. A node that imports a retriever, a
gateway, a provider or the web layer has coupled the runtime to a
replaceable part, and ``tests/runtime/test_ports.py`` checks that it has not.
"""

from agentic_erp_assistant.runtime.ports import (
    DocumentRetrieverPort,
    ToolGatewayPort,
    ToolOutcome,
)
from agentic_erp_assistant.runtime.transitions import (
    ALLOWED,
    advance,
    assert_transition,
    IllegalTransition,
    TERMINAL_ROUTES,
)

__all__ = [
    "ALLOWED",
    "advance",
    "assert_transition",
    "DocumentRetrieverPort",
    "IllegalTransition",
    "TERMINAL_ROUTES",
    "ToolGatewayPort",
    "ToolOutcome",
]
