"""The ERP behind the tools -- today, a synthetic one.

A single mock store over a JSON fixture in ``data/erp/``. It is deliberately
plain: no port, no adapter, no interface with one implementation. The seam that
matters for swapping a real ERP in is
:class:`~agentic_erp_assistant.engine.ports.ToolGatewayPort`, which already
exists; adding a second abstraction here would be an interface invented before
its second implementation.
"""

from agentic_erp_assistant.erp.mock import (
    Budget,
    DEFAULT_DATASET_PATH,
    Milestone,
    MockErp,
    Project,
    Risk,
    RiskSeverity,
    Sprint,
)

__all__ = [
    "Budget",
    "DEFAULT_DATASET_PATH",
    "Milestone",
    "MockErp",
    "Project",
    "Risk",
    "RiskSeverity",
    "Sprint",
]
