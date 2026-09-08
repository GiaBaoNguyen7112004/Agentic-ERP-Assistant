"""Builders the memory tests share.

Kept here rather than repeated per file because a memory record has eleven fields
and only two or three matter in any one test. A helper that defaults the rest
lets each test say what it is actually about -- and lets a new required field be
added in one place rather than in ninety.

A plain module rather than a ``conftest.py``, because these are factories a test
calls with overrides, not fixtures pytest injects. The suite runs with
``--import-mode=importlib`` and ``consider_namespace_packages``, so
``tests.memory.builders`` imports by its real path.
"""

from datetime import UTC, datetime

from agentic_erp_assistant.memory.intent import IntentState
from agentic_erp_assistant.memory.models import MemoryScope
from agentic_erp_assistant.state.memory import MemoryRecord

RECORDED = datetime(2026, 9, 8, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 8, 10, 0, tzinfo=UTC)
LATEST = datetime(2026, 9, 8, 11, 0, tzinfo=UTC)


def make_scope(**overrides: object) -> MemoryScope:
    fields: dict[str, object] = {
        "project_code": "atlas",
        "session_id": "sess-1",
        "scopes": frozenset({"project.docs.read"}),
    }
    fields.update(overrides)
    actor = fields.pop("actor", "priya")
    return MemoryScope.for_actor(actor, **fields)  # type: ignore[arg-type]


def make_record(**overrides: object) -> MemoryRecord:
    fields: dict[str, object] = {
        "memory_id": "mem-1",
        "kind": "preference",
        "key": "reply_language",
        "statement": "Prefers replies written in Vietnamese.",
        "project_code": "atlas",
        "required_scope": "project.docs.read",
        "actor": "priya",
        "session_id": "sess-1",
        "recorded_in_run": "run-1",
        "recorded_at": RECORDED,
        "confidence": 0.9,
    }
    fields.update(overrides)
    return MemoryRecord(**fields)  # type: ignore[arg-type]


def make_intent(**overrides: object) -> IntentState:
    fields: dict[str, object] = {
        "intent_id": "int-1",
        "goal": "Draft the Q4 risk register",
        "project_code": "atlas",
        "required_scope": "project.docs.read",
        "actor": "priya",
        "session_id": "sess-1",
        "unresolved_slots": ("quarter",),
        "opened_at": RECORDED,
        "updated_at": RECORDED,
    }
    fields.update(overrides)
    return IntentState(**fields)  # type: ignore[arg-type]
