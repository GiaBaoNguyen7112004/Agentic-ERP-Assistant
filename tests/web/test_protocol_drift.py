"""The Python event vocabulary and ui/src/protocol.ts must never drift apart.

A renamed or added event type is a wire-format change two independently
typed files have to agree on; nothing at the type-checker level enforces
that agreement across a language boundary. This test is the enforcement:
it reads the TypeScript source as text and checks its `type: "..."`
literals against web/protocol.py's own EVENT_TYPES, so a mismatch fails
`pytest`, not a browser console three files away.
"""

import re
from pathlib import Path

from agentic_erp_assistant.web.protocol import EVENT_TYPES

PROTOCOL_TS = Path(__file__).resolve().parents[2] / "ui" / "src" / "protocol.ts"


def _ts_event_types() -> set[str]:
    """Every `type: '...'` literal declared in protocol.ts's event interfaces.

    Matches ``type: 'turn_started'`` (single-quoted, TypeScript's own
    convention in this file) rather than parsing the file as TypeScript --
    this is a drift guard, not a compiler, and a regex over one narrow,
    consistently-formatted pattern is the whole job.
    """
    source = PROTOCOL_TS.read_text(encoding="utf-8")
    return set(re.findall(r"type:\s*'([a-z_]+)'", source))


def test_protocol_ts_exists() -> None:
    assert PROTOCOL_TS.exists(), (
        f"{PROTOCOL_TS} is missing -- has ui/src/protocol.ts moved or been renamed?"
    )


def test_every_python_event_type_is_named_in_protocol_ts() -> None:
    ts_types = _ts_event_types()
    missing = set(EVENT_TYPES) - ts_types
    assert not missing, f"protocol.ts is missing event type(s): {sorted(missing)}"


def test_protocol_ts_names_no_event_type_python_does_not_have() -> None:
    ts_types = _ts_event_types()
    extra = ts_types - set(EVENT_TYPES)
    assert not extra, f"protocol.ts declares event type(s) EVENT_TYPES does not: {sorted(extra)}"


def test_event_types_export_matches_the_ts_array_too() -> None:
    """EVENT_TYPES itself (the const array ui/src/protocol.ts also exports)
    must list the same members -- catches a type-only interface added
    without also being added to the runtime array a test might iterate."""
    source = PROTOCOL_TS.read_text(encoding="utf-8")
    match = re.search(r"EVENT_TYPES:.*?=\s*\[(.*?)\]\s*as const", source, re.DOTALL)
    assert match, "protocol.ts must export EVENT_TYPES as a const array"
    declared = set(re.findall(r"'([a-z_]+)'", match.group(1)))
    assert declared == set(EVENT_TYPES)
