"""Keep the developer's real configuration out of the composition tests.

The same reasoning as ``tests/llm/conftest.py``: a real ``.env`` on disk would
make "the setting is missing" tests green for the wrong reason. Settings.from_env
reads its own environment and also calls
``persistence.connection.url_from_environment``, which calls ``load_dotenv``
again in its own module -- both are neutralized here.
"""

import pytest

from agentic_erp_assistant.composition import settings as settings_module
from agentic_erp_assistant.persistence import connection as connection_module

_VARIABLES = (
    "OPENAI_MODEL",
    "OPENAI_CONTEXT_WINDOW",
    "OPENAI_OUTPUT_RESERVE",
    "POSTGRES_URL",
    "MEMORY_PROPOSER",
    "DEV_TOOL_RATE_LIMIT",
    "DEV_FLAKY_STATUS",
    "DEV_MAX_STEPS",
    "DEV_HISTORY_TURN_LIMIT",
    "USERS_PATH",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every setting this package reads, and neutralize ``.env`` loading."""
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(connection_module, "load_dotenv", lambda *a, **k: False)
    for name in _VARIABLES:
        monkeypatch.delenv(name, raising=False)
