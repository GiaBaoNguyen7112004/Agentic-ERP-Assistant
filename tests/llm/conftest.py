"""Keep the developer's real configuration out of the client tests.

Two ways an ambient environment could make these tests lie: a real
``OPENAI_API_KEY`` exported in the shell, and a real ``.env`` on disk that
``OpenAIChatClient`` would load at construction. Either one turns the
"configuration is missing" tests green for the wrong reason on the author's
machine and red in CI -- or worse, the reverse.

So the fixture below removes both, for every test in this directory. That the
suite then runs with no key and no model anywhere is not an inconvenience; it is
the point. Nothing here should be able to reach the network even by accident.
"""

import pytest

from agentic_erp_assistant.llm import client as client_module

_OPENAI_VARIABLES = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the provider variables and neutralize ``.env`` loading."""
    monkeypatch.setattr(
        client_module, "load_dotenv", lambda *args, **kwargs: False
    )
    for name in _OPENAI_VARIABLES:
        monkeypatch.delenv(name, raising=False)
