"""``assert_test_database`` and ``test_url_from_environment``: pure, no database.

ADR 0018's guard is only as good as these two functions, so they get their
own tests independent of a running Postgres -- unlike everything else in this
package, which is marked ``postgres`` and needs the container.
"""

import pytest

from agentic_erp_assistant.persistence import (
    DEFAULT_POSTGRES_TEST_URL,
    StoreConfigurationError,
    assert_test_database,
)
from agentic_erp_assistant.persistence import connection as connection_module

# Not imported by name: pytest's default collection treats any top-level
# callable matching `test_*` as a test function, imported or not -- binding
# it here would make pytest try to run the real function as a test case.
_url_from_environment = connection_module.test_url_from_environment


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real ``.env``/``POSTGRES_TEST_URL`` on disk must not leak into these."""
    monkeypatch.setattr(connection_module, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("POSTGRES_TEST_URL", raising=False)


def test_a_url_naming_a_test_database_is_accepted() -> None:
    assert_test_database("postgresql://u:p@localhost:5432/agentic_erp_test")


def test_a_url_naming_the_dev_database_is_refused() -> None:
    with pytest.raises(StoreConfigurationError, match="agentic_erp"):
        assert_test_database("postgresql://u:p@localhost:5432/agentic_erp")


def test_a_url_naming_no_database_is_refused() -> None:
    with pytest.raises(StoreConfigurationError, match="none"):
        assert_test_database("postgresql://u:p@localhost:5432/")


def test_a_url_naming_a_database_merely_containing_test_is_refused() -> None:
    """The suffix is checked, not a substring -- ``test_agentic_erp`` or
    ``agentic_erp_testing`` must not sneak past a naive ``in`` check."""
    with pytest.raises(StoreConfigurationError):
        assert_test_database("postgresql://u:p@localhost:5432/agentic_erp_testing")


def test_environment_url_falls_back_to_the_compose_test_default() -> None:
    assert _url_from_environment() == DEFAULT_POSTGRES_TEST_URL


def test_environment_url_reads_postgres_test_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_TEST_URL", "postgresql://u:p@db:5432/agentic_erp_test")
    assert _url_from_environment() == "postgresql://u:p@db:5432/agentic_erp_test"
