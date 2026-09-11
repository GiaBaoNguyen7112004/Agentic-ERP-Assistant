"""Settings.from_env: the environment, typed, or a named local failure."""

import logging
import re
from pathlib import Path

import pytest

from agentic_erp_assistant.composition.settings import (
    DEFAULT_USERS_PATH,
    MEMORY_REQUIRED_SCOPE,
    Settings,
    SettingsError,
)
from agentic_erp_assistant.tools.registry import RateLimitPolicy


def set_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "128000")


# --------------------------------------------------------------------------
# The two required fields
# --------------------------------------------------------------------------


def test_a_missing_model_is_a_named_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "128000")

    with pytest.raises(SettingsError, match="OPENAI_MODEL"):
        Settings.from_env()


def test_a_missing_context_window_is_a_named_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    with pytest.raises(SettingsError, match="OPENAI_CONTEXT_WINDOW"):
        Settings.from_env()


def test_a_non_integer_context_window_is_a_named_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "a lot")

    with pytest.raises(SettingsError, match="OPENAI_CONTEXT_WINDOW"):
        Settings.from_env()


def test_both_required_fields_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required(monkeypatch)

    settings = Settings.from_env()

    assert settings.model == "gpt-4o"
    assert settings.context_window == 128000


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------


def test_output_reserve_defaults_to_1024(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)

    assert Settings.from_env().output_reserve == 1024


def test_memory_required_scope_is_the_fixed_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required(monkeypatch)

    assert Settings.from_env().memory_required_scope == MEMORY_REQUIRED_SCOPE
    assert MEMORY_REQUIRED_SCOPE == "project.docs.read"


def test_postgres_url_falls_back_to_the_compose_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required(monkeypatch)

    assert "agentic_erp" in Settings.from_env().postgres_url


def test_users_path_defaults_to_the_repo_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required(monkeypatch)

    assert Settings.from_env().users_path == DEFAULT_USERS_PATH


def test_every_dev_toggle_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)

    settings = Settings.from_env()

    assert settings.dev_tool_rate_limit is None
    assert settings.dev_flaky_status is False
    assert settings.dev_max_steps is None
    assert settings.dev_history_turn_limit is None
    assert settings.memory_proposer is True


# --------------------------------------------------------------------------
# MEMORY_PROPOSER
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [("off", False), ("on", True), ("", True)])
def test_memory_proposer_only_off_disables_it(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("MEMORY_PROPOSER", raw)

    assert Settings.from_env().memory_proposer is expected


# --------------------------------------------------------------------------
# DEV_TOOL_RATE_LIMIT
# --------------------------------------------------------------------------


def test_dev_tool_rate_limit_parses_calls_slash_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("DEV_TOOL_RATE_LIMIT", "2/20")

    policy = Settings.from_env().dev_tool_rate_limit

    assert policy == RateLimitPolicy(max_calls=2, per_seconds=20.0)


def test_dev_tool_rate_limit_tolerates_surrounding_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("DEV_TOOL_RATE_LIMIT", " 3 / 45.5 ")

    policy = Settings.from_env().dev_tool_rate_limit

    assert policy == RateLimitPolicy(max_calls=3, per_seconds=45.5)


def test_a_malformed_dev_tool_rate_limit_is_a_named_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("DEV_TOOL_RATE_LIMIT", "not-a-rate")

    with pytest.raises(SettingsError, match="DEV_TOOL_RATE_LIMIT"):
        Settings.from_env()


# --------------------------------------------------------------------------
# The other dev toggles
# --------------------------------------------------------------------------


def test_dev_flaky_status_only_the_literal_1_turns_it_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("DEV_FLAKY_STATUS", "true")

    assert Settings.from_env().dev_flaky_status is False

    monkeypatch.setenv("DEV_FLAKY_STATUS", "1")

    assert Settings.from_env().dev_flaky_status is True


def test_dev_max_steps_and_history_turn_limit_parse_as_integers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("DEV_MAX_STEPS", "2")
    monkeypatch.setenv("DEV_HISTORY_TURN_LIMIT", "2")

    settings = Settings.from_env()

    assert settings.dev_max_steps == 2
    assert settings.dev_history_turn_limit == 2


def test_users_path_override_is_honoured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_required(monkeypatch)
    custom = tmp_path / "custom-users.json"
    monkeypatch.setenv("USERS_PATH", str(custom))

    assert Settings.from_env().users_path == custom


# --------------------------------------------------------------------------
# log_dev_toggles
# --------------------------------------------------------------------------


def test_log_dev_toggles_is_silent_with_nothing_active(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    set_required(monkeypatch)
    settings = Settings.from_env()

    with caplog.at_level(logging.WARNING):
        settings.log_dev_toggles()

    assert caplog.records == []


def test_log_dev_toggles_warns_once_per_active_toggle(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("DEV_TOOL_RATE_LIMIT", "2/20")
    monkeypatch.setenv("DEV_FLAKY_STATUS", "1")
    monkeypatch.setenv("DEV_MAX_STEPS", "2")
    monkeypatch.setenv("DEV_HISTORY_TURN_LIMIT", "2")
    monkeypatch.setenv("MEMORY_PROPOSER", "off")
    settings = Settings.from_env()

    with caplog.at_level(logging.WARNING):
        settings.log_dev_toggles()

    messages = "\n".join(caplog.messages)
    assert "DEV_TOOL_RATE_LIMIT" in messages
    assert "DEV_FLAKY_STATUS" in messages
    assert "DEV_MAX_STEPS" in messages
    assert "DEV_HISTORY_TURN_LIMIT" in messages
    assert "MEMORY_PROPOSER" in messages
    assert len(caplog.records) == 5
