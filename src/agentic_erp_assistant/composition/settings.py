"""Typed configuration, read from the environment exactly once per process.

Every field here has a row in the table this class's docstring names. That is
deliberate: a setting that is read with a bare ``os.environ.get`` scattered at
the point of use is a setting a reviewer cannot find by reading one file, and
one this module exists specifically to make findable.

Dev toggles are the other half of the point. ``DEV_TOOL_RATE_LIMIT``,
``DEV_FLAKY_STATUS``, ``DEV_MAX_STEPS`` and ``DEV_HISTORY_TURN_LIMIT`` let a
person exercise policies from a browser that a scripted test reaches with a
fake clock and a tight fixture instead -- a rate limit of 30/60s, an eight-step
budget and a six-turn window are none of them things a person clicking around
a demo can exhaust in a reasonable session. They live only here, read once at
startup and logged loudly (:meth:`Settings.log_dev_toggles`) so their effect on
a run is never a silent surprise to someone reading a trace later. Nothing in
``engine/``, ``tools/``, or ``memory/`` reads a ``DEV_*`` variable itself --
only this module does, and only this module ever will.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from agentic_erp_assistant.persistence.connection import url_from_environment
from agentic_erp_assistant.tools.registry import RateLimitPolicy

__all__ = ["DEFAULT_USERS_PATH", "MEMORY_REQUIRED_SCOPE", "Settings", "SettingsError"]

DEFAULT_USERS_PATH = Path(__file__).resolve().parents[3] / "data" / "users.json"
"""Where the dev chat's actor directory lives, relative to this file -- the
same depth :data:`~agentic_erp_assistant.erp.mock.DEFAULT_DATASET_PATH` and
:data:`~agentic_erp_assistant.rag.manifest.DEFAULT_MANIFEST_PATH` compute
theirs at, so all three review-material files sit beside each other under
``data/``."""

MEMORY_REQUIRED_SCOPE = "project.docs.read"
"""The entitlement a reader of a newly written memory will need.

A constant, not an environment variable: it is a fact about this deployment's
document baseline (every actor with any document access holds this scope),
not something an operator should be able to mistune per restart. See
:attr:`~agentic_erp_assistant.memory.service.MemoryService.required_scope`
for what it protects.
"""

_RATE_LIMIT_PATTERN = re.compile(r"^\s*(\d+)\s*/\s*(\d+(?:\.\d+)?)\s*$")

logger = logging.getLogger(__name__)


class SettingsError(RuntimeError):
    """Required local configuration is missing or unreadable.

    A startup bug, the same way
    :class:`~agentic_erp_assistant.llm.ports.ClientConfigurationError` is one
    for the adapter: nothing was sent to any provider, a `.env` is incomplete,
    and the fix is local.
    """


def _require(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise SettingsError(
            f"{name} is not set. Put it in .env (see .env.example); this "
            f"project chooses no default for it."
        )
    return value


def _require_int(name: str) -> int:
    raw = _require(name)
    try:
        return int(raw)
    except ValueError:
        raise SettingsError(f"{name}={raw!r} is not an integer") from None


def _optional_int(name: str) -> int | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise SettingsError(f"{name}={raw!r} is not an integer") from None


def _parse_rate_limit(raw: str) -> RateLimitPolicy:
    match = _RATE_LIMIT_PATTERN.match(raw)
    if not match:
        raise SettingsError(
            f"DEV_TOOL_RATE_LIMIT={raw!r} is not shaped like 'N/S' "
            f"(calls/seconds), e.g. '2/20'"
        )
    max_calls, per_seconds = match.groups()
    return RateLimitPolicy(max_calls=int(max_calls), per_seconds=float(per_seconds))


@dataclass(frozen=True)
class Settings:
    """Everything the composition root reads from the environment, typed.

    Built once, at process startup (:meth:`from_env`), and passed down --
    nothing downstream of it reads the environment again.
    """

    model: str
    """``OPENAI_MODEL``. Required; the adapter itself refuses a blank value at
    construction, and this is the same requirement enforced earlier, where a
    misconfigured deployment fails before a connection is even opened."""

    context_window: int
    """``OPENAI_CONTEXT_WINDOW``. Required, with no default -- the same
    argument :attr:`~agentic_erp_assistant.llm.gateway.LLMGateway.context_window`
    makes: a default would be one model's number silently applied to whichever
    model ``.env`` happens to name."""

    output_reserve: int = 1024
    """``OPENAI_OUTPUT_RESERVE``. Tokens held back for the reply."""

    postgres_url: str = ""
    """``POSTGRES_URL``, via
    :func:`~agentic_erp_assistant.persistence.connection.url_from_environment`
    -- blank here only ever means "use the compose default", set by
    :meth:`from_env`, never a genuinely empty string reaching a connection."""

    memory_proposer: bool = True
    """``MEMORY_PROPOSER``: ``"off"`` disables it, anything else (including
    unset) keeps it on. A deployment turns this off while deciding whether
    the model call consolidation spends is worth its cost; recall and the
    intent operations are unaffected either way."""

    memory_required_scope: str = MEMORY_REQUIRED_SCOPE

    dev_tool_rate_limit: RateLimitPolicy | None = None
    """``DEV_TOOL_RATE_LIMIT``, shaped ``"N/S"`` (e.g. ``"2/20"``). ``None``
    keeps the registry's own declared budgets."""

    dev_flaky_status: bool = False
    """``DEV_FLAKY_STATUS=1``: offer ``get_project_status_flaky`` to the
    planner *instead of* ``get_project_status``, so the retry-then-succeed
    path is reachable from a browser without a scripted fake."""

    dev_max_steps: int | None = None
    """``DEV_MAX_STEPS``. ``None`` keeps
    :data:`~agentic_erp_assistant.engine.workflow.MAX_STEPS`."""

    dev_history_turn_limit: int | None = None
    """``DEV_HISTORY_TURN_LIMIT``. ``None`` keeps
    :data:`~agentic_erp_assistant.context.history_injection.HISTORY_TURN_LIMIT`."""

    dev_trace_model_io: bool = False
    """``DEV_TRACE_MODEL_IO=1``: a turn's model calls carry their prompt and
    reply text to the trace panel live, through
    :class:`~agentic_erp_assistant.llm.inspection.ModelCallInspector`
    (``LLMGateway.inspect_io``). A different shape of dev toggle from its
    siblings above -- it does not loosen a budget a person could otherwise
    exhaust, it turns on a capability that is expensive to leave on by
    default (every evidence, history and memory block a turn's calls saw,
    restated, on every turn) and never persisted regardless. Off by
    default for that reason: the model calls a turn made, and what they
    cost, are always inspectable; what they *said* is opt-in."""

    users_path: Path = DEFAULT_USERS_PATH
    """``USERS_PATH``. Where the dev chat's actor directory is read from."""

    @classmethod
    def from_env(cls) -> "Settings":
        """Read every field from the environment.

        ``load_dotenv(override=False)`` runs once here rather than at import,
        so a test process that sets the environment after import still
        observes its own values -- the same discipline every ``from_env`` in
        this project follows.

        Raises:
            SettingsError: ``OPENAI_MODEL``, ``OPENAI_CONTEXT_WINDOW``, or a
                ``DEV_*`` toggle with a value that does not parse.
        """
        load_dotenv(override=False)

        dev_tool_rate_limit_raw = (os.environ.get("DEV_TOOL_RATE_LIMIT") or "").strip()
        users_path_raw = (os.environ.get("USERS_PATH") or "").strip()

        return cls(
            model=_require("OPENAI_MODEL"),
            context_window=_require_int("OPENAI_CONTEXT_WINDOW"),
            output_reserve=_optional_int("OPENAI_OUTPUT_RESERVE") or 1024,
            postgres_url=url_from_environment(),
            memory_proposer=(os.environ.get("MEMORY_PROPOSER") or "").strip().lower()
            != "off",
            dev_tool_rate_limit=(
                _parse_rate_limit(dev_tool_rate_limit_raw)
                if dev_tool_rate_limit_raw
                else None
            ),
            dev_flaky_status=(os.environ.get("DEV_FLAKY_STATUS") or "").strip() == "1",
            dev_max_steps=_optional_int("DEV_MAX_STEPS"),
            dev_history_turn_limit=_optional_int("DEV_HISTORY_TURN_LIMIT"),
            dev_trace_model_io=(os.environ.get("DEV_TRACE_MODEL_IO") or "").strip() == "1",
            users_path=Path(users_path_raw) if users_path_raw else DEFAULT_USERS_PATH,
        )

    def log_dev_toggles(self, log: logging.Logger = logger) -> None:
        """Warn once, loudly, per active dev toggle.

        A trace that ran under a tightened rate limit or a shortened step
        budget looks unusual read on its own; the startup log is where that
        stops being a mystery.
        """
        if self.dev_tool_rate_limit is not None:
            log.warning(
                "DEV_TOOL_RATE_LIMIT active: every read tool limited to "
                "%d call(s) per %gs",
                self.dev_tool_rate_limit.max_calls,
                self.dev_tool_rate_limit.per_seconds,
            )
        if self.dev_flaky_status:
            log.warning(
                "DEV_FLAKY_STATUS active: get_project_status_flaky is offered "
                "instead of get_project_status"
            )
        if self.dev_max_steps is not None:
            log.warning("DEV_MAX_STEPS active: step budget is %d", self.dev_max_steps)
        if self.dev_history_turn_limit is not None:
            log.warning(
                "DEV_HISTORY_TURN_LIMIT active: short-term window holds %d turn(s)",
                self.dev_history_turn_limit,
            )
        if self.dev_trace_model_io:
            log.warning(
                "DEV_TRACE_MODEL_IO active: model call prompts and replies "
                "stream live to the trace panel"
            )
        if not self.memory_proposer:
            log.warning("MEMORY_PROPOSER=off: consolidation proposes nothing")
