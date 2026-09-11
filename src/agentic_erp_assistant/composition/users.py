"""Who may use the dev chat, and as what -- read from one reviewable file.

There is no login. The brief asks for a dev-only actor switch, and CLAUDE.md's
approval rule does not care how an actor's identity reached the request, only
that it did and that it is recorded -- so the switch is a ``<select>`` in the
UI, and this module is where the six choices behind it come from. Synthetic
people over the same synthetic corpus every other fixture in this repo uses,
scopes drawn from the same vocabulary
:mod:`agentic_erp_assistant.tools.registry` and
:mod:`agentic_erp_assistant.rag.manifest` already speak.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["UnknownUser", "User", "UserDirectory"]


class UnknownUser(KeyError):
    """The directory was asked for an actor it does not have.

    A ``KeyError`` because that is what it is, and named for the same reason
    :class:`~agentic_erp_assistant.tools.registry.UnknownTool` is: the caller
    -- a request naming an actor the switcher never offered -- gets a typed
    404, not an unlabelled lookup failure.
    """


class User(BaseModel):
    """One actor the dev chat may act as.

    Frozen and ``extra="forbid"``: this file is reviewed, and a field
    smuggled into one row is a scope granted without anyone reading it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor: str = Field(min_length=1)
    """The identity every trace, audit row, and citation check is stamped
    with -- the same string that flows into ``AgentState.actor``."""

    display_name: str = Field(min_length=1)
    """What the actor switcher shows."""

    role: str = Field(min_length=1)
    """One line of context for the switcher -- never consulted for policy."""

    project_code: str = Field(min_length=1)
    """The project this actor is bound to for the turn. See ADR 0017."""

    scopes: frozenset[str] = frozenset()
    """What this actor is entitled to. Empty is a real answer -- ``guest``
    holds none."""

    @property
    def can_approve(self) -> bool:
        """Whether the web layer's approval endpoint accepts this actor as a
        decider. A scope like any other, checked only there -- the engine
        itself does not gate who may call ``resume_approval``, the same way
        it does not gate who may call ``handle``; that boundary belongs to
        whatever is exposing the engine, which for now is only this one."""
        return "approvals.decide" in self.scopes


class UserDirectory:
    """Every actor the dev chat may switch to, by name.

    Loaded once, at composition time, the same as
    :class:`~agentic_erp_assistant.rag.manifest.SourceDocument` and the ERP
    fixture -- a reviewable file, not a database table, because the cast for
    a graded demo is exactly the kind of thing that belongs in review
    material.
    """

    def __init__(self, users: tuple[User, ...]) -> None:
        by_actor: dict[str, User] = {}
        for user in users:
            if user.actor in by_actor:
                raise ValueError(
                    f"actor {user.actor!r} is declared twice in the users "
                    f"file; one of the two rows would silently win"
                )
            by_actor[user.actor] = user
        self._by_actor = by_actor

    @classmethod
    def load(cls, path: Path) -> "UserDirectory":
        """Read and validate the users file.

        Raises:
            OSError: The file cannot be read.
            ValueError: The JSON is malformed, a row fails validation, or an
                actor is declared twice.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("users") if isinstance(raw, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: expected a non-empty 'users' array")
        return cls(tuple(User.model_validate(row) for row in rows))

    def get(self, actor: str) -> User:
        """The user record for ``actor``.

        Raises:
            UnknownUser: No actor by that name is in the directory.
        """
        try:
            return self._by_actor[actor]
        except KeyError:
            raise UnknownUser(actor) from None

    def all(self) -> tuple[User, ...]:
        """Every user, in file order -- what the actor switcher lists."""
        return tuple(self._by_actor.values())
