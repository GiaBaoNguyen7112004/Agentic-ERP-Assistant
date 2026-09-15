"""UserDirectory: the reviewable cast the dev chat's actor switcher lists."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.composition.settings import DEFAULT_USERS_PATH
from agentic_erp_assistant.composition.users import UnknownUser, User, UserDirectory


def write_users(tmp_path: Path, users: list[dict]) -> Path:
    path = tmp_path / "users.json"
    path.write_text(json.dumps({"users": users}), encoding="utf-8")
    return path


def a_row(**overrides) -> dict:
    row = {
        "actor": "priya",
        "display_name": "Priya Raman",
        "role": "Delivery lead",
        "project_code": "atlas",
        "scopes": ["project.docs.read", "approvals.decide"],
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------
# The repo's own fixture loads and validates
# --------------------------------------------------------------------------


def test_the_repo_fixture_loads_and_validates() -> None:
    directory = UserDirectory.load(DEFAULT_USERS_PATH)

    actors = {user.actor for user in directory.all()}
    assert actors == {"priya", "wei", "tomas", "sponsor", "orion.lead", "guest"}


def test_the_fixture_lives_where_a_reviewer_would_look() -> None:
    assert DEFAULT_USERS_PATH.exists()
    assert DEFAULT_USERS_PATH.parts[-2:] == ("data", "users.json")


def test_priya_can_approve_and_guest_cannot() -> None:
    directory = UserDirectory.load(DEFAULT_USERS_PATH)

    assert directory.get("priya").can_approve
    assert not directory.get("guest").can_approve
    assert directory.get("guest").scopes == frozenset()


def test_orion_lead_is_bound_to_orion_and_holds_risk_write() -> None:
    lead = UserDirectory.load(DEFAULT_USERS_PATH).get("orion.lead")

    assert lead.project_code == "orion"
    assert "project.risk.write" in lead.scopes


# --------------------------------------------------------------------------
# Loading and lookup
# --------------------------------------------------------------------------


def test_load_reads_a_users_file(tmp_path: Path) -> None:
    path = write_users(tmp_path, [a_row()])

    directory = UserDirectory.load(path)

    assert directory.get("priya").display_name == "Priya Raman"


def test_get_an_unknown_actor_raises(tmp_path: Path) -> None:
    directory = UserDirectory.load(write_users(tmp_path, [a_row()]))

    with pytest.raises(UnknownUser):
        directory.get("nobody")


def test_a_duplicate_actor_is_rejected(tmp_path: Path) -> None:
    path = write_users(tmp_path, [a_row(), a_row(display_name="Priya Again")])

    with pytest.raises(ValueError, match="twice"):
        UserDirectory.load(path)


def test_an_empty_users_array_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="users"):
        UserDirectory.load(write_users(tmp_path, []))


def test_all_returns_every_user_in_file_order(tmp_path: Path) -> None:
    path = write_users(
        tmp_path,
        [a_row(actor="a"), a_row(actor="b"), a_row(actor="c")],
    )

    assert [user.actor for user in UserDirectory.load(path).all()] == ["a", "b", "c"]


# --------------------------------------------------------------------------
# The User model
# --------------------------------------------------------------------------


def test_scopes_default_to_empty() -> None:
    user = User.model_validate(a_row(scopes=[]))

    assert user.scopes == frozenset()
    assert not user.can_approve


def test_an_unmodelled_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        User.model_validate(a_row(is_admin=True))


def test_a_blank_actor_is_rejected() -> None:
    with pytest.raises(ValidationError):
        User.model_validate(a_row(actor=""))
