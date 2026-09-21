"""Snapshot builders: bounded, and honest about what was cut."""

from agentic_erp_assistant.llm.inspection import (
    SNAPSHOT_TEXT_LIMIT,
    snapshot_request,
    snapshot_response,
)


def test_snapshot_request_carries_role_and_content_pairs_in_order() -> None:
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hi"},
    ]
    snapshot = snapshot_request("answer", messages)

    assert snapshot.kind == "answer"
    assert snapshot.messages == (("system", "policy"), ("user", "hi"))
    assert snapshot.message_chars == (len("policy"), len("hi"))
    assert snapshot.tools == ()
    assert snapshot.tool_choice is None


def test_snapshot_request_carries_the_offered_tools_and_choice() -> None:
    snapshot = snapshot_request(
        "tools", [{"role": "user", "content": "hi"}], tools=["list_risks", "create_risk"],
        tool_choice="required", temperature=0.0,
    )

    assert snapshot.kind == "tools"
    assert snapshot.tools == ("list_risks", "create_risk")
    assert snapshot.tool_choice == "required"


def test_snapshot_request_clips_long_message_content_but_keeps_the_real_length() -> None:
    long_text = "x" * (SNAPSHOT_TEXT_LIMIT + 500)
    snapshot = snapshot_request("answer", [{"role": "evidence", "content": long_text}])

    assert len(snapshot.messages[0][1]) == SNAPSHOT_TEXT_LIMIT
    assert snapshot.message_chars[0] == SNAPSHOT_TEXT_LIMIT + 500


def test_snapshot_response_defaults_to_all_absent() -> None:
    snapshot = snapshot_response()

    assert snapshot.content is None
    assert snapshot.content_chars is None
    assert snapshot.tool_name is None
    assert snapshot.arguments is None
    assert snapshot.stop_reason is None


def test_snapshot_response_carries_content_and_its_real_length() -> None:
    snapshot = snapshot_response(content="hello", stop_reason="stop")

    assert snapshot.content == "hello"
    assert snapshot.content_chars == 5
    assert snapshot.stop_reason == "stop"


def test_snapshot_response_clips_long_content_but_keeps_the_real_length() -> None:
    long_text = "x" * (SNAPSHOT_TEXT_LIMIT + 500)
    snapshot = snapshot_response(content=long_text)

    assert len(snapshot.content) == SNAPSHOT_TEXT_LIMIT
    assert snapshot.content_chars == SNAPSHOT_TEXT_LIMIT + 500


def test_snapshot_response_carries_a_tool_call_and_copies_its_arguments() -> None:
    arguments = {"project_id": "atlas"}
    snapshot = snapshot_response(tool_name="list_risks", arguments=arguments)

    assert snapshot.tool_name == "list_risks"
    assert snapshot.arguments == arguments
    assert snapshot.arguments is not arguments  # copied, not aliased
