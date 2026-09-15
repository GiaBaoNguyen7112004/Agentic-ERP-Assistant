"""Does the planner actually route the two shapes the memory refactor was
built for? (refactor-memory-plan.md §8 step 3, live because the prompt
changes they verify are only as good as the model reading them.)

Deliberately outside ``tests/llm/`` for the same reason
``test_estimate_drift.py`` is: that directory neutralizes ``OPENAI_API_KEY``
by design. Marked ``live``; skipped without a key, like the drift test.

    uv run pytest -m live -q
"""

import os
import uuid

import pytest
from dotenv import load_dotenv

from agentic_erp_assistant.llm.adapters.openai_chat import OpenAIChatClient
from agentic_erp_assistant.llm.gateway import LLMGateway
from agentic_erp_assistant.llm.prompts import Principal
from agentic_erp_assistant.llm.telemetry import InMemoryTelemetry
from agentic_erp_assistant.reasoning.planner import Planner
from agentic_erp_assistant.state.agent_state import AgentState

from tests.memory.builders import make_turn

pytestmark = pytest.mark.live


def _skip_without_key() -> None:
    load_dotenv(override=False)
    if not (os.environ.get("OPENAI_API_KEY") or "").strip():
        pytest.skip(
            "no OPENAI_API_KEY: set it in .env to run the live routing checks"
        )


def _planner() -> tuple[Planner, OpenAIChatClient]:
    """The planner as composition/turn.py builds it for priya: real client,
    the principal block in the system role. No stream, no inspector."""
    client = OpenAIChatClient()
    gateway = LLMGateway(
        client,
        context_window=128_000,
        telemetry=InMemoryTelemetry(),
        principal=Principal(
            actor="priya",
            display_name="Priya Raman",
            role="project manager",
            project_code="atlas",
            project_name="Atlas ERP rollout",
        ),
    )
    return Planner(gateway), client


def _state(request: str, *, history=()) -> AgentState:
    return AgentState(
        request=request,
        actor="priya",
        project_code="atlas",
        trace_id=f"run-{uuid.uuid4().hex[:12]}",
        session_id="sess-live",
        history=history,
    )


def test_a_question_about_the_conversation_routes_to_answer() -> None:
    """"What was my previous question?" -- S1's shape, given a two-turn
    history. Before the conversational route this had no legal path: history
    was forbidden as a reason to answer, so the planner refused or asked for
    clarification."""
    _skip_without_key()
    planner, client = _planner()
    try:
        decision = planner.plan(
            _state(
                "what was my previous question?",
                history=(
                    make_turn(
                        trace_id="run-0",
                        request="What risks does the project have?",
                        response="Two open risks: R-1 and R-2.",
                    ),
                    make_turn(
                        trace_id="run-1",
                        request="what about sprints?",
                        response="Sprint 13 is in progress.",
                    ),
                ),
            )
        )
    finally:
        client.close()

    assert decision.route == "answer"
    assert decision.message  # the reply names what was asked


def test_a_direct_request_routes_to_its_tool_with_the_project_filled() -> None:
    """"List the risks" -- S3/S4's shape. The principal block names the
    project, so the model fills project_id=atlas itself instead of asking
    which project is meant."""
    _skip_without_key()
    planner, client = _planner()
    try:
        decision = planner.plan(_state("list the risks"))
    finally:
        client.close()

    assert decision.route == "call_tool"
    assert decision.required_tool == "list_risks"
    assert decision.tool_arguments["project_id"] == "atlas"