"""A model may nominate and nothing more, and every way its answer can be
unusable ends in remembering nothing rather than in a failed request."""

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_erp_assistant.llm.prompts import MEMORY_CONTRACT, NO_REPLY
from agentic_erp_assistant.llm.tools import (
    DEFAULT_TOOLS,
    PLANNING_TOOLS,
    ToolCallResult,
    ToolSpec,
)
from agentic_erp_assistant.memory.extractor import (
    MAX_PROPOSALS,
    PROPOSABLE_KINDS,
    PROPOSE_MEMORIES_TOOL,
    LLMMemoryProposer,
    MemoryProposal,
    MemoryProposerPort,
    ProposeMemoriesArguments,
)
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.memory import MemoryKind
from agentic_erp_assistant.state.tool_outcome import ToolOutcome

from tests.memory.builders import make_record


class FakeModel:
    """Returns a scripted choice and records what it was asked."""

    def __init__(self, result: ToolCallResult) -> None:
        self.result = result
        self.messages: list[Sequence[Any]] = []
        self.offered: list[Sequence[ToolSpec]] = []

    def call_tools(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
    ) -> ToolCallResult:
        self.messages.append(messages)
        self.offered.append(tools)
        return self.result


def proposal(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "kind": "preference",
        "key": "reply_language",
        "statement": "Prefers replies written in Vietnamese.",
        "confidence": 0.9,
    }
    fields.update(overrides)
    return fields


def called(*proposals: dict[str, object]) -> ToolCallResult:
    return ToolCallResult.from_tool_call(
        tool_name=PROPOSE_MEMORIES_TOOL.name,
        arguments={"candidates": list(proposals)},
    )


def state(**overrides: object) -> AgentState:
    fields: dict[str, object] = {
        "request": "Answer me in Vietnamese from now on, and how is M2 tracking?",
        "actor": "priya",
        "project_code": "atlas",
        "trace_id": "run-1",
        "session_id": "sess-1",
        "response": "M2 is tracking to plan.",
        "terminal": True,
    }
    fields.update(overrides)
    return AgentState(**fields)  # type: ignore[arg-type]


def propose(result: ToolCallResult, **overrides: object):
    model = FakeModel(result)
    proposer = LLMMemoryProposer(model=model, **overrides)  # type: ignore[arg-type]
    return proposer.propose(state(), required_scope="project.docs.read"), model


# --------------------------------------------------------------------------
# The tool declaration
# --------------------------------------------------------------------------


def test_the_tool_is_strict_compatible_including_the_nested_proposal() -> None:
    """ToolSpec checks every definition under $defs, not only the root -- so a
    nested model that accepted anything would fail at import."""
    schema = PROPOSE_MEMORIES_TOOL.schema

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["MemoryProposal"]["additionalProperties"] is False


def test_the_model_cannot_propose_a_projection() -> None:
    """intent and session_summary are derived from state the system holds. A
    model that could nominate one would be writing a task by assertion."""
    with pytest.raises(ValidationError, match="kind"):
        MemoryProposal(**proposal(kind="intent"))
    with pytest.raises(ValidationError, match="kind"):
        MemoryProposal(**proposal(kind="session_summary"))


def test_the_proposable_kinds_are_a_subset_of_the_stored_kinds() -> None:
    """Keeps the Literal and the tuple honest, since a Literal needs literals
    and cannot be built from the tuple."""
    from typing import get_args

    assert set(PROPOSABLE_KINDS) < set(get_args(MemoryKind))


def test_the_proposal_tool_is_never_offered_during_a_turn() -> None:
    """A model shown it mid-turn would eventually choose to remember something
    instead of answering."""
    assert PROPOSE_MEMORIES_TOOL not in DEFAULT_TOOLS
    assert PROPOSE_MEMORIES_TOOL not in PLANNING_TOOLS


def test_the_proposal_tool_does_not_mutate_erp_data() -> None:
    """It changes nothing at all: routing it through a human would ask somebody
    to approve a sentence that may never be stored."""
    assert PROPOSE_MEMORIES_TOOL.mutating is False


def test_an_empty_list_is_a_valid_call() -> None:
    """"Propose nothing" needs an unambiguous spelling."""
    assert ProposeMemoriesArguments(candidates=[]).candidates == []


# --------------------------------------------------------------------------
# The expected path
# --------------------------------------------------------------------------


def test_the_proposer_satisfies_the_port_structurally() -> None:
    assert isinstance(LLMMemoryProposer(model=FakeModel(called())), MemoryProposerPort)


def test_a_proposal_becomes_a_candidate() -> None:
    candidates, _ = propose(called(proposal()))

    assert len(candidates) == 1
    assert candidates[0].kind == "preference"
    assert candidates[0].key == "reply_language"
    assert candidates[0].statement == "Prefers replies written in Vietnamese."


def test_the_scope_comes_from_the_caller_and_not_from_the_model() -> None:
    """A proposer that could name its own scope could declassify the document it
    learned from by remembering it."""
    candidates, _ = propose(called(proposal()))

    assert candidates[0].required_scope == "project.docs.read"


def test_the_turns_own_sources_travel_with_the_candidate() -> None:
    """What makes the policy's source rule a comparison rather than a guess."""
    model = FakeModel(called(proposal()))
    turn = state().evolve(
        evidence=(
            EvidenceSnippet(source_id="doc-1", locator="3.2", text="The cutover is Thursday."),
        ),
        observations=(
            ToolOutcome(
                tool_name="list_risks",
                status="ok",
                summary="two open risks",
                source_ids=("PRJ-1",),
            ),
        ),
    )

    candidates = LLMMemoryProposer(model=model).propose(
        turn, required_scope="project.docs.read"
    )

    assert candidates[0].evidence_texts == ("The cutover is Thursday.",)
    assert candidates[0].tool_summaries == ("two open risks",)


def test_a_failed_tool_call_is_not_offered_as_a_source() -> None:
    """Comparing a statement against what a call did not return would refuse a
    memory for restating something nobody was told."""
    model = FakeModel(called(proposal()))
    turn = state().evolve(
        observations=(
            ToolOutcome(tool_name="list_risks", status="failed", error="backend down"),
        )
    )

    candidates = LLMMemoryProposer(model=model).propose(
        turn, required_scope="project.docs.read"
    )

    assert candidates[0].tool_summaries == ()


def test_an_empty_call_proposes_nothing_and_that_is_not_an_error() -> None:
    candidates, _ = propose(called())

    assert candidates == ()


# --------------------------------------------------------------------------
# What the model is shown
# --------------------------------------------------------------------------


def test_the_proposer_is_asked_in_its_own_prompt() -> None:
    _, model = propose(called())

    roles = [message["role"] for message in model.messages[0]]
    assert roles == [
        "system",
        "developer",
        "user",
        "evidence",
        "observation",
        "memory",
        "assistant",
    ]
    assert model.messages[0][1]["content"] == MEMORY_CONTRACT


def test_the_reply_reaches_the_proposer_because_how_a_turn_ended_matters() -> None:
    """A turn that refused for want of evidence established nothing."""
    _, model = propose(called())

    assert model.messages[0][6]["content"] == "M2 is tracking to plan."


def test_a_turn_with_no_reply_says_so_rather_than_omitting_the_block() -> None:
    model = FakeModel(called())

    LLMMemoryProposer(model=model).propose(
        state(response=None, terminal=False), required_scope="project.docs.read"
    )

    assert model.messages[0][6]["content"] == NO_REPLY


def test_only_the_proposal_function_is_offered() -> None:
    _, model = propose(called())

    assert list(model.offered[0]) == [PROPOSE_MEMORIES_TOOL]


def test_the_turns_own_memories_reach_the_proposer(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_memory_messages takes memories so the model can see what it would
    be restating -- an instruction to avoid duplicates given without showing
    the existing memories is one nobody could follow. The call never passed
    them; the docstring already promised it."""
    seen: dict[str, object] = {}

    def recording_build(*args: object, **kwargs: object):
        seen["memories"] = kwargs.get("memories")
        return [
            {"role": "system", "content": ""},
            {"role": "developer", "content": ""},
            {"role": "user", "content": ""},
            {"role": "evidence", "content": ""},
            {"role": "observation", "content": ""},
            {"role": "memory", "content": ""},
            {"role": "assistant", "content": ""},
        ]

    monkeypatch.setattr(
        "agentic_erp_assistant.memory.extractor.build_memory_messages",
        recording_build,
    )
    memory = make_record()

    LLMMemoryProposer(model=FakeModel(called())).propose(
        state(memories=(memory,)), required_scope="project.docs.read"
    )

    assert seen["memories"] == (memory,)


def test_the_rendered_key_reaches_the_proposers_prompt() -> None:
    """Not mocked this time: the real ``build_memory_messages`` renders each
    recalled memory's key, so the model can reuse it for a changed preference
    instead of inventing a new one (see ``memory.policy.TOPIC_OVERLAP_RATIO``
    for what happens when it does)."""
    memory = make_record(key="budget_reporting_format")
    model = FakeModel(called())

    LLMMemoryProposer(model=model).propose(
        state(memories=(memory,)), required_scope="project.docs.read"
    )

    memory_block = next(
        message["content"]
        for message in model.messages[0]
        if message["role"] == "memory"
    )
    assert "key: budget_reporting_format" in memory_block


def test_the_candidate_carries_the_turns_own_words() -> None:
    """The policy's not_established checks need them: a stated preference has
    to be found in the request, and a fact must not merely restate the reply."""
    candidates, _ = propose(called(proposal()))

    assert candidates[0].request_text == state().request
    assert candidates[0].response_text == state().response


# --------------------------------------------------------------------------
# Every unusable answer means remembering nothing, never a failed request
# --------------------------------------------------------------------------


def test_prose_instead_of_a_call_proposes_nothing() -> None:
    candidates, _ = propose(ToolCallResult.from_content("Nothing worth keeping."))

    assert candidates == ()


def test_calling_a_function_it_was_not_offered_proposes_nothing() -> None:
    candidates, _ = propose(
        ToolCallResult.from_tool_call(tool_name="create_risk", arguments={})
    )

    assert candidates == ()


def test_arguments_the_schema_refuses_propose_nothing() -> None:
    """An invented kind, a missing field, a confidence above one -- all the same
    outcome, because the turn has already answered the user."""
    candidates, _ = propose(called(proposal(kind="observation")))

    assert candidates == ()


def test_a_missing_candidates_list_proposes_nothing() -> None:
    candidates, _ = propose(
        ToolCallResult.from_tool_call(
            tool_name=PROPOSE_MEMORIES_TOOL.name, arguments={}
        )
    )

    assert candidates == ()


# --------------------------------------------------------------------------
# The bound on how much one turn may propose
# --------------------------------------------------------------------------


def test_a_turn_may_not_propose_more_than_the_cap() -> None:
    """A turn producing five durable facts has misunderstood the question, and
    without a bound every turn pays for four rejections in rows and tokens."""
    candidates, _ = propose(
        called(*(proposal(key=f"key_{n}") for n in range(MAX_PROPOSALS + 3)))
    )

    assert len(candidates) == MAX_PROPOSALS


def test_the_kept_candidates_are_the_first_ones() -> None:
    candidates, _ = propose(
        called(*(proposal(key=f"key_{n}") for n in range(MAX_PROPOSALS + 1)))
    )

    assert [c.key for c in candidates] == [f"key_{n}" for n in range(MAX_PROPOSALS)]
