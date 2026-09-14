"""The event vocabulary: what an event may contain, and how a reply's
Sources: trailer becomes typed citations."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from agentic_erp_assistant.engine.nodes import _with_sources
from agentic_erp_assistant.state.agent_state import AgentState
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.web.protocol import (
    AnswerEvent,
    ApprovalRequiredEvent,
    CitationOut,
    clip_text,
    ContextEvent,
    ContractOut,
    encode_sse,
    ErrorEvent,
    EVENT_TYPES,
    parse_citations,
    ResetEvent,
    ServerEvent,
    StepEvent,
    TokenEvent,
    TraceRow,
    TurnFinishedEvent,
    TurnStartedEvent,
)

SCOPES = frozenset({"project.status.read"})

_MODELS_BY_TYPE = {
    "turn_started": TurnStartedEvent,
    "context": ContextEvent,
    "trace": TraceRow,
    "step": StepEvent,
    "token": TokenEvent,
    "reset": ResetEvent,
    "approval_required": ApprovalRequiredEvent,
    "answer": AnswerEvent,
    "turn_finished": TurnFinishedEvent,
    "error": ErrorEvent,
}


def state(**changes) -> AgentState:
    base = AgentState(
        request="How is M2 tracking?", actor="priya", project_code="atlas",
        trace_id="run-1", scopes=SCOPES,
    )
    return base.evolve(**changes) if changes else base


# --------------------------------------------------------------------------
# EVENT_TYPES matches the union, exactly
# --------------------------------------------------------------------------


def test_event_types_names_every_member_and_no_extra_one() -> None:
    assert set(EVENT_TYPES) == set(_MODELS_BY_TYPE)
    assert len(EVENT_TYPES) == len(set(EVENT_TYPES)), "no duplicates"


@pytest.mark.parametrize("event_type,model", list(_MODELS_BY_TYPE.items()))
def test_each_models_own_type_default_matches_its_entry(event_type, model) -> None:
    assert model.model_fields["type"].default == event_type


# --------------------------------------------------------------------------
# encode_sse
# --------------------------------------------------------------------------


def test_encode_sse_shape() -> None:
    encoded = encode_sse(TokenEvent(text="Hello"))

    assert encoded == b'event: token\ndata: {"type":"token","text":"Hello"}\n\n'


def test_encode_sse_output_round_trips_through_the_discriminated_union() -> None:
    encoded = encode_sse(TurnStartedEvent(trace_id="run-1", session_id="s1", actor="priya", resumed=True))
    _, _, data_line = encoded.decode("utf-8").partition("data: ")
    payload = json.loads(data_line.strip())

    adapter: TypeAdapter[ServerEvent] = TypeAdapter(ServerEvent)
    parsed = adapter.validate_python(payload)

    assert isinstance(parsed, TurnStartedEvent)
    assert parsed.resumed is True


# --------------------------------------------------------------------------
# extra="forbid" on every event
# --------------------------------------------------------------------------


def test_an_unmodelled_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TokenEvent.model_validate({"type": "token", "text": "hi", "extra": 1})


# --------------------------------------------------------------------------
# parse_citations
# --------------------------------------------------------------------------


def test_no_trailer_leaves_the_response_unchanged() -> None:
    body, citations = parse_citations("I cannot help with that.", state())

    assert body == "I cannot help with that."
    assert citations == ()


def test_a_none_response_stays_none() -> None:
    body, citations = parse_citations(None, state())

    assert body is None
    assert citations == ()


def test_a_document_citation_is_kept_only_when_its_id_was_retrieved() -> None:
    snippet = EvidenceSnippet(source_id="m2-status.md", locator="p.2", text="M2 slipped.")
    response = _with_sources("M2 slipped two weeks.", ["[m2-status.md#p.2]"])

    body, citations = parse_citations(response, state(evidence=(snippet,)))

    assert body == "M2 slipped two weeks."
    assert citations == (
        CitationOut(
            source_id="m2-status.md", locator="p.2", tag="[m2-status.md#p.2]", kind="document"
        ),
    )


def test_a_bracketed_id_not_in_the_retrieved_evidence_falls_back_to_erp() -> None:
    """Should never happen for a real GroundedAnswer -- _ungrounded() already
    refuses it -- but the parser does not trust the shape alone."""
    response = _with_sources("An answer.", ["[ghost-report.md#p.9]"])

    body, citations = parse_citations(response, state())  # no evidence at all

    assert citations[0].kind == "erp"
    assert citations[0].source_id == "[ghost-report.md#p.9]"
    assert citations[0].locator is None


def test_an_erp_identifier_has_no_brackets_and_no_locator() -> None:
    response = _with_sources("Recorded R-3.", ["risk-r-3", "project-atlas"])

    body, citations = parse_citations(response, state())

    assert [c.source_id for c in citations] == ["risk-r-3", "project-atlas"]
    assert all(c.kind == "erp" and c.locator is None for c in citations)


def test_a_locator_containing_a_space_is_preserved_whole() -> None:
    snippet = EvidenceSnippet(source_id="risk-register", locator="row R-2", text="x")
    response = _with_sources("A risk is open.", ["[risk-register#row R-2]"])

    body, citations = parse_citations(response, state(evidence=(snippet,)))

    assert citations[0].locator == "row R-2"


def test_mixed_document_and_erp_citations_in_one_trailer() -> None:
    snippet = EvidenceSnippet(source_id="m2-status.md", locator="p.2", text="x")
    response = _with_sources(
        "Because of a documented issue, R-3 was recorded.",
        ["[m2-status.md#p.2]", "risk-r-3"],
    )

    body, citations = parse_citations(response, state(evidence=(snippet,)))

    assert [c.kind for c in citations] == ["document", "erp"]


# --------------------------------------------------------------------------
# clip_text
# --------------------------------------------------------------------------


def test_clip_text_leaves_short_text_unmarked() -> None:
    out = clip_text("a short passage")

    assert out.text == "a short passage"
    assert out.truncated is False
    assert out.chars == len("a short passage")


def test_clip_text_marks_and_cuts_long_text() -> None:
    text = "x" * 10
    out = clip_text(text, limit=4)

    assert out.text == "xxxx"
    assert out.truncated is True
    assert out.chars == 10, "the original length, not the clipped one"


def test_clip_text_at_exactly_the_limit_is_not_truncated() -> None:
    out = clip_text("abcd", limit=4)

    assert out.truncated is False
    assert out.text == "abcd"


# --------------------------------------------------------------------------
# ContextEvent
# --------------------------------------------------------------------------


def test_context_event_accepts_no_history_no_memory_and_an_unchecked_contract() -> None:
    event = ContextEvent(
        request="hi", history=(), memories=(), contract=None, model_calls=(),
        state=state(),
    )

    assert event.history == ()
    assert event.memories == ()
    assert event.contract is None
    assert event.model_calls == ()


def test_context_event_carries_a_declared_contract_with_no_needs() -> None:
    event = ContextEvent(
        request="hi", history=(), memories=(),
        contract=ContractOut(needs=(), document_query=None),
        model_calls=(),
        state=state(),
    )

    assert event.contract is not None
    assert event.contract.needs == ()


def test_context_event_requires_a_state() -> None:
    with pytest.raises(ValidationError):
        ContextEvent(request="hi", history=(), memories=(), contract=None, model_calls=())


def test_step_event_carries_the_state_and_round_trips_through_json() -> None:
    turn_state = state(
        tool_name="get_project_status",
        tool_arguments={"milestone_id": "M2"},
        evidence=(EvidenceSnippet(source_id="doc-1", locator="p.1", text="hi"),),
    )
    event = StepEvent(
        route="call_tool",
        tool_name="get_project_status",
        tool_arguments={"milestone_id": "M2"},
        tool_mutating=False,
        approval="not_required",
        step_count=1,
        terminal=False,
        node="think",
        elapsed_ms=1.0,
        evidence=None,
        observations=(),
        response=None,
        failure="none",
        error_detail=None,
        draft=None,
        redirected_needs=(),
        retry_count=0,
        retrieval=None,
        model_calls=(),
        state=turn_state,
    )

    round_tripped = AgentState.model_validate(
        json.loads(event.model_dump_json())["state"]
    )
    assert round_tripped == turn_state
