"""eval/routing.py against a scripted client: the harness's own mechanics,
never a live model. Whether any contract actually routes better is what
`scripts/run_routing_comparison.py` measures for real (ADR 0020).
"""

from agentic_erp_assistant.composition.settings import DEFAULT_USERS_PATH
from agentic_erp_assistant.composition.users import UserDirectory
from agentic_erp_assistant.eval.routing import run_comparison
from agentic_erp_assistant.eval.routing_cases import DEFAULT_CASES, RoutingCase
from agentic_erp_assistant.eval.routing_prompts import RouterPrompt
from agentic_erp_assistant.llm.gateway import LLMGateway
from agentic_erp_assistant.llm.ports import TransientProviderError
from agentic_erp_assistant.llm.tools import ToolCallResult

USERS = UserDirectory.load(DEFAULT_USERS_PATH)


def called(name: str, **arguments) -> ToolCallResult:
    return ToolCallResult.from_tool_call(tool_name=name, arguments=arguments)


def answered(text: str) -> ToolCallResult:
    return ToolCallResult.from_content(text)


def declared(needs: list[str], document_query: str | None = None) -> ToolCallResult:
    return ToolCallResult.from_tool_call(
        tool_name="declare_reply_contract",
        arguments={"needs": needs, "document_query": document_query},
    )


EMPTY_DECLARATION = declared([])
"""What a case with no scripted declaration gets -- a real, readable answer
of "nothing needed", never an unreadable one that would silently fall back
to EMPTY_CONTRACT and mask a test that forgot to script one."""


class ScriptedRoutingClient:
    """Keys its answer off the question text -- the same request reaches
    the client under every prompt, so one response map serves all three.

    Declaration calls (``tool_choice="required"``, offering only
    ``declare_reply_contract``) are answered from a *separate* map and
    tracked in their own ``declaration_calls`` list -- ``run_comparison``
    makes one declaration call per (case, repeat), independent of the
    routing prompt, so ``calls`` (routing only) keeps its one-entry-per-
    prompt shape existing tests already assert on.
    """

    model_name = "fake-model-1"

    def __init__(
        self,
        responses: dict[str, ToolCallResult],
        *,
        declarations: dict[str, ToolCallResult] | None = None,
        raise_for: str | None = None,
    ):
        self.responses = responses
        self.declarations = declarations or {}
        self.raise_for = raise_for
        self.calls: list[tuple[str, str]] = []  # (question, developer block)
        self.declaration_calls: list[tuple[str, str]] = []

    def call_with_tools(self, messages, *, tools, temperature, on_delta=None, tool_choice="auto"):
        question = next(m["content"] for m in messages if m["role"] == "user")
        developer = next(m["content"] for m in messages if m["role"] == "developer")
        if self.raise_for is not None and question == self.raise_for:
            raise TransientProviderError("scripted transport failure")
        if len(tools) == 1 and tools[0].name == "declare_reply_contract":
            self.declaration_calls.append((question, developer))
            return self.declarations.get(question, EMPTY_DECLARATION)
        self.calls.append((question, developer))
        return self.responses[question]


WELL_BEHAVED_RESPONSES = {
    "Why is milestone M2 late and by how much?": called(
        "search_project_documents", query="M2 delay"
    ),
    "What is the status of milestone M2?": called("get_project_status", milestone_id="M2"),
    "What is the weather forecast in Hanoi next week?": called(
        "refuse", reason="outside project delivery"
    ),
    "How is the sprint going?": called(
        "ask_clarification", question="Which sprint do you mean?"
    ),
    "Record a high severity risk on atlas: hypercare staffing is not "
    "confirmed for the M2 cutover.": called("list_risks", project_id="atlas"),
    "Record a medium risk on atlas: warehouse depot hardware refresh is "
    "unfunded.": called("list_risks", project_id="atlas"),
}


def gateway_factory_for(client) -> "callable":
    def factory(contract: str) -> LLMGateway:
        return LLMGateway(
            client,
            context_window=128_000,
            planner_contract=contract,
            sleep=lambda seconds: None,
        )

    return factory


def test_three_prompts_six_cases_one_repeat_produce_eighteen_rows() -> None:
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES)
    prompts = (
        RouterPrompt("p1", "contract one"),
        RouterPrompt("p2", "contract two"),
        RouterPrompt("p3", "contract three"),
    )

    report = run_comparison(
        prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1
    )

    assert len(report.results) == 18


def test_every_row_carries_chosen_accepted_matched_and_hallucinated() -> None:
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES)
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    for row in report.results:
        assert row.chosen_route is not None
        assert row.accepted_first_routes
        assert isinstance(row.matched, bool)
        assert isinstance(row.hallucinated_tool, bool)


def test_the_contract_the_client_receives_differs_per_prompt() -> None:
    """Wired, not ignored: three prompts produce three distinct developer
    blocks on the wire, in the order the prompts were given."""
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES)
    prompts = (
        RouterPrompt("p1", "contract one"),
        RouterPrompt("p2", "contract two"),
        RouterPrompt("p3", "contract three"),
    )

    run_comparison(prompts, DEFAULT_CASES[:1], gateway_factory_for(client), USERS, repeats=1)

    developer_blocks = [developer for _question, developer in client.calls]
    assert developer_blocks == ["contract one", "contract two", "contract three"]


def test_list_risks_matches_the_write_cases_but_not_documents() -> None:
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES)
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    by_case = {row.case_id: row for row in report.results}
    assert by_case["A1"].matched is True
    assert by_case["A1"].chosen_tool == "list_risks"
    assert by_case["A9"].matched is True
    assert by_case["R1"].matched is True  # documents, not list_risks -- a different case entirely


def test_a_client_that_raises_on_one_pair_yields_one_error_row_and_seventeen_others() -> None:
    raising_question = "How is the sprint going?"
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES, raise_for=raising_question)
    prompts = (
        RouterPrompt("p1", "contract one"),
        RouterPrompt("p2", "contract two"),
        RouterPrompt("p3", "contract three"),
    )

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    errored = [row for row in report.results if row.error is not None]
    clean = [row for row in report.results if row.error is None]
    assert len(errored) == 3  # one per prompt: T9/1 raises under every contract
    assert len(clean) == 15
    assert all(row.matched is False for row in errored)
    assert all("scripted transport failure" in row.error for row in errored)


def test_a_hallucinated_tool_call_is_flagged_and_never_matched() -> None:
    case = RoutingCase(
        case_id="X1",
        actor="priya",
        request="Do the impossible thing.",
        accepted_first_routes=frozenset({("refuse", None)}),
        note="a tool that does not exist",
    )
    client = ScriptedRoutingClient({"Do the impossible thing.": called("delete_project")})
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, [case], gateway_factory_for(client), USERS, repeats=1)

    (row,) = report.results
    assert row.hallucinated_tool is True
    assert row.matched is False
    assert row.chosen_route == "fail"


def test_the_report_round_trips_through_json() -> None:
    import json

    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES)
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    dumped = json.dumps(report.to_dict())
    reloaded = json.loads(dumped)
    assert len(reloaded["results"]) == 6
    assert reloaded["summary"][0]["prompt"] == "p1"
    assert reloaded["summary"][0]["total"] == 6


def test_summary_reports_match_rate_and_cost_per_prompt() -> None:
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES)
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    (summary,) = report.summary()
    assert summary.prompt == "p1"
    assert summary.total == 6
    assert 0.0 <= summary.match_rate <= 1.0
    assert summary.contract_length == len("contract one")


# --------------------------------------------------------------------------
# Declarations: one per (case, repeat), independent of the routing prompt
# --------------------------------------------------------------------------

WELL_BEHAVED_DECLARATIONS = {
    "Why is milestone M2 late and by how much?": declared(
        ["document_passage", "erp_field"], "M2 delay"
    ),
    "What is the status of milestone M2?": declared(["erp_field"]),
    "How is the sprint going?": declared(["erp_field"]),
}


def test_six_cases_one_repeat_produce_six_declaration_rows_beside_eighteen_routing_rows() -> None:
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES, declarations=WELL_BEHAVED_DECLARATIONS)
    prompts = (
        RouterPrompt("p1", "contract one"),
        RouterPrompt("p2", "contract two"),
        RouterPrompt("p3", "contract three"),
    )

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    assert len(report.results) == 18
    assert len(report.declarations) == 6


def test_a_declaration_is_not_repeated_per_prompt() -> None:
    """The contract does not enter the declaration call at all -- one
    declaration serves every prompt's routing pass for the same case."""
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES, declarations=WELL_BEHAVED_DECLARATIONS)
    prompts = (
        RouterPrompt("p1", "contract one"),
        RouterPrompt("p2", "contract two"),
        RouterPrompt("p3", "contract three"),
    )

    run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    assert len(client.declaration_calls) == 6


def test_a_matching_declaration_is_scored_by_set_equality() -> None:
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES, declarations=WELL_BEHAVED_DECLARATIONS)
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    by_case = {row.case_id: row for row in report.declarations}
    assert by_case["R1"].matched is True
    assert by_case["R1"].declared_needs == ("document_passage", "erp_field")
    assert by_case["R1"].document_query == "M2 delay"
    assert by_case["T1"].matched is True


def test_an_over_declaration_does_not_match_even_as_a_superset() -> None:
    """Set equality, not superset: a contract that asks for a passage a
    field question does not need fails exactly as T1 already catches on
    the routing side."""
    over_declaring = dict(WELL_BEHAVED_DECLARATIONS)
    over_declaring["What is the status of milestone M2?"] = declared(
        ["erp_field", "document_passage"], "milestone M2"
    )
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES, declarations=over_declaring)
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    (row,) = [r for r in report.declarations if r.case_id == "T1"]
    assert row.matched is False


def test_unscored_cases_are_excluded_from_the_declaration_match_rate() -> None:
    """R10, A1 and A9 declare expected_needs=None -- any declaration is a
    legitimate answer, so they are neither matched nor unmatched."""
    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES, declarations=WELL_BEHAVED_DECLARATIONS)
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    unscored = {row.case_id for row in report.declarations if row.expected_needs is None}
    assert unscored == {"R10", "A1", "A9"}
    assert all(
        row.matched is None for row in report.declarations if row.case_id in unscored
    )
    # Three scored cases (R1, T1, T9/1), all matching the well-behaved map.
    assert report.declaration_match_rate() == 1.0


def test_a_declaration_that_raises_yields_an_error_row() -> None:
    client = ScriptedRoutingClient(
        WELL_BEHAVED_RESPONSES,
        declarations=WELL_BEHAVED_DECLARATIONS,
        raise_for="How is the sprint going?",
    )
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    (row,) = [r for r in report.declarations if r.case_id == "T9/1"]
    assert row.declared_needs is None
    assert row.matched is None
    assert "scripted transport failure" in row.error


def test_the_report_round_trips_through_json_with_declarations() -> None:
    import json

    client = ScriptedRoutingClient(WELL_BEHAVED_RESPONSES, declarations=WELL_BEHAVED_DECLARATIONS)
    prompts = (RouterPrompt("p1", "contract one"),)

    report = run_comparison(prompts, DEFAULT_CASES, gateway_factory_for(client), USERS, repeats=1)

    reloaded = json.loads(json.dumps(report.to_dict()))
    assert len(reloaded["declarations"]) == 6
    assert reloaded["declaration_match_rate"] == 1.0
