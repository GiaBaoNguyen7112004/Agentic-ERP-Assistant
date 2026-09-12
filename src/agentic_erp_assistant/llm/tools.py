"""What a tool is, before any provider is involved.

A provider's function-calling API wants a name, a description and a JSON Schema
for the arguments. That is not enough to run a tool safely, so the declaration
kept here carries one more thing the wire format has no slot for: whether the
call *mutates* anything. The router reads that flag to decide whether a call has
to stop and wait for a human, and the flag has to live with the declaration --
a separate list of "the dangerous ones" maintained somewhere else is a list that
will be wrong exactly once.

So a :class:`ToolSpec` is provider-neutral and slightly richer than any single
vendor's schema, and the adapter renders it down to the vendor's shape. The
alternative -- writing OpenAI's ``{"type": "function", ...}`` dict directly here
-- would put the wire format in the core, leave nowhere for ``mutating`` to
live, and make the tool registry unusable by a second provider or by a direct
in-process execution path.

The argument schema is emitted from a pydantic model rather than hand-written,
for the reason ``schemas.py`` gives: the shape asked for and the shape accepted
have to be one artifact. Here it buys something extra -- OpenAI's strict
function calling demands that every property be required and that
``additionalProperties`` be false, and :class:`ToolSpec` refuses to be
constructed with a schema that is not strict-compatible. A tool that cannot be
sent strictly fails at import, not in production, where the failure mode is a
model inventing an argument nobody declared.
"""

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_erp_assistant.state.reply_contract import ReplyNeed

__all__ = [
    "ASK_CLARIFICATION_TOOL",
    "AskClarificationArguments",
    "BudgetSummaryArguments",
    "CONTROL_TOOLS",
    "CREATE_RISK_TOOL",
    "CreateRiskArguments",
    "DECLARE_REPLY_CONTRACT_TOOL",
    "DeclareReplyContractArguments",
    "DEFAULT_TOOLS",
    "GET_BUDGET_SUMMARY_TOOL",
    "GET_PROJECT_STATUS_FLAKY_TOOL",
    "GET_PROJECT_STATUS_TOOL",
    "GET_SPRINT_PROGRESS_TOOL",
    "LIST_RISKS_TOOL",
    "ListRisksArguments",
    "PLANNING_TOOLS",
    "ProjectStatusArguments",
    "REFUSE_TOOL",
    "RefuseArguments",
    "SEARCH_PROJECT_DOCUMENTS_TOOL",
    "SearchProjectDocumentsArguments",
    "SprintProgressArguments",
    "StrictArguments",
    "ToolCallResult",
    "ToolSpec",
]


class StrictArguments(BaseModel):
    """Base for every tool's argument model.

    ``extra="forbid"`` renders as ``additionalProperties: false``, which is both
    half of what strict function calling requires and the check that rejects an
    argument the model invented. ``frozen`` because arguments that have been
    validated and then handed to a router must not be edited on the way -- an
    approver who read "move SPR-14 to done" has to be approving the call that
    actually runs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectStatusArguments(StrictArguments):
    """Arguments for ``get_project_status``.

    One required field, no defaults. That is deliberate rather than minimal:
    strict function calling requires every property to appear in ``required``,
    so an optional argument would have to be modelled as required-and-nullable.
    Worth knowing when the second tool arrives -- and worth contrasting with
    ``GroundedAnswer``, whose defaults are why ``response_format`` goes out
    non-strict.
    """

    milestone_id: str = Field(
        min_length=1,
        description="The milestone to report on, e.g. 'M2'.",
    )


def _object_schemas(schema: Mapping[str, Any]):
    """Yield the root schema and every definition under it.

    Nested models land in ``$defs``, and strict mode judges each of them
    separately, so a check that only looked at the root would pass a tool whose
    nested object silently accepts anything.
    """
    yield schema
    for definition in (schema.get("$defs") or {}).values():
        if isinstance(definition, dict):
            yield definition


def _require_strict_compatible(name: str, schema: Mapping[str, Any]) -> None:
    """Reject a schema OpenAI's strict mode would refuse, at construction time."""
    for index, part in enumerate(_object_schemas(schema)):
        if part.get("type") != "object":
            continue

        where = "arguments" if index == 0 else part.get("title", "a nested object")
        if part.get("additionalProperties") is not False:
            raise ValueError(
                f"tool {name!r}: {where} must set additionalProperties=false "
                f"(inherit from StrictArguments)"
            )

        properties = sorted(part.get("properties") or {})
        required = sorted(part.get("required") or [])
        if properties != required:
            missing = sorted(set(properties) - set(required))
            raise ValueError(
                f"tool {name!r}: strict function calling requires every "
                f"property to be required; {where} leaves {missing} optional. "
                f"Model an optional argument as required-and-nullable instead."
            )


@dataclass(frozen=True)
class ToolSpec:
    """One tool the model may be offered, declared once for every consumer.

    The adapter renders this to a vendor's function schema; the router reads
    ``mutating`` to decide whether the call needs approval; the executor uses
    ``arguments`` to validate what came back. All three read the same object,
    which is the point.
    """

    name: str
    """The identifier the model calls and the trace records."""

    description: str
    """What the tool does, written for the model. This is the only thing
    steering it toward or away from the tool, so it is prompt text, not a
    docstring afterthought."""

    arguments: type[BaseModel]
    """The argument contract. Must be strict-compatible; see the module doc."""

    mutating: bool
    """Whether calling this changes ERP data.

    The flag the wire format has no room for, and the reason this class exists.
    A true value means the router must route the call through an approval and
    record the decision before anything executes.
    """

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a tool needs a name")
        if not self.description.strip():
            raise ValueError(f"tool {self.name!r} needs a description")
        _require_strict_compatible(self.name, self.schema)

    @property
    def schema(self) -> dict[str, Any]:
        """The JSON Schema for the arguments, emitted from the model."""
        return self.arguments.model_json_schema()

    def validate_arguments(self, raw: Mapping[str, Any]) -> BaseModel:
        """Validate parsed arguments against the declaration.

        Deliberately *not* called by the adapter. An adapter that validated
        would have to know the tool registry, and a failure there would be
        indistinguishable from a transport problem. Here the failure is a
        ``ValidationError``: the contract was broken, resampling will not fix
        it, and :func:`~agentic_erp_assistant.llm.retry.retry_with_backoff`
        will not retry it.

        Raises:
            ValidationError: The arguments do not satisfy the declaration --
                an unknown key, a missing one, or the wrong type.
        """
        return self.arguments.model_validate(dict(raw))


class SprintProgressArguments(StrictArguments):
    """Arguments for ``get_sprint_progress``."""

    sprint_id: str = Field(
        min_length=1,
        description="The sprint to report on, e.g. 'SPR-13'.",
    )


class BudgetSummaryArguments(StrictArguments):
    """Arguments for ``get_budget_summary``.

    ``include_forecast`` is required rather than defaulted, which looks odd for
    a flag until you remember why: strict function calling demands every
    property appear in ``required``. A flag with a default would have to be
    modelled as required-and-nullable, which is a worse contract than simply
    making the caller state what it wants.
    """

    project_id: str = Field(
        min_length=1,
        description="The project to report on, e.g. 'atlas'.",
    )
    include_forecast: bool = Field(
        description=(
            "Whether to include the forecast at completion alongside approved "
            "and spent amounts."
        ),
    )


class ListRisksArguments(StrictArguments):
    """Arguments for ``list_risks``."""

    project_id: str = Field(
        min_length=1,
        description="The project whose open risks to list, e.g. 'atlas'.",
    )


class SearchProjectDocumentsArguments(StrictArguments):
    """Arguments for ``search_project_documents``."""

    query: str = Field(
        min_length=1,
        description=(
            "What to look for in the project documents, in the user's own "
            "terms. Prefer the words the question used over a paraphrase."
        ),
    )


class AskClarificationArguments(StrictArguments):
    """Arguments for ``ask_clarification``."""

    question: str = Field(
        min_length=1,
        description=(
            "The single question to put back to the user, phrased so that one "
            "short answer unblocks the request."
        ),
    )


class RefuseArguments(StrictArguments):
    """Arguments for ``refuse``."""

    reason: str = Field(
        min_length=1,
        description=(
            "Why this request will not be carried out, in one sentence the "
            "user will read."
        ),
    )


class CreateRiskArguments(StrictArguments):
    """Arguments for ``create_risk`` -- the one tool that changes anything.

    ``severity`` is a Literal, not a string. The model picks this value, and a
    free string would let it invent a severity nobody can sort, filter or
    escalate on, discovered only when a report tried to group by it.
    """

    project_id: str = Field(
        min_length=1,
        description="The project to record the risk against, e.g. 'atlas'.",
    )
    title: str = Field(
        min_length=1,
        description="One line describing the risk, as it will be stored.",
    )
    severity: Literal["low", "medium", "high"] = Field(
        description="How serious the risk is.",
    )


GET_PROJECT_STATUS_TOOL = ToolSpec(
    name="get_project_status",
    description=(
        "Look up the delivery status of one milestone: schedule, budget "
        "consumed, and open risks. Read-only. Use it whenever the user asks "
        "about a specific milestone by identifier."
    ),
    arguments=ProjectStatusArguments,
    mutating=False,
)
"""The first real tool. Read-only, so it needs no approval -- which is a fact
recorded in the declaration rather than an assumption made at the call site."""


GET_PROJECT_STATUS_FLAKY_TOOL = ToolSpec(
    name="get_project_status_flaky",
    description=(
        "Identical to get_project_status, but its backend fails intermittently. "
        "Present so the retry budget can be exercised end to end."
    ),
    arguments=ProjectStatusArguments,
    mutating=False,
)
"""A deliberately unreliable twin, and deliberately not in :data:`DEFAULT_TOOLS`.

It exists to be executed, not to be offered: a model shown two tools that do
the same thing would sometimes pick the broken one, and the trace would record
a retry nobody asked for. Which is the distinction between the registry and
this tuple -- the registry holds everything the runtime can run, the tuple
holds what the model is invited to choose from.
"""


GET_SPRINT_PROGRESS_TOOL = ToolSpec(
    name="get_sprint_progress",
    description=(
        "Report one sprint's committed and completed points and days "
        "remaining. Read-only. Use it for questions about sprint burn-down or "
        "whether a sprint will land."
    ),
    arguments=SprintProgressArguments,
    mutating=False,
)


GET_BUDGET_SUMMARY_TOOL = ToolSpec(
    name="get_budget_summary",
    description=(
        "Report a project's approved budget, amount spent, and optionally the "
        "forecast at completion. Read-only. Use it for questions about money."
    ),
    arguments=BudgetSummaryArguments,
    mutating=False,
)


LIST_RISKS_TOOL = ToolSpec(
    name="list_risks",
    description=(
        "List the open risks recorded against a project, with their severity. "
        "Read-only. Use it before answering questions about what could go "
        "wrong, and before proposing a new risk that may already exist."
    ),
    arguments=ListRisksArguments,
    mutating=False,
)


CREATE_RISK_TOOL = ToolSpec(
    name="create_risk",
    description=(
        "Record a new risk against a project. This changes ERP data and "
        "requires a human to approve it first. Use it only when the user has "
        "asked for a risk to be recorded, and check list_risks first so an "
        "existing risk is not duplicated."
    ),
    arguments=CreateRiskArguments,
    mutating=True,
)
"""The only mutating tool. ``mutating=True`` is what the transition guard and
the gateway both read; the sentence about approval in the description is for
the model, and is not what enforces anything."""


SEARCH_PROJECT_DOCUMENTS_TOOL = ToolSpec(
    name="search_project_documents",
    description=(
        "Search the project documents -- status reports, meeting notes, "
        "contracts -- and return the passages that answer a question, each "
        "with the source it came from. Read-only. Use it for anything the ERP "
        "tools do not hold as a field: decisions, commitments, explanations, "
        "and any question whose answer has to be quoted rather than looked up."
    ),
    arguments=SearchProjectDocumentsArguments,
    mutating=False,
)
"""Retrieval, offered exactly like any other tool.

Which is the point: the model chooses to search through the same mechanism it
chooses to call ``list_risks``, so there is one decision channel and no second
rule about when retrieval happens. What differs is where the choice is carried
out -- this one is executed by the retriever port and produces typed passages
with locators, not a
:class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome` summary bound
for an audit row. Running it through the tool gateway would flatten those
passages to a string and a list of ids before the prompt was built, and a
citation without a locator is not resolvable. Uniform decision, different
execution shape.
"""


ASK_CLARIFICATION_TOOL = ToolSpec(
    name="ask_clarification",
    description=(
        "Ask the user one question instead of answering. Use it when the "
        "request does not name what it is about -- no milestone, no project, "
        "no sprint -- and guessing would produce a confident answer about the "
        "wrong thing."
    ),
    arguments=AskClarificationArguments,
    mutating=False,
)


REFUSE_TOOL = ToolSpec(
    name="refuse",
    description=(
        "Decline the request. Use it when what is asked falls outside project "
        "delivery operations, or when no available tool and no project "
        "document could support an answer -- never as a way to avoid a hard "
        "lookup."
    ),
    arguments=RefuseArguments,
    mutating=False,
)


class DeclareReplyContractArguments(StrictArguments):
    """Arguments for ``declare_reply_contract`` (ADR 0021).

    ``document_query`` is required-and-nullable rather than optional, for the
    reason ``BudgetSummaryArguments.include_forecast`` already states: strict
    function calling requires every property to appear in ``required``, so an
    argument that only sometimes applies has to be modelled as present-and-
    possibly-null rather than absent. Whether the two fields actually agree
    with each other (a query only when ``document_passage`` is among
    ``needs``) is not checked here -- that is
    :class:`~agentic_erp_assistant.state.reply_contract.ReplyContract`'s own
    rule, applied once, by
    :meth:`~agentic_erp_assistant.reasoning.planner.Planner.declare` when it
    builds one from these.
    """

    needs: list[ReplyNeed] = Field(
        description=(
            "What kinds of fact the reply must rest on. 'document_passage' "
            "for anything that has to be quoted from a project document -- an "
            "explanation, a decision, a commitment. 'erp_field' for a live "
            "value the ERP holds -- a status, a burn-down, a budget, the open "
            "risks. Both, if the question asks for both. Neither (an empty "
            "list) for a request to record something, a question outside the "
            "project, or one too vague to act on yet."
        ),
    )
    document_query: str | None = Field(
        description=(
            "The words a project document would have to contain to answer "
            "this, in the user's own terms -- resolve a reference like 'that "
            "milestone' from the history role first. Required exactly when "
            "'document_passage' is in needs; null otherwise."
        ),
    )


DECLARE_REPLY_CONTRACT_TOOL = ToolSpec(
    name="declare_reply_contract",
    description=(
        "Before anything is looked up, say what a complete reply to this "
        "question must rest on. Call it exactly once, first, for every turn."
    ),
    arguments=DeclareReplyContractArguments,
    mutating=False,
)
"""The one function a declaration call offers, sent with ``tool_choice:
'required'`` (ADR 0021) so the result is always a call, never prose.

Deliberately absent from ``DEFAULT_TOOLS``, ``CONTROL_TOOLS`` and
``PLANNING_TOOLS`` -- offered alone, the same way
:data:`~agentic_erp_assistant.memory.extractor.PROPOSE_MEMORIES_TOOL` is. The
planner offers the model what it may *do* during a turn; this is asked
*before* any of that, in its own call, and a model shown it alongside the
other nine would sometimes declare instead of acting.
"""


DEFAULT_TOOLS: tuple[ToolSpec, ...] = (
    GET_PROJECT_STATUS_TOOL,
    GET_SPRINT_PROGRESS_TOOL,
    GET_BUDGET_SUMMARY_TOOL,
    LIST_RISKS_TOOL,
    CREATE_RISK_TOOL,
)
"""What the model is offered when a caller does not say otherwise.

A tuple, and the offering is data: adding a tool is an edit here, not a new
branch in the adapter. Note what is absent --
:data:`GET_PROJECT_STATUS_FLAKY_TOOL` is executable but not offered; see its
docstring.
"""


CONTROL_TOOLS: tuple[ToolSpec, ...] = (
    SEARCH_PROJECT_DOCUMENTS_TOOL,
    ASK_CLARIFICATION_TOOL,
    REFUSE_TOOL,
)
"""The three the runtime carries out itself rather than handing to the gateway.

Not a different kind of thing to the model -- it sees six or nine functions and
picks one. The split exists on this side of the boundary, because these three
are executed by the graph (retrieval by the retriever port, the other two by
ending the turn) and the registry has no entry for any of them.
"""


PLANNING_TOOLS: tuple[ToolSpec, ...] = DEFAULT_TOOLS + CONTROL_TOOLS
"""Everything the planner offers when it asks the model what to do next.

Deliberately the whole decision in one list. The alternative -- ask for a route
first, then ask again for arguments -- costs a round trip and invents a second
place where "what should happen next" is decided, which is the layer this
project is graded on keeping singular.

There is no ``final_answer`` here, and its absence is the contract:
:class:`ToolCallResult` is already "a call, or direct content, never both", and
the adapter sends ``tool_choice: "auto"``. Content with no call *is* the answer
route. Adding a function to say the same thing would give the model two ways to
answer and the runtime a tie to break.
"""


class ToolCallResult(BaseModel):
    """The outcome of one tool-enabled turn: a call, or an answer, never both.

    The model either decided to use a tool or decided to reply directly, and a
    result object that can hold both leaves every caller to invent its own rule
    for which one wins. The validator below makes the ambiguous states
    unconstructable, so the rule is decided once, here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str | None = None
    """The tool the model chose, or ``None`` when it answered directly."""

    arguments: dict[str, Any] | None = None
    """The call's arguments, already parsed out of the provider's JSON string.

    Parsed but *not* validated -- see :meth:`ToolSpec.validate_arguments` for
    where that belongs and why it is not here.
    """

    tool_call_id: str | None = None
    """The provider's id for this call.

    Carried because a real tool round trip has to quote it back in the
    following ``tool`` message. Dropping it here would mean the execution layer
    could run the tool and then have no way to report the result.
    """

    content: str | None = None
    """The direct reply, when no tool applied."""

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> "ToolCallResult":
        called = self.tool_name is not None
        if called and (self.arguments is None):
            raise ValueError("arguments: a tool call must carry parsed arguments")
        if not called and self.arguments is not None:
            raise ValueError("arguments: present without a tool_name")
        if not called and self.tool_call_id is not None:
            raise ValueError("tool_call_id: present without a tool_name")
        if called == (self.content is not None):
            raise ValueError(
                "a result is either a tool call or direct content, never both "
                "and never neither"
            )
        return self

    @classmethod
    def from_tool_call(
        cls,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        tool_call_id: str | None = None,
    ) -> "ToolCallResult":
        """Build the tool-call form."""
        return cls(
            tool_name=tool_name,
            arguments=dict(arguments),
            tool_call_id=tool_call_id,
        )

    @classmethod
    def from_content(cls, content: str) -> "ToolCallResult":
        """Build the direct-answer form."""
        return cls(content=content)
