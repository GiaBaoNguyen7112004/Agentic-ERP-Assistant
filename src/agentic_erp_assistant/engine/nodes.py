"""The three things a turn can be doing, as functions from state to state.

One node per execution shape, and the shapes are the routes: thinking, searching
documents, running a tool. A node takes an
:class:`~agentic_erp_assistant.state.agent_state.AgentState` and returns the next
one, having moved through
:func:`~agentic_erp_assistant.engine.transitions.advance` -- so a node cannot
invent a move the table does not declare, and cannot end a turn without leaving
a response or a failure behind.

What a node is not allowed to do
--------------------------------

Loop. A node performs one action and returns; the runtime decides whether
another node runs. That is what keeps the step budget meaningful -- a node that
retried internally would spend time and money outside the counter that is
supposed to bound it. The one place this shows is the rate-limit path in
:meth:`GraphNodes.execute_tool`: it returns a state on the *same* route, with
``retry_count`` raised, and the engine calls it again. A retry is a repeated node
execution, never a graph edge, which is why ``call_tool -> call_tool`` is not in
the transition table and should not be.

Where the reason-act cycle actually closes
------------------------------------------

A successful tool call routes back to ``think`` rather than straight to an
answer. That costs a second model call and buys the thing the loop exists for:
the model sees what its own action returned and decides whether that is enough,
whether another call is needed, or whether the result changes the question. An
edge from ``call_tool`` to ``answer`` would make the graph a single shot with a
planner attached, and the step budget would be guarding a cycle that could not
happen.
"""

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from agentic_erp_assistant.reasoning.completeness import Completeness, assess, next_redirect
from agentic_erp_assistant.reasoning.decision import DecisionRoute, ReasoningDecision
from agentic_erp_assistant.engine.ports import (
    AnswerComposerPort,
    DocumentRetrieverPort,
    PlannerPort,
    ToolGatewayPort,
)
from agentic_erp_assistant.engine.transitions import advance
from agentic_erp_assistant.state.agent_state import (
    AgentState,
    ERROR_DETAIL_MAX_CHARS,
)
from agentic_erp_assistant.state.events import (
    EVENT_DETAIL_MAX_CHARS,
    EventKind,
    TraceEvent,
)
from agentic_erp_assistant.state.evidence import EvidenceSnippet
from agentic_erp_assistant.state.tool_outcome import ToolOutcome
from agentic_erp_assistant.state.tool_request import ToolRequest, summarize_tool_call

__all__ = [
    "EVIDENCE_LIMIT",
    "GraphNodes",
    "NO_EVIDENCE_REPLY",
    "Node",
    "NodeTable",
    "RETRIEVAL_TOOL",
    "SOURCES_PREFIX",
]

logger = logging.getLogger(__name__)


Node = Callable[[AgentState], AgentState]
"""One step of the graph: a state in, the next state out."""

NodeTable = Mapping[DecisionRoute | None, Node]
"""Which node runs for a state sitting on which route.

``None`` is a real key: an unrouted state is where every turn starts, and a
table that began at the second step would leave the first one undispatched.
"""


EVIDENCE_LIMIT = 4
"""How many passages one retrieval pulls.

A runtime decision rather than a retriever default, for the reason
:class:`~agentic_erp_assistant.engine.ports.DocumentRetrieverPort` gives: it
sets the size of every prompt, and a number living inside a backend would change
that from a place no reviewer looks.
"""


RETRIEVAL_TOOL = "search_project_documents"
"""The offered function that means "search", carried on the state as the tool
this turn chose.

Retrieval is chosen exactly like a tool and is not run like one, so the choice
is recorded in ``tool_name``/``tool_arguments`` while the execution happens
against the retriever port. A reviewer reading a trace sees the same shape for
every decision the model made, which is the point.
"""


NO_EVIDENCE_REPLY = (
    "I could not find anything in the project documents that answers that, so "
    "I am not going to guess."
)
"""What an empty retrieval says. A refusal, not an apology with a hedge in it."""


SOURCES_PREFIX = "Sources: "
"""How a reply lists what backs it. One prefix, so a UI can find the line."""


def _clip(text: str, limit: int) -> str:
    """Cut free text to a field's cap, marking that it was cut.

    Traces are capped for a reason (see
    :data:`~agentic_erp_assistant.state.events.EVENT_DETAIL_MAX_CHARS`), and a
    node that let a provider's error message overflow one would turn a bad turn
    into a failed construction.
    """
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _event(node: str, kind: EventKind, detail: str = "") -> TraceEvent:
    return TraceEvent(node=node, kind=kind, detail=_clip(detail, EVENT_DETAIL_MAX_CHARS))


def _with_sources(body: str, sources: Sequence[str]) -> str:
    """Append the identifiers a reply rests on, if there are any.

    Deterministic, and built from what was actually retrieved or returned rather
    than from anything a model wrote. A citation the model composed is a claim;
    this is a record.
    """
    if not sources:
        return body
    return f"{body}\n\n{SOURCES_PREFIX}{', '.join(sources)}"


def _dedupe_by_tag(snippets: Iterable[EvidenceSnippet]) -> tuple[EvidenceSnippet, ...]:
    """The union of several searches' hits, each passage kept once (ADR 0027).

    Order preserved, first occurrence wins -- so a passage two queries both
    match is shown to the composer once, at the rank the earlier query gave
    it, rather than once per query it happened to satisfy.
    :attr:`~agentic_erp_assistant.state.evidence.EvidenceSnippet.tag` is the
    identity: the same string the composer is told to cite, so two snippets
    that would render the same citation are the same passage for this
    purpose regardless of which search produced the copy.
    """
    seen: set[str] = set()
    deduped: list[EvidenceSnippet] = []
    for snippet in snippets:
        if snippet.tag in seen:
            continue
        seen.add(snippet.tag)
        deduped.append(snippet)
    return tuple(deduped)


@dataclass(frozen=True)
class GraphNodes:
    """The nodes, bound to the four ports they run against.

    Frozen, and every collaborator is a protocol. A test builds this with four
    fakes and gets the real routing logic; nothing here knows what a retriever
    indexes, what a gateway executes, or who serves the model.
    """

    retriever: DocumentRetrieverPort
    tools: ToolGatewayPort
    planner: PlannerPort
    composer: AnswerComposerPort

    evidence_limit: int = EVIDENCE_LIMIT
    """How many passages to pull per search."""

    tool_retry_budget: int = 1
    """How many times one turn may wait out a rate limit and try again.

    A budget, not a policy of persistence: the limiter says how long to wait and
    this says how many times that is worth doing. Zero is a legitimate setting
    and means "report the throttle, do not wait".
    """

    sleep: Callable[[float], object] = lambda seconds: None
    """How to wait between rate-limited attempts.

    Defaults to not waiting at all, which is the right default for a component
    whose tests must not sleep. A deployment passes ``time.sleep``; the runtime
    passes whatever it was given.
    """

    def table(self) -> NodeTable:
        """Which node handles which route.

        A table, not a chain of ``if``. A route with no entry is a routing bug
        the engine reports by name, rather than a state that silently does
        nothing and looks like a hang.
        """
        return {
            None: self.think,
            "think": self.think,
            "retrieve_project_documents": self.retrieve_and_answer,
            "call_tool": self.execute_tool,
        }

    # -- reason ------------------------------------------------------------

    def think(self, state: AgentState) -> AgentState:
        """Ask the planner what to do next, and move to whatever it said.

        The rationale goes into the trace as the ``route_selected`` detail --
        capped, and never read by a branch. What the graph acts on is the typed
        route beside it.

        A planner that raises is a provider that failed, which is a turn that
        ends rather than an exception the caller has to catch: the run is
        already half-recorded, and the trace has to say why it stopped.

        ADR 0019: after the normal call, :func:`_repeats_a_success` asks
        whether the decision just made names a call already sitting in
        ``observations`` as ``ok``. If so, the planner is re-asked once with
        tools withheld -- which can only come back ``answer`` or ``fail``
        (the real provider's ``tool_choice: "none"`` cannot produce a tool
        call) -- and ``forced_note`` is what the trace says about *why*:
        carried through to the ``route_selected`` event on success, and to a
        ``planner_loop`` failure, never ``provider_failure``, if the forced
        call still named one.
        """
        forced_note: str | None = None
        try:
            decision = self.planner.plan(state)
            if _repeats_a_success(state, decision):
                forced_note = (
                    f"{decision.required_tool} repeated with the same "
                    f"arguments; tools withheld"
                )
                decision = self.planner.plan(state, tool_choice="none")
        except Exception as error:  # noqa: BLE001 - deliberately broad; see above
            logger.warning("planner failed on %s: %s", state.trace_id, error)
            return advance(
                state,
                "fail",
                mutating=state.tool_mutating if state.route == "call_tool" else None,
                failure="provider_failure",
                error_detail=_clip(
                    f"{type(error).__name__}: {error}", ERROR_DETAIL_MAX_CHARS
                ),
                events=state.events
                + (
                    _event("think", "failed", f"planner raised {type(error).__name__}"),
                ),
            )

        events = state.events + (
            _event(
                "think",
                "route_selected",
                f"{decision.route}: {forced_note}"
                if forced_note is not None
                else f"{decision.route}: {decision.rationale}",
            ),
        )

        if forced_note is not None and decision.route not in ("answer", "fail"):
            # Belt and suspenders beside Planner._unreadable's own guarantee
            # (tool_choice="none" -> only "answer" or "fail" comes back): this
            # is the boundary the graph itself enforces, so a PlannerPort
            # implementation that does not honor the contract as strictly
            # still cannot act on a call that should have been impossible --
            # the real provider's tool_choice: "none" cannot produce one.
            detail = f"{forced_note}; the forced call named {decision.route!r} anyway"
            return advance(
                state,
                "fail",
                failure="planner_loop",
                error_detail=_clip(detail, ERROR_DETAIL_MAX_CHARS),
                events=events + (_event("think", "failed", detail),),
            )

        # ADR 0021 / ADR 0025: does this decision satisfy what the planner
        # itself declared a complete reply needs? "answer",
        # "retrieve_project_documents" and "refuse" are worth asking --
        # clarify and the routes that feed a later think() (call_tool,
        # request_approval) are not. A refusal is a claim, exactly like an
        # answer is: "nothing available could support this" is untested
        # until the need the contract itself declared has actually been
        # looked for. `gap` stays available for the "answer" branch below,
        # which is the one place a still-missing need, after its one
        # redirect, has to be delivered rather than silently dropped.
        gap: Completeness | None = None
        if decision.route in ("answer", "retrieve_project_documents", "refuse"):
            gap = assess(state.contract, state)
            need = next_redirect(gap, state.redirected_needs)

            if need == "erp_field":
                # D5: fields before passages. Whether the model wanted to
                # answer or already chose to search, an unmet ERP field is
                # fetched first -- retrieval is withheld, not the tools that
                # could supply it, so the model must pick one of those, ask
                # back, or refuse.
                events = events + (
                    _event(
                        "think",
                        "contract_enforced",
                        "erp_field: a call was required, search withheld",
                    ),
                )
                state = state.evolve(
                    redirected_needs=state.redirected_needs | {"erp_field"}
                )
                try:
                    decision = self.planner.plan(
                        state, tool_choice="required", withhold={RETRIEVAL_TOOL}
                    )
                except Exception as error:  # noqa: BLE001 - see the first call's handler above
                    logger.warning(
                        "planner failed on %s: %s", state.trace_id, error
                    )
                    return advance(
                        state,
                        "fail",
                        failure="provider_failure",
                        error_detail=_clip(
                            f"{type(error).__name__}: {error}", ERROR_DETAIL_MAX_CHARS
                        ),
                        events=events
                        + (
                            _event(
                                "think",
                                "failed",
                                f"planner raised {type(error).__name__}",
                            ),
                        ),
                    )
                events = events + (
                    _event(
                        "think", "route_selected", f"{decision.route}: {decision.rationale}"
                    ),
                )

            elif need == "document_passage" and decision.route in (
                "answer",
                "refuse",
            ):
                # Only reachable with route == "answer" or "refuse": a
                # decision that already chose retrieve_project_documents is
                # already doing this, with its own query, and is left alone
                # -- forcing the contract's query over the model's own would
                # discard a choice that was already correct.
                assert state.contract is not None  # need only exists if it is
                withheld_route = decision.route
                events = events + (
                    _event(
                        "think",
                        "route_selected",
                        f"retrieve_project_documents: contract needs a document "
                        f"passage; {withheld_route} withheld",
                    ),
                    _event(
                        "think",
                        "contract_enforced",
                        f"document_passage: {withheld_route} withheld, searching "
                        f"{state.contract.document_query!r}"
                        + (
                            f" (refusal reason: {decision.message!r})"
                            if withheld_route == "refuse"
                            else ""
                        ),
                    ),
                )
                # ADR 0025: a refusal's own message is a claim about why
                # nothing could support an answer, not a draft answer -- so
                # unlike the "answer" branch, no draft is carried forward.
                # If the redirected search still finds nothing,
                # retrieve_and_answer's no-passages path (state.draft is
                # None) falls through to an ordinary, now-tested refusal
                # instead of delivering the model's untested claim as if it
                # were a withheld answer.
                return advance(
                    state,
                    "retrieve_project_documents",
                    tool_name=RETRIEVAL_TOOL,
                    tool_arguments={"queries": [state.contract.document_query]},
                    tool_mutating=False,
                    draft=decision.message if withheld_route == "answer" else None,
                    redirected_needs=state.redirected_needs | {"document_passage"},
                    events=events,
                )

        if decision.route == "retrieve_project_documents":
            return advance(
                state,
                decision.route,
                tool_name=RETRIEVAL_TOOL,
                tool_arguments={"queries": list(decision.search_queries)},
                tool_mutating=False,
                events=events,
            )

        if decision.route == "request_approval":
            # ADR 0016: a write is put to a human only after the checks that
            # would refuse it anyway have already passed. The transition
            # guard still forbids an unapproved write from reaching
            # call_tool -- this only decides whether the human is asked at
            # all.
            outcome = self.tools.preflight(
                ToolRequest(
                    trace_id=state.trace_id,
                    tool_name=decision.required_tool,
                    arguments=decision.tool_arguments or {},
                    actor=state.actor,
                    project_code=state.project_code,
                    scopes=state.scopes,
                    # Deliberately not state.approval: that field describes
                    # the *previous* tool call this turn made, if any -- an
                    # approved write earlier in the same turn must not make a
                    # second, unrelated write look pre-approved here. Nothing
                    # has been decided yet about *this* call, so the honest
                    # value is the request's own default.
                    approval="not_required",
                )
            )
            if outcome.status != "approval_required":
                return advance(
                    state,
                    "fail",
                    tool_name=decision.required_tool,
                    tool_arguments=decision.tool_arguments,
                    tool_mutating=decision.mutating,
                    observations=state.observations + (outcome,),
                    failure="tool_failure",
                    error_detail=_clip(
                        f"{outcome.tool_name} -> {outcome.status}: "
                        f"{outcome.error or ''}",
                        ERROR_DETAIL_MAX_CHARS,
                    ),
                    events=events
                    + (
                        _event(
                            "think",
                            "failed",
                            f"{outcome.tool_name} refused before approval: "
                            f"{outcome.status}",
                        ),
                    ),
                )
            return advance(
                state,
                decision.route,
                tool_name=decision.required_tool,
                tool_arguments=decision.tool_arguments,
                tool_mutating=decision.mutating,
                approval="pending",
                observations=state.observations + (outcome,),
                events=events
                + (
                    _event(
                        "think",
                        "approval_requested",
                        f"{decision.required_tool} needs a human",
                    ),
                ),
            )

        if decision.route == "call_tool":
            return advance(
                state,
                decision.route,
                mutating=decision.mutating,
                tool_name=decision.required_tool,
                tool_arguments=decision.tool_arguments,
                tool_mutating=decision.mutating,
                events=events,
            )

        if decision.route == "answer":
            response = _with_sources(
                decision.message or "", _observed_sources(state.observations)
            )
            if gap is not None and gap.missing:
                # Every need named here has already been redirected once
                # (the block above spends the one redirect a need gets) and
                # is still missing -- on the real provider this is
                # unreachable (a redirected call cannot come back "answer"
                # with the need still unmet; see reasoning/completeness.py's
                # module docstring), reachable only against a PlannerPort
                # that does not honor tool_choice or withhold as strictly as
                # this graph assumes. Delivered anyway, exactly as it would
                # have been without this check -- a reply the check cannot
                # complete is still worth more to the user than none -- but
                # marked, never silently.
                missing = ", ".join(sorted(gap.missing))
                return advance(
                    state,
                    "answer",
                    response=response,
                    failure="incomplete_reply",
                    error_detail=_clip(
                        f"contract not met after redirect: {missing}",
                        ERROR_DETAIL_MAX_CHARS,
                    ),
                    events=events
                    + (
                        _event(
                            "think",
                            "contract_enforced",
                            f"unmet after redirect: {missing}",
                        ),
                    ),
                )
            return advance(state, "answer", response=response, events=events)

        if decision.route in ("clarify", "refuse"):
            return advance(state, decision.route, response=decision.message, events=events)

        # Only reachable normally as "the model named a tool that does not
        # exist" (Planner._unreadable). With forced_note set, it means the
        # forced (tool_choice="none") call still named a tool -- something the
        # real provider's tool_choice: "none" cannot do -- and the failure
        # says so by name rather than reading as an ordinary provider hiccup.
        failure_detail = (
            f"{forced_note}; {decision.rationale}"
            if forced_note is not None
            else decision.rationale
        )
        return advance(
            state,
            "fail",
            failure="planner_loop" if forced_note is not None else "provider_failure",
            error_detail=_clip(failure_detail, ERROR_DETAIL_MAX_CHARS)
            or "the planner could not produce an actionable decision",
            events=events + (_event("think", "failed", failure_detail),),
        )

    # -- act: documents ----------------------------------------------------

    def retrieve_and_answer(self, state: AgentState) -> AgentState:
        """Search, then answer from what came back -- or refuse, citing nothing.

        Retrieval and composition are one node because they are one obligation:
        an answer to a document question exists only if passages back it. Split
        across two nodes, there would be a state in between where the graph held
        evidence and had not yet decided whether it was enough, and something
        would eventually answer from it.
        """
        queries = self._queries(state)
        try:
            hits_by_query = tuple(
                tuple(self.retriever.search(query, limit=self.evidence_limit))
                for query in queries
            )
        except Exception as error:  # noqa: BLE001 - a failed turn, not a crash
            logger.warning("retriever failed on %s: %s", state.trace_id, error)
            return advance(
                state,
                "fail",
                failure="provider_failure",
                error_detail=_clip(
                    f"{type(error).__name__}: {error}", ERROR_DETAIL_MAX_CHARS
                ),
                events=state.events
                + (
                    _event(
                        "retrieve", "failed", f"retriever raised {type(error).__name__}"
                    ),
                ),
            )
        # ADR 0027: one query per document, unioned and deduped by citation
        # tag rather than composed one query at a time -- retrieval and
        # composition stay one obligation (see this method's own docstring),
        # and a document two queries both happen to match is shown to the
        # composer once, not twice.
        snippets = _dedupe_by_tag(snippet for hits in hits_by_query for snippet in hits)
        events = state.events + (
            _event(
                "retrieve",
                "evidence_retrieved",
                f"{'+'.join(str(len(hits)) for hits in hits_by_query)} passage(s) "
                f"for {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}: "
                f"{', '.join(repr(query) for query in queries)}",
            ),
        )

        # ADR 0021: when the check's own redirect sent this search out, the
        # planner's withheld reply is delivered anyway -- never worse than
        # the turn would have been without the check -- marked incomplete
        # rather than refused or failed. A search the *model* chose ends
        # exactly as it always has, below: only a redirect this check made
        # gets the softer landing. Three ways a redirect can come back
        # without a grounded reply, one landing for all of them: nothing
        # retrieved; something retrieved that the composer could not
        # ground a reply on (it refused, or cited something never
        # retrieved); or a composer reply that broke its own schema --
        # "grounded, and citing nothing" is what a model that answered
        # from memory or history rather than the passages sends back.
        redirected = "document_passage" in state.redirected_needs and state.draft is not None

        def deliver_draft(why: str, evidence: Sequence[EvidenceSnippet]) -> AgentState:
            return advance(
                state,
                "answer",
                evidence=tuple(evidence),
                response=_with_sources(state.draft, _observed_sources(state.observations)),
                failure="incomplete_reply",
                error_detail=_clip(
                    f"contract needs a document passage; {why}", ERROR_DETAIL_MAX_CHARS
                ),
                events=events
                + (
                    _event(
                        "retrieve",
                        "contract_enforced",
                        "unmet after redirect: document_passage",
                    ),
                ),
            )

        if not snippets:
            if redirected:
                return deliver_draft(f"the redirected search for {queries!r} found none", ())
            return advance(
                state,
                "refuse",
                evidence=(),
                response=NO_EVIDENCE_REPLY,
                failure="insufficient_evidence",
                events=events + (_event("retrieve", "failed", "no passages"),),
            )

        try:
            answer = self.composer.answer(
                state.request, snippets, state.memories, state.history, state.observations
            )
        except Exception as error:  # noqa: BLE001 - a failed turn, not a crash
            logger.warning("composer failed on %s: %s", state.trace_id, error)
            if redirected and isinstance(error, ValidationError):
                # The provider answered and the reply broke the schema: the
                # passages did not ground a reply, the same landing as
                # finding none. A provider that never answered (network,
                # auth, budget) is a real failure and stays one below.
                return deliver_draft(
                    f"the redirected search for {queries!r} found "
                    f"{len(snippets)} passage(s) that did not ground a reply "
                    f"(composer raised ValidationError)",
                    snippets,
                )
            return advance(
                state,
                "fail",
                evidence=snippets,
                failure="provider_failure",
                error_detail=_clip(
                    f"{type(error).__name__}: {error}", ERROR_DETAIL_MAX_CHARS
                ),
                events=events
                + (
                    _event(
                        "retrieve", "failed", f"composer raised {type(error).__name__}"
                    ),
                ),
            )

        problem = _ungrounded(answer, snippets)
        if problem is not None:
            if redirected:
                return deliver_draft(
                    f"the redirected search for {queries!r} found "
                    f"{len(snippets)} passage(s) that did not ground a reply ({problem})",
                    snippets,
                )
            return advance(
                state,
                "refuse",
                evidence=snippets,
                response=answer.refusal_reason or NO_EVIDENCE_REPLY,
                failure="insufficient_evidence",
                error_detail=_clip(problem, ERROR_DETAIL_MAX_CHARS),
                events=events + (_event("retrieve", "failed", problem),),
            )

        sources = [f"[{cite.source_id}#{cite.locator}]" for cite in answer.citations]
        sources += [
            source_id
            for source_id in _observed_sources(state.observations)
            if source_id not in sources
        ]
        response = _with_sources(answer.answer, sources)

        # Belt and suspenders (ADR 0021): the redirect logic in think() means
        # an erp_field need cannot reach this point still missing against a
        # well-behaved PlannerPort -- see reasoning/completeness.py's module
        # docstring -- but a non-conforming one is not this method's problem
        # to trust away.
        gap = assess(state.contract, state.evolve(evidence=snippets))
        if gap.missing:
            missing = ", ".join(sorted(gap.missing))
            return advance(
                state,
                "answer",
                evidence=snippets,
                response=response,
                failure="incomplete_reply",
                error_detail=_clip(
                    f"contract not met after redirect: {missing}", ERROR_DETAIL_MAX_CHARS
                ),
                events=events
                + (
                    _event(
                        "retrieve", "contract_enforced", f"unmet after redirect: {missing}"
                    ),
                ),
            )

        return advance(
            state,
            "answer",
            evidence=snippets,
            response=response,
            events=events,
        )

    def _queries(self, state: AgentState) -> tuple[str, ...]:
        """What to search for: the planner's queries, or the request behind
        them (ADR 0027).

        The planner's own ``queries`` are preferred because each is the part
        of the question that has to be looked up for one document. The
        legacy singular ``query`` key is read too, wrapped as one-entry
        tuple -- a state built or replayed before ADR 0027 (a redirect from
        an older trace, a hand-built test state) still runs. Falling back to
        the raw request rather than failing keeps a hand-built state -- a
        replay, a test, a resumed run -- runnable without a planner having
        been involved at all.
        """
        arguments = state.tool_arguments or {}

        queries = arguments.get("queries")
        if isinstance(queries, list):
            cleaned = tuple(
                q for q in queries if isinstance(q, str) and q.strip()
            )
            if cleaned:
                return cleaned

        query = arguments.get("query")
        if isinstance(query, str) and query.strip():
            return (query,)

        return (state.request,)

    # -- act: tools --------------------------------------------------------

    def execute_tool(self, state: AgentState) -> AgentState:
        """Run the chosen call through the gateway and route on what came back.

        The gateway is the authority on whether the call may happen: this node
        hands it the actor, their scopes and the recorded approval and reads the
        status it returns. It never decides that a call is permitted, which is
        why an ``approval_required`` coming back is routed to a human rather
        than treated as a contradiction -- policy can escalate a read, and only
        the registry knows that it did.
        """
        outcome = self.tools.execute(
            ToolRequest(
                trace_id=state.trace_id,
                tool_name=state.tool_name or "",
                arguments=state.tool_arguments or {},
                actor=state.actor,
                project_code=state.project_code,
                scopes=state.scopes,
                approval=state.approval,
            )
        )
        observations = state.observations + (outcome,)
        events = state.events + (
            _event(
                "execute_tool",
                "tool_called" if outcome.status == "ok" else "failed",
                f"{outcome.tool_name} -> {outcome.status}",
            ),
        )

        if outcome.status == "ok":
            # Back to the planner, not straight to an answer: the model has to
            # see what the call returned before deciding the turn is done.
            return advance(
                state,
                "think",
                mutating=state.tool_mutating,
                observations=observations,
                events=events,
            )

        if outcome.status == "rate_limited":
            return self._throttled(state, outcome, observations, events)

        if outcome.status == "approval_required":
            return advance(
                state,
                "request_approval",
                mutating=state.tool_mutating,
                approval="pending",
                observations=observations,
                events=events
                + (
                    _event(
                        "execute_tool",
                        "approval_requested",
                        f"{outcome.tool_name} was escalated by policy",
                    ),
                ),
            )

        return advance(
            state,
            "fail",
            mutating=state.tool_mutating,
            observations=observations,
            failure="tool_failure",
            error_detail=_clip(
                f"{outcome.tool_name} -> {outcome.status}: {outcome.error or ''}",
                ERROR_DETAIL_MAX_CHARS,
            ),
            events=events,
        )

    def _throttled(
        self,
        state: AgentState,
        outcome: ToolOutcome,
        observations: tuple[ToolOutcome, ...],
        events: tuple[TraceEvent, ...],
    ) -> AgentState:
        """Wait out a spent budget once, or report it and stop.

        No route change on the retry path. The state comes back sitting on
        ``call_tool`` with one more retry spent, and the engine runs this node
        again -- so the attempt is inside the step budget, appears in the trace
        as its own step, and needs no edge that would also permit a loop nobody
        intended.
        """
        wait = outcome.retry_after_seconds or 0.0
        events = events + (
            _event(
                "execute_tool",
                "rate_limited",
                f"{outcome.tool_name} throttled, {wait:g}s to wait",
            ),
        )

        if state.retry_count >= self.tool_retry_budget:
            return advance(
                state,
                "fail",
                mutating=state.tool_mutating,
                observations=observations,
                failure="tool_failure",
                error_detail=_clip(
                    f"{outcome.tool_name} stayed rate limited after "
                    f"{state.retry_count} retr"
                    f"{'y' if state.retry_count == 1 else 'ies'}",
                    ERROR_DETAIL_MAX_CHARS,
                ),
                events=events,
            )

        self.sleep(wait)
        return state.evolve(
            observations=observations,
            retry_count=state.retry_count + 1,
            events=events
            + (
                _event(
                    "execute_tool",
                    "retry_scheduled",
                    f"attempt {state.retry_count + 2} of "
                    f"{self.tool_retry_budget + 1}",
                ),
            ),
        )


def _observed_sources(observations: Sequence[ToolOutcome]) -> list[str]:
    """Every identifier this turn's calls actually touched, in order, deduped."""
    seen: list[str] = []
    for outcome in observations:
        for source_id in outcome.source_ids:
            if source_id not in seen:
                seen.append(source_id)
    return seen


def _repeats_a_success(state: AgentState, decision: ReasoningDecision) -> bool:
    """Does ``decision`` name a call already sitting in ``observations`` as
    ``ok``, arguments and all?

    ADR 0019's guard: a decision that repeats a call already known to have
    succeeded is refused a re-run and the planner is re-asked once with tools
    withheld instead, ending the turn in an answer rather than a repeat that
    can only tell it what it already knows. This is A11's actual shape --
    ``list_risks`` had already succeeded once (the contract's own "check
    first" step, before the write it was checking for) and came back after
    the write succeeded, not because the model needed a second, *different*
    action.

    Deliberately **not** "any tool call right after a successful write":
    that reading was tried first and rejected, live, against
    ``test_a_resumed_turn_can_pause_again_and_waits_anew`` -- a turn that
    writes two different things in sequence routes a second, genuinely new
    ``request_approval`` right after the first write succeeds, and blocking
    every tool call post-write would have blocked that legitimate one along
    with A11's unproductive one. Only a call *identical* to one already
    known to have succeeded is refused.

    Compared by
    :func:`~agentic_erp_assistant.state.tool_request.summarize_tool_call`'s
    rendering rather than the raw arguments, so this is the same notion of
    "the same call" an approver or an auditor reading the trace would use.
    """
    if decision.required_tool is None:
        return False
    summary = summarize_tool_call(decision.required_tool, decision.tool_arguments or {})
    return any(
        observation.tool_name == decision.required_tool
        and observation.status == "ok"
        and observation.arguments_summary == summary
        for observation in state.observations
    )


def _ungrounded(answer, snippets: Sequence[EvidenceSnippet]) -> str | None:
    """Say why this answer must not be delivered as fact, or ``None``.

    Three checks, and the third is the one that matters. A composer is trusted
    to write and is not trusted to have cited something real, so every citation
    is matched against the passages actually retrieved. A model that cites a
    plausible source id it never saw produces an answer that looks grounded and
    is not, and no downstream reader can tell.
    """
    if not answer.grounded:
        return f"the composer refused: {answer.refusal_reason or 'no reason given'}"
    if not answer.citations:
        return "the answer claimed grounding and cited nothing"

    retrieved = {snippet.source_id for snippet in snippets}
    invented = sorted(
        {cite.source_id for cite in answer.citations} - retrieved
    )
    if invented:
        return f"cited sources that were never retrieved: {', '.join(invented)}"
    return None
