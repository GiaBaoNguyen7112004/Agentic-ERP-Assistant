"""The response contracts an answer must satisfy before a caller ever sees it.

``ports.py`` settles *how* a provider call is made and how it fails. This module
settles *what shape an answer has to have to be acceptable*, and it does so with
validators rather than with documentation: a model that claims grounding it
cannot back up produces a ``ValidationError`` instead of an object.

These classes serve two purposes at once, deliberately:

* They are the guardrail. The cross-field rule on :class:`GroundedAnswer` is the
  cheapest possible enforcement of the project rule that every factual answer
  carries citations -- it runs at construction, above every caller.
* They are the prompt contract. ``model_json_schema()`` on these classes is what
  gets handed to the provider as a structured-output response format, so the
  shape we ask for and the shape we accept are one artifact and cannot drift.

Every model is frozen and forbids unknown fields. Frozen because state that has
been validated and then crosses a node boundary must not be edited afterwards --
the check would no longer mean anything. ``extra="forbid"`` because it renders as
``additionalProperties: false`` in the emitted JSON Schema, which structured
output requires, and because an unmodelled field is behavior smuggled past the
type system.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_erp_assistant.state.evidence import EvidenceSnippet

__all__ = [
    "ApprovalRequest",
    "Citation",
    "EvidenceSnippet",
    "GroundedAnswer",
]


class _Contract(BaseModel):
    """Shared configuration for every response contract in this module."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# :class:`EvidenceSnippet` is re-exported rather than defined here. It used to
# live in this module, beside :class:`Citation`, on the argument that the two
# belong together. They do not: a citation comes *out* of an answer and is a
# response contract, while a snippet goes *in* and is carried in the graph's
# state from the moment retrieval produces it. Defining it here forced
# :mod:`agentic_erp_assistant.state` to import the whole ``llm`` package -- a
# tokenizer and a gateway -- to name the type it holds. It now lives in
# :mod:`agentic_erp_assistant.state.evidence`; this name stays so every
# existing import keeps working, and there is still exactly one definition.


class Citation(_Contract):
    """One resolvable pointer back into a source document.

    ``source_id`` and ``locator`` together are what makes a reference checkable
    by a reader -- which is the entire content of the citation rule. ``locator``
    is an opaque string on purpose: retrieval may address a section, a page, or a
    chunk id, and this module is not the place to decide which.
    """

    source_id: str = Field(min_length=1)
    """The document the claim came from, e.g. ``"sprint-12-report.md"``."""

    locator: str = Field(min_length=1)
    """Where inside it, e.g. ``"3.2"``, ``"chunk-14"``, ``"p.4"``."""

    quote: str | None = None
    """The supporting span, when the retriever can supply one."""


class GroundedAnswer(_Contract):
    """An answer, plus the evidence that it is allowed to be believed.

    The invariant this class exists to enforce has two halves, and they are
    checked together in one validator so a reader sees the whole rule at once:
    an answer either shows its sources or says why it is refusing. There is no
    third state in which it asserts something unsupported.
    """

    answer: str
    """The text shown to the user. Empty is legitimate for a bare refusal."""

    citations: list[Citation] = Field(default_factory=list)
    """The sources backing ``answer``. Required whenever ``grounded`` is true."""

    grounded: bool
    """Whether the answer is claimed to rest on retrieved sources."""

    confidence: float = Field(ge=0.0, le=1.0)
    """0.0-1.0. Reported to the trace; never a substitute for a citation."""

    refusal_reason: str | None = None
    """Why no grounded answer was produced. Required when ``grounded`` is false."""

    @model_validator(mode="after")
    def _grounding_must_be_backed_or_explained(self) -> "GroundedAnswer":
        if self.grounded and not self.citations:
            raise ValueError(
                "citations: a grounded answer must cite at least one source"
            )
        if not self.grounded and not (self.refusal_reason or "").strip():
            raise ValueError(
                "refusal_reason: an ungrounded answer must say why it refused"
            )
        return self


class ApprovalRequest(_Contract):
    """A mutating tool call, paused and put to a human in words.

    ``arguments_summary`` is a summary written for the approver, not the raw
    argument payload. That is a decision, not a shortcut: the approver has to be
    able to read and judge it in one line, and an approval prompt must never
    become a place where a full argument dict -- credentials included -- is
    rendered to a screen and into the trace.
    """

    tool_name: str = Field(min_length=1)
    """The tool awaiting a decision, as the router knows it."""

    reason: str = Field(min_length=1)
    """Why this call is being asked for, in terms the approver can weigh."""

    arguments_summary: str = Field(min_length=1)
    """What it will do, summarized -- e.g. ``"move SPR-14 to done"``."""
