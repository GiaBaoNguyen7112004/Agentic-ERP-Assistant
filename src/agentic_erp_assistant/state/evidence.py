"""A retrieved passage, on its way into a prompt -- and into a citation.

This type used to live in :mod:`agentic_erp_assistant.llm.schemas`, next to the
contracts an answer has to satisfy. It moved here because the state layer has
to carry it: evidence travels through the graph from the retriever to the
prompt, so a turn's state holds these, and ``state`` is the innermost layer --
importing the ``llm`` package to reach the type would pull a tokenizer and a
gateway into every test that builds a state. ``llm.schemas`` re-exports it, so
every existing import keeps working and there is still exactly one definition.

The direction matters
---------------------

:class:`EvidenceSnippet` goes *into* a prompt.
:class:`~agentic_erp_assistant.llm.schemas.Citation` comes *back out* of an
answer, and stays in ``llm.schemas`` where the response contracts are. They
share only an identifier, deliberately: two types travelling in opposite
directions is what turns "did the model cite something we actually gave it?"
into a question code can answer rather than a hope.

Why a passage is not a plain string
-----------------------------------

Every factual answer here has to arrive with a citation a reviewer can resolve.
That requirement is either enforced by the type that carries retrieved text or
it is enforced nowhere: once evidence travels as ``list[str]``, the moment the
strings are joined into a prompt the link back to the document is gone, and no
later guardrail can rebuild it. So the passage keeps its address with it.

What is deliberately not here
-----------------------------

A retrieval score. The port returns passages in rank order and the runtime
truncates from the end, so nothing downstream compares scores -- and a number
that no branch reads, sitting inside the object that goes into the prompt,
would be retrieval telemetry masquerading as evidence. It belongs to the trace,
or to a ranked wrapper at the ``rag`` boundary if an evaluator ever needs it.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["EvidenceSnippet"]


_FORBIDDEN_IN_TAG = frozenset("[]#\n\r")


class EvidenceSnippet(BaseModel):
    """One retrieved passage, on its way *into* a prompt.

    Frozen, because a passage that was retrieved and then edited before it
    reached the model would make its citation describe text nobody read.
    ``extra="forbid"`` because an unmodelled field here is retrieval metadata
    entering the reasoning state that no reviewer agreed to carry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    """The document this passage came from, e.g. ``"sprint-12-report.md"``."""

    locator: str = Field(min_length=1)
    """Where inside it -- an opaque string, so retrieval may address a section
    (``"3.2"``), a chunk (``"chunk-14"``) or a page (``"p.4"``) without this
    module deciding which. The same form
    :class:`~agentic_erp_assistant.llm.schemas.Citation` uses."""

    text: str = Field(min_length=1)
    """The passage itself. Read by the model as data, never as an instruction."""

    @property
    def tag(self) -> str:
        """The exact token the model is told to cite, e.g. ``[doc-12#3.2]``."""
        return f"[{self.source_id}#{self.locator}]"

    @field_validator("source_id", "locator")
    @classmethod
    def _must_not_forge_a_tag(cls, value: str) -> str:
        """Reject the characters that build a tag.

        A security check, not tidiness. ``source_id`` and ``locator`` can arrive
        from a retrieved document, so if they may contain ``[``, ``]``, ``#`` or
        a newline then that document controls part of the rendered tag and can
        manufacture a reference to a source that does not exist.
        """
        if _FORBIDDEN_IN_TAG & set(value):
            raise ValueError("must not contain '[', ']', '#', or a line break")
        return value
