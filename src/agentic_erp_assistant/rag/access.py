"""Who may read a chunk. One rule, in one function, used by every search path.

The failure this module exists to prevent is specific and common: a system with
two retrieval paths gets a security fix on one of them. Lexical search learns to
filter by project and dense search does not, or the vector index gains a payload
filter and the in-memory fallback keeps the old comparison, and the gap is
invisible until somebody quotes a document they should never have seen.

So the rule is written once, here, as :func:`is_authorized`, and both search
paths call it. The vector index additionally pushes the same rule to the server
as a payload filter -- and :func:`qdrant_filter`, which builds that filter, is in
this file rather than in the adapter, so the two expressions of one policy sit
next to each other where a reviewer reads them together and a test can assert
they agree on the same chunks.

Deny by default, and never partially
------------------------------------

Authorization requires **both** halves: the chunk must belong to the context's
project, and the context must hold the chunk's required scope. There is no
public tier, no wildcard scope, and no fallback for a context that arrived with
nothing -- an actor holding no scopes is refused every document rather than
granted the unrestricted ones, because a context with empty scopes is far more
likely to be a bug in how it was built than a real person with no entitlements.

Why this reuses the scopes the tool gateway already uses
--------------------------------------------------------

:attr:`~agentic_erp_assistant.state.agent_state.AgentState.scopes` already exists
and already gates tool execution, with names describing what they grant
(``project.risk.write``) rather than who holds them. Document access uses the
same vocabulary (``project.docs.read``, ``project.docs.finance.read``) rather
than introducing a parallel role table. Two permission systems in one assistant
is two places to grant something, and the interesting question -- "what can this
actor see?" -- would have needed both to answer.

Where the context comes from
----------------------------

A :class:`RetrievalContext` is built once per turn and bound to a retriever,
never passed into ``search()``. See
:class:`~agentic_erp_assistant.rag.retriever.RetrievalService` for why.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - a type name, not a dependency
    from qdrant_client import models as qdrant_models

__all__ = [
    "PROJECT_FIELD",
    "RetrievalContext",
    "SCOPE_FIELD",
    "Restricted",
    "is_authorized",
    "qdrant_filter",
]

logger = logging.getLogger(__name__)


PROJECT_FIELD = "project_code"
SCOPE_FIELD = "required_scope"
"""The payload keys the stored copy of the policy lives under.

Constants because three places have to agree on them -- the writer that stores a
chunk, the filter that queries it, and the model that reads it back. A literal
string repeated in three files is a rename away from a filter that silently
matches nothing, which fails open in the worst way: it returns no results rather
than the wrong ones, so it looks like a retrieval problem and not a policy one.
"""


class Restricted(Protocol):
    """The two fields authorization reads, and nothing else.

    Typed as a structural protocol rather than as
    :class:`~agentic_erp_assistant.rag.chunking.Chunk` so this module does not
    depend on the chunk model, and so a test can check the policy against a
    two-field stub. It also keeps the rule honest: authorization can only ever
    consult what is declared here, so no future edit can quietly start deciding
    access from a classification label or a document type.
    """

    project_code: str
    required_scope: str


@dataclass(frozen=True)
class RetrievalContext:
    """Who is searching, on which project, with what entitlements.

    Frozen, because it is snapshotted when a turn begins and bound to a
    retriever for that turn. Entitlements that could change midway would let one
    turn read two documents under two different permissions, and the trace would
    show neither -- the same argument
    :attr:`~agentic_erp_assistant.state.agent_state.AgentState.scopes` makes.
    """

    actor: str
    """Who is asking. Not consulted by the rule; carried so a trace can say who
    a refusal was for."""

    project_code: str
    """The project whose documents this turn may read."""

    scopes: frozenset[str] = frozenset()
    """What the actor is entitled to. Empty means entitled to nothing."""

    def __post_init__(self) -> None:
        if not self.actor.strip():
            raise ValueError(
                "actor: must name who is searching; a blank one would make a "
                "refusal unattributable"
            )
        if not self.project_code.strip():
            raise ValueError(
                "project_code: must name a project. A blank one would match "
                "nothing rather than everything, which is the right direction "
                "but the wrong way to find out about a bug in the caller."
            )

    @classmethod
    def for_actor(
        cls, actor: str, *, project_code: str, scopes: Iterable[str] = ()
    ) -> "RetrievalContext":
        """Build a context from whatever iterable of scopes the caller holds."""
        return cls(
            actor=actor, project_code=project_code, scopes=frozenset(scopes)
        )


def is_authorized(chunk: Restricted, context: RetrievalContext) -> bool:
    """Whether ``context`` may read ``chunk``. The whole access-control policy.

    Both conditions are required, and neither has an exception:

    * the chunk belongs to the project the context is scoped to, so a valid
      entitlement on one project never reaches another project's documents;
    * the context holds the exact scope the chunk declares, so a document with a
      narrower requirement is refused inside its own project.

    Comparison is exact. Scopes and project codes are identifiers written in a
    manifest a human reviewed, and case-folding them here would mean the policy
    depended on a normalization rule that the stored payload filter would also
    have to implement identically -- one more thing for the two paths to
    disagree about.

    Args:
        chunk: Anything carrying ``project_code`` and ``required_scope``.
        context: The turn's snapshotted entitlements.

    Returns:
        ``True`` only if both halves hold.
    """
    return (
        chunk.project_code == context.project_code
        and chunk.required_scope in context.scopes
    )


def qdrant_filter(context: RetrievalContext) -> "qdrant_models.Filter":
    """The same rule, expressed as a server-side payload filter.

    Pushing it to the index is not an optimization. A restricted chunk that
    reaches the client has already occupied a slot in the ranked result and
    displaced something the reader was entitled to see; filtering after the fact
    returns fewer results than asked for and silently degrades recall for
    exactly the users with the fewest permissions. Filtered at the index, the
    top ``k`` is the top ``k`` of what this actor may read.

    An empty scope set is expressed as a match against an empty list, which
    matches nothing. That is deliberate rather than an edge case to special-case
    away: it is the same answer :func:`is_authorized` gives, and a filter that
    omitted the clause when there was nothing to match would turn "no
    entitlements" into "no restriction".

    The import is local so that importing this policy module -- which the
    lexical path also does -- does not pull a vector-database client in behind
    it.

    Args:
        context: The turn's snapshotted entitlements.

    Returns:
        A filter matching exactly the chunks :func:`is_authorized` accepts.
    """
    from qdrant_client import models as qdrant_models

    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key=PROJECT_FIELD,
                match=qdrant_models.MatchValue(value=context.project_code),
            ),
            qdrant_models.FieldCondition(
                key=SCOPE_FIELD,
                match=qdrant_models.MatchAny(any=sorted(context.scopes)),
            ),
        ]
    )


def authorized(
    chunks: Iterable[Any], context: RetrievalContext
) -> list[Any]:
    """Filter an iterable through :func:`is_authorized`, logging what was refused.

    A convenience with one real job: it is the only place a search path is meant
    to call the rule in bulk, so the count of refused chunks is logged once, in
    one format, rather than in whatever way each path invented.
    """
    kept, refused = [], 0
    for chunk in chunks:
        if is_authorized(chunk, context):
            kept.append(chunk)
        else:
            refused += 1

    if refused:
        logger.debug(
            "refused %d chunk(s) for %s on project %s",
            refused,
            context.actor,
            context.project_code,
        )
    return kept
