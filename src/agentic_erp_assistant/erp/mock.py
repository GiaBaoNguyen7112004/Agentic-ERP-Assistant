"""A small, synthetic ERP a tool can actually read from and write to.

The tools need something behind them, and a graded project needs that something
to be inspectable. So the data is a JSON file in the repo -- ``data/erp/`` --
short enough to read in review, and every record carries the ``source_id`` a
tool result will cite. Nothing here is real, and nothing real belongs here.

Loaded into typed records, not dicts
------------------------------------

Every row is validated into a frozen model with ``extra="forbid"``. That is not
ceremony: a typo in the JSON -- ``daysLate`` for ``days_late``, a missing
``source_id`` -- fails at load with the field named, instead of surfacing later
as a tool that answered with ``None`` or cited nothing. The dataset is a fixture
and fixtures rot silently; this is what makes the rot loud.

Reads return records, misses return ``None``
--------------------------------------------

A lookup that finds nothing is not an exception here. Which milestone the model
asked about is the model's guess, and "M9 does not exist" is an ordinary answer
a tool has to report and the turn has to route on. The handler turns it into a
failed :class:`~agentic_erp_assistant.state.tool_outcome.ToolOutcome`; raising
from the store would make a bad guess look like a broken backend.

Writes reach the file the store was loaded from
-----------------------------------------------

:meth:`MockErp.create_risk` appends to this instance and then rewrites the file
the store was loaded from, atomically: a temporary file beside it, then
:func:`os.replace`. The dataset is tens of rows and readable in review, so a
whole-file rewrite costs milliseconds and keeps the file something a person can
still read -- which a ledger of appended edits would not.

A store built without a path -- :meth:`from_mapping` in a test, say -- refuses
to write rather than quietly keeping the change in memory. An append that
vanished with the process is precisely the defect this module used to have,
and a loud failure is how it stays fixed. The file's ``_``-prefixed
documentation keys are carried across a rewrite: they are review material, and
a write that erased its own readme would be worse than one that never touched
the file.

``create_risk`` holds a lock around its append-flush-rollback sequence: the
web layer means a second request can now arrive while the first is
mid-write, and without one, two threads reading ``len(self.risks)`` before
either appends would hand out the same id twice.
"""

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Budget",
    "DEFAULT_DATASET_PATH",
    "ErpAccessError",
    "ErpNotPersistedError",
    "Milestone",
    "MockErp",
    "Project",
    "ProjectErp",
    "Risk",
    "RiskSeverity",
    "Sprint",
]


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "erp" / "project.json"
)
"""Where the fixture lives, relative to this file.

At the repo root rather than inside the package, because it is review material
first and runtime data second: a reviewer looking for "what data does this
thing serve?" should find it beside the other project data, not buried under
``src/``.
"""


RiskSeverity = Literal["low", "medium", "high"]
"""How bad a risk is. A closed set, so a severity nobody can sort or report on
fails at the record rather than in a summary."""


class _Record(BaseModel):
    """Frozen, closed configuration for every row in the dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    """What a tool result cites when it reports this row.

    On every record, with no default. A row that cannot be cited is a row a
    grounded answer cannot use, and discovering that at answer time is too
    late.
    """


class Project(_Record):
    project_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class Milestone(_Record):
    milestone_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    due_on: str = Field(min_length=1)
    schedule_status: Literal["on_track", "at_risk", "late", "done"]
    days_late: int = Field(ge=0)


class Sprint(_Record):
    sprint_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    points_committed: int = Field(ge=0)
    points_completed: int = Field(ge=0)
    days_remaining: int = Field(ge=0)


class Budget(_Record):
    project_id: str = Field(min_length=1)
    approved_usd: float = Field(ge=0)
    spent_usd: float = Field(ge=0)
    forecast_at_completion_usd: float = Field(ge=0)
    as_of: str = Field(min_length=1)


class Risk(_Record):
    risk_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    severity: RiskSeverity


class ErpNotPersistedError(RuntimeError):
    """A write was asked of a store that has no file to write to.

    Raised rather than the write being quietly kept in memory: a store built by
    :meth:`MockErp.from_mapping` without a path is a store whose writes have
    nowhere to land, and the caller needs to hear that at the write, not
    discover it after the process exits. Handlers translate it into a
    :class:`~agentic_erp_assistant.tools.models.ToolError` so nothing escapes
    the gateway; defined here rather than there so ``erp/`` stays importable
    without the tool layer.
    """


class ErpAccessError(RuntimeError):
    """A write was attempted through a :class:`ProjectErp` view for a project
    it is not bound to.

    Defence in depth, not the first line: the tool gateway's project check
    (``ToolDefinition.project_argument``) refuses a call naming another
    project before a handler ever runs. This is what fires if that check were
    ever bypassed or a handler were called directly -- the view itself must
    still refuse, the same way :class:`ErpNotPersistedError` is a second,
    independent guarantee rather than trust that every caller remembered the
    first one.
    """


class MockErp:
    """One project's delivery data, for the tools to work against.

    Not frozen, because :meth:`create_risk` is the write the whole approval
    story exists for and it has to change something observable. Everything it
    changes lives on the instance and, when the store was loaded from a file,
    in that file -- so a test asserts on its own store and no test can see
    another's write.
    """

    def __init__(
        self,
        *,
        projects: tuple[Project, ...],
        milestones: tuple[Milestone, ...],
        sprints: tuple[Sprint, ...],
        budgets: tuple[Budget, ...],
        risks: tuple[Risk, ...],
        path: Path | None = None,
    ) -> None:
        self.projects = projects
        self.milestones = milestones
        self.sprints = sprints
        self.budgets = budgets
        self.risks = list(risks)
        self._path = path
        self._documentation: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], path: Path | None = None) -> Self:
        """Build a store from already-parsed JSON.

        Separate from :meth:`load` so a test can hand in three rows instead of
        depending on the repo's fixture staying the shape it expects. Keys
        beginning with an underscore are documentation for the human reading
        the file: ignored by the record models, but remembered here so a
        rewrite does not erase them.
        """
        store = cls(
            projects=tuple(Project.model_validate(row) for row in raw["projects"]),
            milestones=tuple(
                Milestone.model_validate(row) for row in raw["milestones"]
            ),
            sprints=tuple(Sprint.model_validate(row) for row in raw["sprints"]),
            budgets=tuple(Budget.model_validate(row) for row in raw["budgets"]),
            risks=tuple(Risk.model_validate(row) for row in raw["risks"]),
            path=path,
        )
        store._documentation = {
            key: value for key, value in raw.items() if key.startswith("_")
        }
        return store

    @classmethod
    def load(cls, path: Path = DEFAULT_DATASET_PATH) -> Self:
        """Read the fixture from disk and validate every row.

        The path is remembered: it is where a later write goes.
        """
        return cls.from_mapping(
            json.loads(path.read_text(encoding="utf-8")), path=path
        )

    # -- reads -------------------------------------------------------------

    def milestone(self, milestone_id: str) -> Milestone | None:
        """The milestone with that id, or ``None`` -- see the module docstring."""
        return next(
            (m for m in self.milestones if m.milestone_id == milestone_id), None
        )

    def sprint(self, sprint_id: str) -> Sprint | None:
        return next((s for s in self.sprints if s.sprint_id == sprint_id), None)

    def budget(self, project_id: str) -> Budget | None:
        return next((b for b in self.budgets if b.project_id == project_id), None)

    def project(self, project_id: str) -> Project | None:
        return next((p for p in self.projects if p.project_id == project_id), None)

    def risks_for(self, project_id: str) -> tuple[Risk, ...]:
        """Every risk on a project, in the order they were recorded.

        A project with no risks is an empty tuple, not ``None``: "none open" is
        a real, reportable answer, while ``None`` would make the handler decide
        whether it meant "no risks" or "no such project".
        """
        return tuple(risk for risk in self.risks if risk.project_id == project_id)

    def for_project(self, project_code: str) -> "ProjectErp":
        """One project's slice of this store -- see :class:`ProjectErp`."""
        return ProjectErp(store=self, project_code=project_code)

    # -- the one write -----------------------------------------------------

    def create_risk(self, *, project_id: str, title: str, severity: RiskSeverity) -> Risk:
        """Record a new risk, persist it, and return it.

        The id is derived from the count rather than randomly generated, so a
        test can assert on the result and an audit row and a trace entry all
        name the same thing. Nothing here checks permission or approval -- that
        is the gateway's job, and a store that also enforced policy would be a
        second place to look for the rule.

        Persists through :meth:`_flush` before returning, so a caller that was
        told "recorded" can reload the file and find it. A store without a path
        raises :class:`ErpNotPersistedError` instead -- and any flush failure
        rolls the append back, because a store that kept the row in memory
        after refusing to write it would be claiming a write it did not make.

        Holds ``self._lock`` around the append-flush-rollback sequence: the
        web layer can have two approved writes reach this store from two
        request threads, and without a lock one thread's ``len(self.risks)``
        could be read before the other's append lands, handing out the same
        id twice.
        """
        with self._lock:
            created = Risk(
                risk_id=f"R-{len(self.risks) + 1}",
                project_id=project_id,
                title=title,
                severity=severity,
                source_id=f"risk-r-{len(self.risks) + 1}",
            )
            self.risks.append(created)
            try:
                self._flush()
            except BaseException:
                self.risks.pop()
                raise
            return created

    def _flush(self) -> None:
        """Rewrite the file this store was loaded from, atomically.

        The whole file rather than an append, because the dataset is review
        material first: tens of rows, rewritten in milliseconds, still openable
        beside the code that reads it. A ledger of edits would make the file
        a database with none of the tooling.

        The write lands on a temporary file that replaces the real one, so a
        crash mid-write leaves either the old dataset or the new one -- never a
        half-written file that fails the next :meth:`load` and takes the
        assistant down with it.
        """
        if self._path is None:
            raise ErpNotPersistedError(
                "this store was not loaded from a file, so the write has "
                "nowhere to go; a change that only lived in this process's "
                "memory is the defect the flush exists to prevent"
            )
        document = {
            **self._documentation,
            "projects": [row.model_dump() for row in self.projects],
            "milestones": [row.model_dump() for row in self.milestones],
            "sprints": [row.model_dump() for row in self.sprints],
            "budgets": [row.model_dump() for row in self.budgets],
            "risks": [row.model_dump() for row in self.risks],
        }
        temporary = self._path.with_name(self._path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(document, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, self._path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


@dataclass(frozen=True)
class ProjectErp:
    """One project's slice of the store -- the only thing a handler may read.

    Records of another project do not exist through this view: :meth:`milestone`,
    :meth:`sprint`, :meth:`budget`, and :meth:`project` return ``None`` for
    them, and :meth:`risks_for` returns ``()`` -- the same "does not exist for
    you" a filtered document gets (``rag/access.py``). The rule is decided
    here, before a handler sees data, for the reason ``rag/access.py`` filters
    before ranking: a check a handler has to remember to make is a check one
    handler forgets.

    Built by :meth:`MockErp.for_project`, never constructed directly by a
    handler -- the constructor takes the whole store and a project code
    rather than a filtered copy, so a write still lands on the one store
    every other view and every other request shares.
    """

    store: MockErp
    project_code: str

    def milestone(self, milestone_id: str) -> Milestone | None:
        record = self.store.milestone(milestone_id)
        return record if record is not None and record.project_id == self.project_code else None

    def sprint(self, sprint_id: str) -> Sprint | None:
        record = self.store.sprint(sprint_id)
        return record if record is not None and record.project_id == self.project_code else None

    def budget(self, project_id: str) -> Budget | None:
        if project_id != self.project_code:
            return None
        return self.store.budget(project_id)

    def project(self, project_id: str) -> Project | None:
        if project_id != self.project_code:
            return None
        return self.store.project(project_id)

    def risks_for(self, project_id: str) -> tuple[Risk, ...]:
        if project_id != self.project_code:
            return ()
        return self.store.risks_for(project_id)

    def create_risk(self, *, project_id: str, title: str, severity: RiskSeverity) -> Risk:
        """Record a new risk, through the store this view wraps.

        Raises:
            ErpAccessError: ``project_id`` does not match this view's project.
                Defence in depth -- the gateway's project check refuses a
                mismatched call before a handler is ever reached; see
                :class:`ErpAccessError`.
        """
        if project_id != self.project_code:
            raise ErpAccessError(
                f"this view is bound to project {self.project_code!r}; a "
                f"write naming {project_id!r} does not belong to it"
            )
        return self.store.create_risk(
            project_id=project_id, title=title, severity=severity
        )
