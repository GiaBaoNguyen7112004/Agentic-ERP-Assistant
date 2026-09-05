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

Writes are in memory
--------------------

:meth:`MockErp.create_risk` appends to this instance and never touches the file
on disk. A test run must not edit the repo's fixture, and two tests must not be
able to see each other's writes -- so each one builds its own store.
"""

import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Budget",
    "DEFAULT_DATASET_PATH",
    "Milestone",
    "MockErp",
    "Project",
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


class MockErp:
    """One project's delivery data, in memory, for the tools to work against.

    Not frozen, because :meth:`create_risk` is the write the whole approval
    story exists for and it has to change something observable. Everything it
    changes lives on the instance, so a test asserts on its own store and no
    test can see another's write.
    """

    def __init__(
        self,
        *,
        projects: tuple[Project, ...],
        milestones: tuple[Milestone, ...],
        sprints: tuple[Sprint, ...],
        budgets: tuple[Budget, ...],
        risks: tuple[Risk, ...],
    ) -> None:
        self.projects = projects
        self.milestones = milestones
        self.sprints = sprints
        self.budgets = budgets
        self.risks = list(risks)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> Self:
        """Build a store from already-parsed JSON.

        Separate from :meth:`load` so a test can hand in three rows instead of
        depending on the repo's fixture staying the shape it expects. Keys
        beginning with an underscore are documentation for the human reading
        the file and are ignored here.
        """
        return cls(
            projects=tuple(Project.model_validate(row) for row in raw["projects"]),
            milestones=tuple(
                Milestone.model_validate(row) for row in raw["milestones"]
            ),
            sprints=tuple(Sprint.model_validate(row) for row in raw["sprints"]),
            budgets=tuple(Budget.model_validate(row) for row in raw["budgets"]),
            risks=tuple(Risk.model_validate(row) for row in raw["risks"]),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_DATASET_PATH) -> Self:
        """Read the fixture from disk and validate every row."""
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

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

    # -- the one write -----------------------------------------------------

    def create_risk(self, *, project_id: str, title: str, severity: RiskSeverity) -> Risk:
        """Record a new risk and return it.

        The id is derived from the count rather than randomly generated, so a
        test can assert on the result and an audit row and a trace entry all
        name the same thing. Nothing here checks permission or approval -- that
        is the gateway's job, and a store that also enforced policy would be a
        second place to look for the rule.
        """
        created = Risk(
            risk_id=f"R-{len(self.risks) + 1}",
            project_id=project_id,
            title=title,
            severity=severity,
            source_id=f"risk-r-{len(self.risks) + 1}",
        )
        self.risks.append(created)
        return created
