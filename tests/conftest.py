"""Fixtures shared by more than one test area.

Only one today, and it exists because of a change in what a write means: a
``create_risk`` really reaches the file the store was loaded from now, so a
test that runs one has to aim it somewhere disposable. The repo's fixture is
review material -- a suite that wrote to it would leave the dataset in
whatever state the last test to run left it in, and the next reader of
``data/erp/project.json`` would be reading test exhaust.
"""

from pathlib import Path

import pytest

from agentic_erp_assistant.erp.mock import DEFAULT_DATASET_PATH


@pytest.fixture
def erp_file(tmp_path: Path) -> Path:
    """A copy of the repo's ERP fixture in a temp directory, safe to write to.

    A copy rather than a synthesized dataset: the tests this fixture serves
    exercise the real read-validate-write loop against the real file's shape,
    including the ``_readme`` a rewrite has to carry across.
    """
    target = tmp_path / "project.json"
    target.write_text(DEFAULT_DATASET_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return target