"""The generated status blocks and reports must match the machine sources."""

from __future__ import annotations

from tests.docs_inventory import REPO_ROOT
from yolo_agent.tools import docs_status


def test_generated_status_is_fresh() -> None:
    assert docs_status.main(["--check", "--root", str(REPO_ROOT)]) == 0
