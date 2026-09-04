"""CLI-facing orchestration for the current paper training cohort."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.research.paper_training_cohort import (
    PaperTrainingCohort,
    PaperTrainingCohortBuilder,
)


def run_paper_training_cohort(
    *,
    inventory_path: Path | str,
    requirements_path: Path | str,
    assets_path: Path | str,
    readiness_path: Path | str,
    asha_path: Path | str,
    output_path: Path | str = Path(
        "runs/paper-readiness/paper_training_cohort.yaml"
    ),
    expected_paper_count: int = 83,
) -> PaperTrainingCohort:
    """Build the cohort without starting training or probing a GPU."""

    return PaperTrainingCohortBuilder().build(
        inventory_path=inventory_path,
        requirements_path=requirements_path,
        assets_path=assets_path,
        readiness_path=readiness_path,
        asha_path=asha_path,
        output_path=output_path,
        expected_paper_count=expected_paper_count,
    )


__all__ = ["run_paper_training_cohort"]
