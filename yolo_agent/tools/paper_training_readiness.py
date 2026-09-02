"""CLI wrapper for the final paper-training readiness gate."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.paper_training_readiness import (
    PaperTrainingReadinessReport,
    build_paper_training_readiness,
)


def run_paper_training_readiness(
    *,
    inventory_path: Path | str,
    requirements_path: Path | str,
    assets_path: Path | str,
    readiness_path: Path | str,
    asha_path: Path | str,
    output_path: Path | str = Path(
        "runs/paper-readiness/paper_training_readiness.yaml"
    ),
    expected_paper_count: int = 83,
) -> PaperTrainingReadinessReport:
    """Run the offline gate; this function never starts or probes training."""
    return build_paper_training_readiness(
        inventory_path=inventory_path,
        requirements_path=requirements_path,
        assets_path=assets_path,
        readiness_path=readiness_path,
        asha_path=asha_path,
        output_path=output_path,
        expected_paper_count=expected_paper_count,
    )


__all__ = ["run_paper_training_readiness"]
