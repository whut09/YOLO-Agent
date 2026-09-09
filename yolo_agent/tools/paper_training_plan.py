"""CLI wrapper for preparing a paper training plan without executing it."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.research.paper_training_plan import (
    PaperTrainingPlan,
    build_paper_training_plan,
)


def run_paper_training_plan(
    *,
    run_id: str,
    run_root: Path | str = Path("runs"),
    inventory_path: Path | str = Path(
        "runs/coverage-audit/paper_execution_inventory.yaml"
    ),
    requirements_path: Path | str = Path(
        "runs/coverage-audit/paper_execution_requirements.yaml"
    ),
    assets_path: Path | str = Path("runs/paper-readiness/paper_asset_registry.yaml"),
    readiness_path: Path | str = Path(
        "runs/paper-readiness/paper_readiness_report.yaml"
    ),
    model: str = "yolo26n.pt",
    data: Path | str = Path(r"E:\datatset\coco.yaml"),
    output_path: Path | str | None = None,
    maturity_registry_path: Path | str | None = Path(
        "runs/component_maturity_registry.yaml"
    ),
) -> PaperTrainingPlan:
    """Prepare a paired paper cohort; this function never starts training."""

    return build_paper_training_plan(
        run_id=run_id,
        run_root=run_root,
        inventory_path=inventory_path,
        requirements_path=requirements_path,
        assets_path=assets_path,
        readiness_path=readiness_path,
        model=model,
        data=data,
        output_path=output_path,
        maturity_registry_path=maturity_registry_path,
    )


__all__ = ["run_paper_training_plan"]
