"""CLI-facing orchestration for the offline paper readiness audit."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.paper_readiness import (
    PaperReadinessPreflight,
    PaperReadinessReport,
)
from yolo_agent.research.paper_execution_schemas import PaperExecutionInventory
from yolo_agent.tools.paper_execution_inventory import build_paper_execution_inventory


def run_paper_readiness(
    *,
    research_root: Path | str,
    registry_path: Path | str,
    model: str,
    data: Path | str,
    output_path: Path | str = Path("runs/paper-readiness/paper_readiness_report.yaml"),
    inventory_path: Path | str = Path("runs/coverage-audit/paper_execution_inventory.yaml"),
    certification_root: Path | str | None = None,
    expected_compatible_count: int = 83,
    run_cpu_certification: bool = True,
) -> PaperReadinessReport:
    inventory_file = Path(inventory_path)
    if inventory_file.is_file():
        try:
            inventory = PaperExecutionInventory.from_yaml(inventory_file)
        except (OSError, TypeError, ValueError):
            inventory = build_paper_execution_inventory(
                research_root=research_root,
                maturity_registry=registry_path,
                yaml_path=inventory_file,
                markdown_path=inventory_file.with_suffix(".md"),
                expected_compatible_count=expected_compatible_count,
            )
    else:
        inventory = build_paper_execution_inventory(
            research_root=research_root,
            maturity_registry=registry_path,
            yaml_path=inventory_file,
            markdown_path=inventory_file.with_suffix(".md"),
            expected_compatible_count=expected_compatible_count,
        )
    if inventory.compatible_paper_count != expected_compatible_count:
        raise ValueError(
            f"paper readiness requires {expected_compatible_count} compatible papers; "
            f"got {inventory.compatible_paper_count}"
        )
    return PaperReadinessPreflight().run(
        inventory=inventory,
        registry_path=registry_path,
        model=model,
        data=data,
        output_path=output_path,
        certification_root=certification_root,
        run_cpu_certification=run_cpu_certification,
    )


__all__ = ["run_paper_readiness"]
