"""CLI-facing orchestration for the offline paper readiness audit."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

from yolo_agent.certification.paper_readiness import (
    PaperReadinessPreflight,
    PaperReadinessReport,
)
from yolo_agent.research.paper_execution_schemas import PaperExecutionInventory
from yolo_agent.research.paper_execution_requirements import (
    build_paper_execution_requirements,
)
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_asset_registry import (
    build_paper_asset_registry,
)
from yolo_agent.research.paper_asset_schemas import PaperAssetRegistry
from yolo_agent.tools.paper_execution_inventory import build_paper_execution_inventory


def run_paper_readiness(
    *,
    research_root: Path | str,
    registry_path: Path | str,
    model: str,
    data: Path | str,
    output_path: Path | str = Path("runs/paper-readiness/paper_readiness_report.yaml"),
    inventory_path: Path | str = Path("runs/coverage-audit/paper_execution_inventory.yaml"),
    requirements_path: Path | str | None = None,
    assets_path: Path | str | None = None,
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
    requirements_file = Path(
        requirements_path
        if requirements_path is not None
        else inventory_file.parent / "paper_execution_requirements.yaml"
    ).resolve()
    if requirements_file.is_file():
        requirements = PaperExecutionRequirementsMatrix.from_yaml(requirements_file)
        if requirements.source_inventory_hash != inventory.inventory_hash:
            raise ValueError(
                "requirements source inventory hash does not match the loaded inventory"
            )
    else:
        requirements = build_paper_execution_requirements(
            inventory_path=inventory_file,
            output_path=requirements_file,
        )
    if requirements.compatible_paper_count != inventory.compatible_paper_count:
        raise ValueError("requirements do not cover the complete inventory")
    requirements_hash = hashlib.sha256(
        json.dumps(
            requirements.model_dump(mode="json", exclude={"generated_at"}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        ).hexdigest()
    assets_file = Path(
        assets_path
        if assets_path is not None
        else Path("runs/paper-readiness/paper_asset_registry.yaml")
    ).resolve()
    if assets_file.is_file():
        assets = PaperAssetRegistry.from_yaml(assets_file)
        if assets.source_inventory_hash != inventory.inventory_hash:
            raise ValueError(
                "asset registry source inventory hash does not match the loaded inventory"
            )
        if assets.source_requirements_hash != _file_hash(requirements_file):
            raise ValueError(
                "asset registry source requirements hash does not match the loaded requirements"
            )
    else:
        assets = build_paper_asset_registry(
            inventory_path=inventory_file,
            requirements_path=requirements_file,
            output_path=assets_file,
            dataset_manifest=data,
        )
    return PaperReadinessPreflight().run(
        inventory=inventory,
        registry_path=registry_path,
        model=model,
        data=data,
        output_path=output_path,
        certification_root=certification_root,
        run_cpu_certification=run_cpu_certification,
        requirements_hash=requirements_hash,
        requirements_path=requirements_file,
        requirements= requirements,
        asset_registry=assets,
        asset_registry_path=assets_file,
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["run_paper_readiness"]
