"""Build the paper-level execution inventory from offline research inputs."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.research.paper_execution_inventory import (
    PaperExecutionInventoryBuilder,
    write_paper_execution_inventory_artifacts,
)
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.paper_mechanism_resolver import PaperMechanismResolver
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.resources import ResourcePaths
from yolo_agent.tools.executable_paper_coverage import (
    build_executable_coverage_baseline,
)
from yolo_agent.tools.maturity_effective_contracts import (
    load_effective_contracts,
)


def build_paper_execution_inventory(
    *,
    research_root: Path | str = Path("research"),
    method_coverage: Path | str | None = None,
    maturity_registry: Path | str = Path("runs/component_maturity_registry.yaml"),
    yaml_path: Path | str = Path("runs/coverage-audit/paper_execution_inventory.yaml"),
    markdown_path: Path | str = Path("runs/coverage-audit/paper_execution_inventory.md"),
    expected_compatible_count: int | None = 83,
    frozen_manifest_path: Path | str | None = None,
):
    """Build and persist one record for each compatible paper.

    The executable coverage baseline is rebuilt from the method report so a
    stale coverage-audit artifact cannot silently authorize this inventory.
    """
    root = Path(research_root)
    method_path = Path(method_coverage) if method_coverage else root / "production" / "paper_method_coverage.yaml"
    executable = build_executable_coverage_baseline(
        research_root=root,
        method_coverage=method_path,
        maturity_registry=maturity_registry,
    )
    method = PaperMethodCoverageReport.from_yaml(method_path)
    recipes = RecipeRegistry.from_paths(
        [ResourcePaths.RECIPE_BUNDLES, *sorted(ResourcePaths.RECIPES_DIR.glob("*.yaml"))],
        strict=False,
    )
    # The mechanism resolver must see the same effective contracts the coverage
    # audit sees: source contracts merged with the machine-local maturity
    # overlays that certification persisted.  Without the overlay merge every
    # resolution reports runtime_execution_ready=False and the inventory can
    # never express a runtime_ready disposition even when the per-component
    # non-mock smoke artifacts exist and the readiness pipeline uses them.
    resolver = PaperMechanismResolver.from_alias_config(
        ComponentAliasResolver.from_yaml().config,
        contracts=load_effective_contracts(maturity_registry).values(),
    )
    inventory = PaperExecutionInventoryBuilder(mechanism_resolver=resolver).build(
        method,
        executable,
        PaperRegistry(root).list(),
        recipes.list(),
        expected_compatible_count=expected_compatible_count,
        frozen_manifest_path=frozen_manifest_path,
    )
    write_paper_execution_inventory_artifacts(
        inventory,
        yaml_path=yaml_path,
        markdown_path=markdown_path,
    )
    return inventory


__all__ = ["build_paper_execution_inventory"]
