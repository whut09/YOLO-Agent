from __future__ import annotations

from pathlib import Path

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.method_profiles import PaperMethodProfileBuilder
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.tools.paper_execution_inventory import (
    build_paper_execution_inventory,
)


def test_build_inventory_service_writes_a_cpu_audit(tmp_path: Path) -> None:
    research = tmp_path / "research"
    registry = PaperRegistry(research)
    registry.add(
        PaperRecord(
            paper_id="sampling-paper",
            title="Sampling paper",
            year=2025,
            component_ids=["small_object_sampling"],
            applicability="direct_adapter_candidate",
        )
    )
    method_path = research / "production" / "paper_method_coverage.yaml"
    method_path.parent.mkdir(parents=True, exist_ok=True)
    PaperMethodProfileBuilder(ComponentAliasResolver.from_yaml()).build(
        registry.list()
    ).to_yaml(method_path)
    maturity = tmp_path / "maturity.yaml"
    maturity.write_text(
        "schema_version: component_maturity_registry.v1\noverlays: []\n",
        encoding="utf-8",
    )

    inventory = build_paper_execution_inventory(
        research_root=research,
        method_coverage=method_path,
        maturity_registry=maturity,
        yaml_path=tmp_path / "inventory.yaml",
        markdown_path=tmp_path / "inventory.md",
        expected_compatible_count=None,
    )

    assert inventory.compatible_paper_count == 1
    assert (tmp_path / "inventory.yaml").is_file()
    assert (tmp_path / "inventory.md").is_file()
