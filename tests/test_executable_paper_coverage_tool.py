"""CLI tests for executable paper coverage generation."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.research.method_profiles import PaperMethodProfileBuilder
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.research.schemas import PaperRecord
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.tools.executable_paper_coverage import main


def test_live_cli_writes_yaml_and_markdown_with_explicit_denominators(
    tmp_path: Path,
    capsys,
) -> None:
    research = tmp_path / "research"
    registry = PaperRegistry(research)
    registry.add(
        PaperRecord(
            paper_id="sampling-paper",
            title="Sampling",
            year=2025,
            component_ids=["small_object_sampling"],
            applicability="direct_adapter_candidate",
        )
    )
    resolver = ComponentAliasResolver.from_yaml()
    method = PaperMethodProfileBuilder(resolver).build(registry.list())
    method_path = tmp_path / "paper_method_coverage.yaml"
    method.to_yaml(method_path)
    maturity = tmp_path / "maturity.yaml"
    maturity.write_text(
        "schema_version: component_maturity_registry.v1\noverlays: []\n",
        encoding="utf-8",
    )
    output = tmp_path / "coverage_baseline.yaml"

    result = main(
        [
            "--method-coverage",
            str(method_path),
            "--maturity-registry",
            str(maturity),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    terminal = capsys.readouterr().out
    assert "all_papers: 1" in terminal
    assert "yolo26_compatible_papers: 1" in terminal
    assert "runtime_ready_papers: 0" in terminal

