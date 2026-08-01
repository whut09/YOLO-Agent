"""Artifact tests for executable paper coverage reports."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.research.executable_coverage_report import (
    render_executable_coverage_markdown,
    write_executable_coverage_artifacts,
)
from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
    PaperCoverageDenominator,
    PaperExecutableCoverageEntry,
)


def _baseline() -> ExecutablePaperCoverageBaseline:
    entry = PaperExecutableCoverageEntry(
        paper_id="paper|one",
        profile_id="profile-one",
        decision="reuse_existing_adapter",
        compatibility_class="yolo26_runtime_ready",
        adaptation_scope="single_component",
        canonical_mechanisms=["sampling.small_object"],
        reusable_adapter_candidates=["sampling.small_object"],
        runtime_ready_adapters=["sampling.small_object"],
        required_runtime_hooks=["build_train_dataloader"],
        source_locations=["summary"],
    )
    denominators = {}
    for name in (
        "all_papers",
        "yolo26_compatible_papers",
        "adaptable_component_papers",
    ):
        denominators[name] = PaperCoverageDenominator(
            name=name,  # type: ignore[arg-type]
            definition=f"definition for {name}",
            paper_count=1,
            paper_ids=["paper|one"],
        )
    denominators["exact_reproduction_candidates"] = PaperCoverageDenominator(
        name="exact_reproduction_candidates",
        definition="exact definition",
    )
    return ExecutablePaperCoverageBaseline(
        source_method_coverage_hash="m" * 64,
        denominators=denominators,
        runtime_ready_paper_count=1,
        reusable_adapter_paper_count=1,
        entries=[entry],
    )


def test_markdown_contains_four_denominators_and_every_paper_field() -> None:
    markdown = render_executable_coverage_markdown(_baseline())

    assert "`all_papers`" in markdown
    assert "`exact_reproduction_candidates`" in markdown
    assert "`paper\\|one`" in markdown
    assert "yolo26_runtime_ready" in markdown
    assert "sampling.small_object" in markdown
    assert "build_train_dataloader" in markdown
    assert "Component adaptation is not exact paper reproduction" in markdown


def test_artifacts_are_written_and_yaml_roundtrips(tmp_path: Path) -> None:
    baseline = _baseline()
    yaml_path, markdown_path = write_executable_coverage_artifacts(
        baseline,
        yaml_path=tmp_path / "coverage_baseline.yaml",
        markdown_path=tmp_path / "coverage_baseline.md",
    )

    loaded = ExecutablePaperCoverageBaseline.from_yaml(yaml_path)
    assert loaded.report_hash == baseline.report_hash
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Executable Paper Coverage Baseline"
    )
    assert not list(tmp_path.glob("*.tmp"))

