"""Offline tests for the real paper acceptance research preflight."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


import yolo_agent.certification.paper_auto_optimization_research as research_module
from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchPreparer,
    load_sampling_contract,
)
from yolo_agent.components.maturity_registry import (
    adapter_source_hash,
    installed_ultralytics_version,
)
from yolo_agent.research.maturity_snapshot import (
    EffectiveComponentMaturityManifest,
    FrozenComponentMaturity,
)
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)
from yolo_agent.research.snapshot import freeze_research_snapshot


def test_preparer_selects_frozen_sampling_profile_and_gpu_overlay(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    research_root = tmp_path / "research"
    artifacts = _snapshot_artifacts(tmp_path / "inputs")
    snapshot, snapshot_dir = freeze_research_snapshot(
        research_root,
        artifacts,
        paper_count=1,
        component_count=1,
        recipe_count=1,
        paper_method_coverage_version="coverage-v1",
        effective_maturity_version="maturity-v1",
        maturity_summary={"gpu_certified": 1},
        source_repository="awesome",
        source_commit="commit-1",
        source_catalog_hash="catalog-1",
        importer_version="test",
    )

    class FakeBuilder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def build(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                status="completed",
                snapshot_path=snapshot_dir.as_posix(),
                errors=[],
                unavailable_reason=None,
            )

    monkeypatch.setattr(research_module, "AwesomeSnapshotBuilder", FakeBuilder)
    output = tmp_path / "research_context.yaml"

    context = PaperAcceptanceResearchPreparer(
        research_root=research_root,
        source=tmp_path / "awesome",
        maturity_registry=tmp_path / "maturity.yaml",
    ).prepare(output)

    assert context.snapshot_hash == snapshot.snapshot_hash
    assert context.paper_ids == ["paper-small-object"]
    assert context.method_profile_ids == ["profile-small-object"]
    assert context.component_id == "sampling.small_object"
    assert context.maturity == "gpu_certified"
    assert context.adapter_hash == adapter_source_hash(load_sampling_contract())
    assert output.is_file()


def _snapshot_artifacts(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    simple_names = (
        "papers",
        "component_contracts",
        "recipes",
        "classifications",
        "component_extractions",
        "compatibility_reviews",
        "reproduction_queue",
    )
    artifacts: dict[str, Path] = {}
    for name in simple_names:
        path = root / f"{name}.yaml"
        path.write_text(f"name: {name}\n", encoding="utf-8")
        artifacts[name] = path

    profile = PaperMethodProfile(
        profile_id="profile-small-object",
        paper_id="paper-small-object",
        canonical_component_ids=["sampling.small_object"],
        source_locations=["paper:summary"],
    )
    decision = PaperImplementationDecision(
        paper_id=profile.paper_id,
        profile_id=profile.profile_id,
        decision="reuse_existing_adapter",
        canonical_component_ids=["sampling.small_object"],
        reusable_adapter_ids=["sampling.small_object"],
        source_locations=["paper:summary"],
    ).with_hash()
    coverage = PaperMethodCoverageReport(
        paper_count=1,
        profile_count=1,
        profiles=[profile],
        decisions=[decision],
    )
    coverage_path = root / "paper_method_coverage.yaml"
    coverage.to_yaml(coverage_path, exclude_none=True, sort_keys=False)
    artifacts["paper_method_coverage"] = coverage_path

    maturity = EffectiveComponentMaturityManifest(
        entries=[
            FrozenComponentMaturity(
                component_id="sampling.small_object",
                adapter_hash=adapter_source_hash(load_sampling_contract()),
                code_commit="commit-1",
                ultralytics_version=installed_ultralytics_version(),
                protocol_hash="component-gpu-protocol",
                effective_maturity="gpu_certified",
                runtime_execution_ready=True,
            )
        ]
    )
    maturity_path = root / "effective_component_maturity.yaml"
    maturity.to_yaml(maturity_path, exclude_none=True, sort_keys=False)
    artifacts["effective_component_maturity"] = maturity_path
    return artifacts
