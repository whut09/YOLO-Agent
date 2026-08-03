"""Offline tests for the real paper acceptance research preflight."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


import yolo_agent.certification.paper_auto_optimization_research as research_module
from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchPreparer,
    load_component_contract,
    load_sampling_contract,
)
from yolo_agent.certification.paper_auto_optimization_tracks import (
    PAPER_ACCEPTANCE_RECIPES,
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


def test_preparer_selects_four_frozen_profiles_and_gpu_overlays(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    research_root = tmp_path / "research"
    artifacts = _snapshot_artifacts(tmp_path / "inputs")
    snapshot, snapshot_dir = freeze_research_snapshot(
        research_root,
        artifacts,
        paper_count=4,
        component_count=4,
        recipe_count=4,
        paper_method_coverage_version="coverage-v1",
        effective_maturity_version="maturity-v1",
        maturity_summary={"gpu_certified": 4},
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
    assert context.method_profile_ids == ["profile-sampling"]
    assert context.component_id == "sampling.small_object"
    assert context.maturity == "gpu_certified"
    assert context.adapter_hash == adapter_source_hash(load_sampling_contract())
    assert {item.component_family for item in context.tracks} == {
        "sampling",
        "auxiliary_loss",
        "distillation",
        "model_graph",
    }
    assert all(item.maturity == "gpu_certified" for item in context.tracks)
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

    profiles = [
        PaperMethodProfile(
            profile_id=f"profile-{recipe.track_id}",
            paper_id=(
                "paper-small-object"
                if recipe.track_id == "sampling"
                else f"paper-{recipe.track_id}"
            ),
            canonical_component_ids=[recipe.component_id],
            source_locations=["paper:summary"],
        )
        for recipe in PAPER_ACCEPTANCE_RECIPES
    ]
    decisions = [
        PaperImplementationDecision(
            paper_id=profile.paper_id,
            profile_id=profile.profile_id,
            decision="reuse_existing_adapter",
            canonical_component_ids=list(profile.canonical_component_ids),
            reusable_adapter_ids=list(profile.canonical_component_ids),
            source_locations=["paper:summary"],
        ).with_hash()
        for profile in profiles
    ]
    coverage = PaperMethodCoverageReport(
        paper_count=4,
        profile_count=4,
        profiles=profiles,
        decisions=decisions,
    )
    coverage_path = root / "paper_method_coverage.yaml"
    coverage.to_yaml(coverage_path, exclude_none=True, sort_keys=False)
    artifacts["paper_method_coverage"] = coverage_path

    maturity = EffectiveComponentMaturityManifest(
        entries=[
            FrozenComponentMaturity(
                component_id=recipe.component_id,
                adapter_hash=adapter_source_hash(
                    load_component_contract(recipe.component_id)
                ),
                code_commit="commit-1",
                ultralytics_version=installed_ultralytics_version(),
                protocol_hash="component-gpu-protocol",
                effective_maturity="gpu_certified",
                runtime_execution_ready=True,
            )
            for recipe in PAPER_ACCEPTANCE_RECIPES
        ]
    )
    maturity_path = root / "effective_component_maturity.yaml"
    maturity.to_yaml(maturity_path, exclude_none=True, sort_keys=False)
    artifacts["effective_component_maturity"] = maturity_path
    return artifacts
