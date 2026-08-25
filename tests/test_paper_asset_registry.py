"""CPU-only tests for real paper asset registration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from yolo_agent.research.paper_asset_registry import PaperAssetRegistryBuilder
from yolo_agent.research.paper_asset_schemas import PaperAssetRecord
from yolo_agent.research.paper_execution_inventory import PaperExecutionInventory
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirement,
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_execution_schemas import PaperExecutionSpec


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    mechanism: str = "loss.quality.correlation",
    required_domain_assets: list[str] | None = None,
    required_teacher_assets: list[str] | None = None,
    required_manifest_assets: list[str] | None = None,
    required_graph_assets: list[str] | None = None,
    route: str = "training",
) -> tuple[PaperExecutionInventory, PaperExecutionRequirementsMatrix]:
    paper = PaperExecutionSpec(
        paper_id="fixture:paper",
        profile_id="fixture:profile",
        title="Real asset fixture",
        source_locations=["fixture"],
        canonical_component_ids=[mechanism],
        paper_specific_mechanism_ids=[mechanism],
        recipe_ids=["fixture_recipe"],
        execution_fingerprint="1" * 64,
        current_disposition="queued",
        disposition_reason="fixture requires real asset validation",
    )
    inventory = PaperExecutionInventory(
        source_method_coverage_hash="2" * 64,
        all_paper_count=1,
        compatible_paper_count=1,
        exact_reproduction_candidates=0,
        records=[paper],
    ).with_hash()
    requirement = PaperExecutionRequirement(
        paper_id=paper.paper_id,
        paper_specific_mechanism=mechanism,
        paper_specific_mechanism_ids=[mechanism],
        execution_route=route,  # type: ignore[arg-type]
        required_adapter="fixture.adapter" if route != "inference" else None,
        required_changed_variables=["fixture.weight"],
        required_runtime_payload={"mode": "real"},
        required_evidence=["matched_control"],
        required_dataset_protocol={"imgsz": 640, "split": "train"},
        required_teacher_assets=required_teacher_assets or [],
        required_domain_assets=required_domain_assets or [],
        required_manifest_assets=required_manifest_assets or [],
        required_graph_assets=required_graph_assets or [],
        compatible_with_yolo26=True,
        training_candidate_allowed=route == "training",
        exact_blocker=None if route == "training" else "inference_only",
        recovery_action="provide the missing real asset",
        recipe_ids=["fixture_recipe"],
        current_disposition="runtime_ready" if route == "training" else "incompatible",
        protocol_hash="3" * 64,
        execution_fingerprint=paper.execution_fingerprint,
    )
    requirements = PaperExecutionRequirementsMatrix(
        source_inventory_path="fixture/inventory.yaml",
        source_inventory_hash=inventory.inventory_hash,
        compatible_paper_count=1,
        requirements=[requirement],
        generated_at="2026-01-01T00:00:00Z",
    )
    return inventory, requirements


def _write(path: Path, payload: object) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _source_files(
    tmp_path: Path,
    inventory: PaperExecutionInventory,
    requirements: PaperExecutionRequirementsMatrix,
) -> tuple[Path, Path]:
    inventory_path = tmp_path / "inventory.yaml"
    requirements_path = tmp_path / "requirements.yaml"
    inventory.to_yaml(inventory_path, sort_keys=False)
    requirements.to_yaml(requirements_path, sort_keys=False)
    return inventory_path, requirements_path


def _all_assets(tmp_path: Path) -> dict[str, Path]:
    source = _write(tmp_path / "source.yaml", {"name": "source", "split": "train"})
    target = _write(tmp_path / "target.yaml", {"name": "target", "split": "train"})
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"frozen teacher checkpoint")
    replay = _write(
        tmp_path / "hard_negative.yaml",
        {"source_split": "train", "records": [{"image_id": 1, "split": "train"}]},
    )
    graph = _write(tmp_path / "graph.yaml", {"graph_identity": "fixture"})
    baseline = _write(
        tmp_path / "baseline.yaml",
        {"protocol_hash": "3" * 64, "imgsz": 640, "dataset_manifest_hash": _sha256(source)},
    )
    return {
        "source_dataset_manifest": source,
        "target_dataset_manifest": target,
        "teacher_checkpoint": teacher,
        "hard_negative_manifest": replay,
        "graph_config": graph,
        "matched_baseline_artifact": baseline,
    }


def test_complete_real_assets_are_available(tmp_path: Path) -> None:
    inventory, requirements = _fixture(
        tmp_path,
        mechanism="sampling.hard_negative_replay",
        required_teacher_assets=["none"],
        required_domain_assets=["none"],
        required_manifest_assets=["train_replay"],
        required_graph_assets=["graph"],
    )
    assets = _all_assets(tmp_path)
    inventory_path, requirements_path = _source_files(tmp_path, inventory, requirements)
    registry = PaperAssetRegistryBuilder().build(
        inventory,
        requirements,
        source_inventory_path=inventory_path,
        source_requirements_path=requirements_path,
        assets_by_paper={inventory.records[0].paper_id: assets},
    )
    record = registry.records[0]
    assert record.availability == "available"
    assert record.exact_blocker == ""
    assert record.asset_sha256
    assert record.teacher_sha256 == _sha256(assets["teacher_checkpoint"])
    assert Path(record.source_dataset_manifest or "").is_absolute()
    assert Path(record.target_dataset_manifest or "").is_absolute()
    assert Path(record.teacher_checkpoint or "").is_absolute()


def test_missing_paths_are_unavailable_without_recording_fake_paths(tmp_path: Path) -> None:
    inventory, requirements = _fixture(
        tmp_path,
        mechanism="feature_distillation",
        required_teacher_assets=["frozen_teacher_checkpoint"],
    )
    missing = tmp_path / "does-not-exist.pt"
    inventory_path, requirements_path = _source_files(tmp_path, inventory, requirements)
    registry = PaperAssetRegistryBuilder().build(
        inventory,
        requirements,
        source_inventory_path=inventory_path,
        source_requirements_path=requirements_path,
        assets_by_paper={inventory.records[0].paper_id: {"teacher_checkpoint": missing}},
    )
    record = registry.records[0]
    assert record.availability == "unavailable"
    assert record.teacher_checkpoint is None
    assert "teacher_checkpoint_missing" in record.exact_blocker
    assert "matched_baseline_artifact_missing" in record.exact_blocker


def test_teacher_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    with pytest.raises(ValidationError, match="teacher_sha256"):
        PaperAssetRecord(
            paper_id="fixture:paper",
            mechanism_id="feature_distillation",
            teacher_checkpoint=str(teacher.resolve()),
            teacher_sha256="0" * 64,
            asset_hashes={"teacher_checkpoint": _sha256(teacher)},
            protocol_hash="3" * 64,
            availability="unavailable",
            exact_blocker="teacher_hash_mismatch",
            recovery_action="replace the teacher checkpoint",
            current_disposition="blocked_runtime",
        )


def test_source_and_target_same_is_rejected(tmp_path: Path) -> None:
    inventory, requirements = _fixture(
        tmp_path,
        mechanism="feature_alignment",
        required_domain_assets=["source", "target"],
    )
    source = _write(tmp_path / "domain.yaml", {"split": "train"})
    inventory_path, requirements_path = _source_files(tmp_path, inventory, requirements)
    with pytest.raises(ValidationError, match="source and target"):
        PaperAssetRegistryBuilder().build(
            inventory,
            requirements,
            source_inventory_path=inventory_path,
            source_requirements_path=requirements_path,
            assets_by_paper={
                inventory.records[0].paper_id: {
                    "source_dataset_manifest": source,
                    "target_dataset_manifest": source,
                }
            },
        )


def test_validation_manifest_cannot_be_used_for_train_replay(tmp_path: Path) -> None:
    inventory, requirements = _fixture(
        tmp_path,
        mechanism="sampling.hard_negative_replay",
        required_manifest_assets=["train_replay"],
    )
    replay = _write(
        tmp_path / "validation_replay.yaml",
        {"source_split": "val", "records": [{"image_id": 1, "split": "val"}]},
    )
    inventory_path, requirements_path = _source_files(tmp_path, inventory, requirements)
    registry = PaperAssetRegistryBuilder().build(
        inventory,
        requirements,
        source_inventory_path=inventory_path,
        source_requirements_path=requirements_path,
        assets_by_paper={inventory.records[0].paper_id: {"hard_negative_manifest": replay}},
    )
    assert registry.records[0].availability == "unavailable"
    assert "hard_negative_manifest_not_train_split" in registry.records[0].exact_blocker


def test_matched_baseline_protocol_mismatch_blocks_availability(tmp_path: Path) -> None:
    inventory, requirements = _fixture(tmp_path)
    source = _write(tmp_path / "source.yaml", {"split": "train"})
    baseline = _write(tmp_path / "baseline.yaml", {"protocol_hash": "4" * 64, "imgsz": 640})
    inventory_path, requirements_path = _source_files(tmp_path, inventory, requirements)
    registry = PaperAssetRegistryBuilder().build(
        inventory,
        requirements,
        source_inventory_path=inventory_path,
        source_requirements_path=requirements_path,
        assets_by_paper={
            inventory.records[0].paper_id: {
                "source_dataset_manifest": source,
                "matched_baseline_artifact": baseline,
            }
        },
    )
    assert registry.records[0].availability == "unavailable"
    assert "matched_baseline_protocol_mismatch" in registry.records[0].exact_blocker
