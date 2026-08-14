from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_agent.agents.paper_recipe_planner import (
    _missing_recipe_evidence,
    _proposal,
)
from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.adapters.data_pipeline.hard_negative import (
    HardNegativeManifest,
    HardNegativeRecord,
)
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.tools.coco_error_importer import import_coco_eval_metrics
from yolo_agent.tools.coco_error_mining import build_hard_negative_replay_manifest


def _predictions(path: Path) -> None:
    path.write_text(json.dumps([
        {
            "image_id": 11,
            "category_id": 3,
            "bbox": [1, 2, 10, 12],
            "score": 0.91,
        },
        {
            "image_id": 11,
            "category_id": 3,
            "bbox": [2, 3, 10, 12],
            "score": 0.72,
        },
    ]), encoding="utf-8")


def test_coco_train_replay_manifest_is_explicit_and_deduplicated(tmp_path: Path) -> None:
    predictions = tmp_path / "train_predictions.json"
    _predictions(predictions)
    manifest = build_hard_negative_replay_manifest(
        predictions,
        dataset_manifest_hash="train-dataset",
        source_run_id="baseline-r1",
        baseline_protocol_hash="protocol-r1",
        train_image_to_sample_index={11: 4},
    )

    assert manifest.source_split == "train"
    assert manifest.sample_indices == [4]
    assert manifest.records[0].score == 0.91
    assert manifest.evidence_id.endswith(manifest.manifest_hash)

    with pytest.raises(ValueError, match="train-side"):
        build_hard_negative_replay_manifest(
            predictions,
            dataset_manifest_hash="train-dataset",
            source_run_id="baseline-r1",
            baseline_protocol_hash="protocol-r1",
            train_image_to_sample_index=None,
        )


def test_importer_emits_train_evidence_recovery_instead_of_using_val_errors(
    tmp_path: Path,
) -> None:
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"mAP50-95": 0.39}), encoding="utf-8")
    predictions = tmp_path / "train_predictions.json"
    _predictions(predictions)
    store = EvidenceStore(tmp_path / "runs")

    recovery = import_coco_eval_metrics(
        eval_path,
        store,
        run_id="run-1",
        candidate_id="baseline",
        node_id="node-1",
        matched_identity={"protocol_hash": "protocol-r1"},
        hard_negative_predictions_path=predictions,
    )
    assert recovery.hard_negative_evidence_status == "evidence_recovery"
    assert "run_train_split_inference_for_hard_negative_replay" in (
        recovery.evidence_recovery_actions
    )
    assert recovery.hard_negative_manifest_path is None

    ready = import_coco_eval_metrics(
        eval_path,
        store,
        run_id="run-2",
        candidate_id="baseline",
        node_id="node-1",
        matched_identity={"protocol_hash": "protocol-r1"},
        hard_negative_predictions_path=predictions,
        train_image_to_sample_index={11: 0},
        train_dataset_manifest_hash="train-dataset",
        hard_negative_manifest_path=tmp_path / "replay.json",
    )
    assert ready.hard_negative_evidence_status == "ready"
    assert ready.hard_negative_manifest_path is not None
    assert ready.hard_negative_manifest_path.is_file()


def test_planner_marks_missing_train_evidence_and_binds_ready_policy() -> None:
    contracts = load_contracts("configs/components/data_pipeline/paper_data_adapters.yaml")
    registry = RecipeRegistry.from_path(
        "configs/recipes/yolo26_data_pipeline.yaml",
        component_contracts=contracts,
    )
    recipe = registry.get("yolo26_hard_negative_replay")
    assert recipe is not None

    missing = _missing_recipe_evidence(recipe, [], None, {})
    assert "recover_train_hard_negative_evidence" in missing
    assert "bind_train_dataset_manifest_hash" in missing

    proposal = _proposal(
        recipe,
        [],
        set(),
        {
            "hard_negative_manifest_path": "runs/run-1/artifacts/replay.json",
            "hard_negative_manifest_hash": "manifest-hash",
            "train_dataset_manifest_hash": "train-dataset",
            "hard_negative_baseline_protocol_hash": "protocol-r1",
            "hard_negative_evidence_id": "hard_negative_replay:manifest-hash",
        },
    )
    assert proposal.train_overrides["manifest_path"].endswith("replay.json")
    assert proposal.train_overrides["manifest_hash"] == "manifest-hash"
    assert proposal.train_overrides["evidence_id"] == (
        "hard_negative_replay:manifest-hash"
    )


def test_manifest_rejects_duplicate_sample_indices() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        HardNegativeManifest.from_records(
            dataset_manifest_hash="train",
            source_run_id="run",
            baseline_protocol_hash="protocol",
            records=[
                HardNegativeRecord(image_id="a", sample_index=1, error_type="fp"),
                HardNegativeRecord(image_id="b", sample_index=1, error_type="fp"),
            ],
        )
