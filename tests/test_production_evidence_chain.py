from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.components.adapters.data_pipeline.hard_negative_evidence import (
    TrainHardNegativePrediction,
    TrainHardNegativePredictionBatch,
    TrainSampleIndex,
    TrainSampleIndexRecord,
    produce_train_hard_negative_manifest,
)
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.matched_baseline import (
    MatchedBaselineArtifact,
    build_matched_baseline_artifact,
    verify_paired_baseline,
)


def _index() -> TrainSampleIndex:
    return TrainSampleIndex(
        dataset_manifest_hash="dataset-train-v1",
        samples=[
            TrainSampleIndexRecord(image_id="image-1", sample_index=0),
            TrainSampleIndexRecord(image_id="image-2", sample_index=1),
        ],
    )


def _batch(*, dataset_hash: str = "dataset-train-v1") -> TrainHardNegativePredictionBatch:
    return TrainHardNegativePredictionBatch(
        dataset_manifest_hash=dataset_hash,
        source_run_id="baseline-train-r1",
        baseline_protocol_hash="protocol-train-v1",
        predictions=[
            TrainHardNegativePrediction(
                image_id="image-1",
                predicted_class=3,
                score=0.91,
                bbox=[1.0, 2.0, 10.0, 12.0],
            ),
            TrainHardNegativePrediction(
                image_id="image-1",
                predicted_class=3,
                score=0.71,
                bbox=[2.0, 3.0, 10.0, 12.0],
            ),
        ],
    )


def test_production_replay_manifest_binds_train_index_and_hashes(tmp_path: Path) -> None:
    index = _index()
    batch = _batch()
    output = tmp_path / "hard_negative_manifest.json"

    manifest = produce_train_hard_negative_manifest(
        batch,
        index,
        output_path=output,
    )

    assert output.is_file()
    assert manifest.source_split == "train"
    assert manifest.train_index_hash == index.index_hash
    assert manifest.prediction_artifact_sha256 == batch.batch_hash
    assert manifest.dataset_sample_count == 2
    assert manifest.records[0].sample_index == 0
    assert manifest.records[0].score == 0.91


def test_replay_rejects_dataset_mismatch_and_unknown_train_image() -> None:
    with pytest.raises(ValueError, match="different dataset manifest"):
        produce_train_hard_negative_manifest(_batch(dataset_hash="other"), _index())

    batch = _batch().model_copy(update={
        "predictions": [
            TrainHardNegativePrediction(
                image_id="validation-only",
                predicted_class=1,
                score=0.99,
                bbox=[0.0, 0.0, 1.0, 1.0],
            )
        ],
        "batch_hash": "",
    })
    with pytest.raises(ValueError, match="absent from the train dataset"):
        produce_train_hard_negative_manifest(batch, _index())


def test_replay_prediction_batch_cannot_claim_validation_split() -> None:
    with pytest.raises(ValueError, match="literal"):
        TrainHardNegativePredictionBatch.model_validate({
            "dataset_manifest_hash": "dataset",
            "source_split": "val",
            "source_run_id": "run",
            "baseline_protocol_hash": "protocol",
        })


def test_train_index_rejects_duplicate_or_non_contiguous_indices() -> None:
    with pytest.raises(ValueError, match="duplicate sample indices"):
        TrainSampleIndex(
            dataset_manifest_hash="dataset",
            samples=[
                TrainSampleIndexRecord(image_id="a", sample_index=0),
                TrainSampleIndexRecord(image_id="b", sample_index=0),
            ],
        )

    with pytest.raises(ValueError, match="contiguous"):
        TrainSampleIndex(
            dataset_manifest_hash="dataset",
            samples=[TrainSampleIndexRecord(image_id="a", sample_index=1)],
        )


def _metric(
    *,
    value: float,
    role: str,
    protocol: str = "protocol-v1",
    dataset: str = "dataset-v1",
    run_id: str = "run-v1",
    candidate_id: str = "baseline",
    node_id: str = "node-baseline",
) -> MetricEvidence:
    return MetricEvidence(
        candidate_id=candidate_id,
        node_id=node_id,
        run_id=run_id,
        origin_run_id=run_id,
        evidence_role=role,  # type: ignore[arg-type]
        dataset_manifest_sha256=dataset,
        subset_manifest_sha256="subset-v1",
        split="val2017",
        protocol_hash=protocol,
        eval_protocol_hash="eval-v1",
        seed=42,
        fidelity="pilot_10",
        epochs=10,
        batch_policy_hash="batch-v1",
        ultralytics_version="8.4.0",
        imgsz=640,
        metric_name="map50_95",
        value=value,
        source="test-evidence",
        verified=True,
    )


def test_matched_baseline_artifact_is_file_backed_and_paired(tmp_path: Path) -> None:
    baseline = _metric(value=0.40, role="baseline_reference")
    artifact_path = tmp_path / "matched_baseline.yaml"
    artifact = build_matched_baseline_artifact(
        baseline,
        model_identity="yolo26n.pt",
        output_path=artifact_path,
    )
    assert MatchedBaselineArtifact.from_path(artifact_path).artifact_hash == artifact.artifact_hash

    candidate = _metric(
        value=0.42,
        role="current_observation",
        candidate_id="candidate",
        node_id="node-candidate",
    )
    result = verify_paired_baseline(
        candidate,
        [baseline],
        artifact,
        model_identity="yolo26n.pt",
    )
    assert result.verified
    assert result.paired_delta is not None
    assert result.paired_delta.paired_delta == pytest.approx(0.02)


def test_baseline_artifact_never_turns_protocol_mismatch_into_delta(tmp_path: Path) -> None:
    baseline = _metric(value=0.40, role="baseline_reference")
    artifact = build_matched_baseline_artifact(
        baseline,
        model_identity="yolo26n.pt",
        output_path=tmp_path / "baseline.json",
    )
    candidate = _metric(
        value=0.42,
        role="current_observation",
        protocol="different-protocol",
        candidate_id="candidate",
        node_id="node-candidate",
    )
    result = verify_paired_baseline(
        candidate,
        [baseline],
        artifact,
        model_identity="yolo26n.pt",
    )
    assert not result.verified
    assert result.paired_delta is None
    assert any("protocol_hash_mismatch" in item for item in result.blockers)


def test_single_baseline_record_cannot_be_packaged_from_current_observation() -> None:
    with pytest.raises(ValueError, match="baseline_reference"):
        build_matched_baseline_artifact(
            _metric(value=0.4, role="current_observation"),
            model_identity="yolo26n.pt",
        )
