"""Active-learning mining tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_agent.agents.active_learning import (
    ActiveLearningMiner,
    MiningConfig,
    PredictionSummary,
    disagreement_rate,
    increment_dataset_version,
    normalized_entropy,
)


def test_low_confidence_high_entropy_and_disagreement_mining() -> None:
    """Miner should support the three core active-learning strategies."""
    miner = ActiveLearningMiner(
        MiningConfig(
            low_confidence_threshold=0.4,
            high_entropy_threshold=0.8,
            disagreement_threshold=0.3,
            max_samples=10,
        )
    )
    predictions = [
        PredictionSummary(
            image_path=Path("unlabeled/low_conf.jpg"),
            max_confidence=0.2,
            class_probabilities=[0.34, 0.33, 0.33],
            model_predictions=["cat", "dog", "cat"],
        ),
        PredictionSummary(
            image_path=Path("unlabeled/easy.jpg"),
            max_confidence=0.95,
            class_probabilities=[0.98, 0.01, 0.01],
            model_predictions=["cat", "cat", "cat"],
        ),
    ]

    plan = miner.mine(predictions, dataset_version="v1", labeling_target="label_studio")

    assert plan.next_dataset_version == "v2"
    assert len(plan.mined_samples) == 1
    sample = plan.mined_samples[0]
    assert sample.image_path == Path("unlabeled/low_conf.jpg")
    assert {"low_confidence", "high_entropy", "model_disagreement"} <= set(sample.reasons)
    assert plan.strategy_counts["low_confidence"] == 1
    assert plan.labeling_manifest.target == "label_studio"


def test_manifest_writes_json(tmp_path: Path) -> None:
    """Labeling manifests should serialize for CVAT/Label Studio handoff."""
    miner = ActiveLearningMiner(MiningConfig(max_samples=1))
    plan = miner.mine(
        [
            PredictionSummary(
                image_path=Path("unlabeled/a.jpg"),
                max_confidence=0.1,
            )
        ],
        dataset_version="dataset_v9",
        labeling_target="cvat",
    )
    output_path = tmp_path / "labeling_manifest.json"

    plan.labeling_manifest.to_json(output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["target"] == "cvat"
    assert data["dataset_version"] == "dataset_v9"
    assert data["next_dataset_version"] == "dataset_v10"
    assert data["samples"][0]["image_path"] == "unlabeled/a.jpg"


def test_load_prediction_summaries_accepts_wrapped_json(tmp_path: Path) -> None:
    """Prediction loader should accept the loop CLI JSON shape."""
    from yolo_agent.agents.active_learning import load_prediction_summaries

    path = tmp_path / "unlabeled_predictions.json"
    path.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "image_path": "unlabeled/a.jpg",
                        "max_confidence": 0.2,
                        "class_probabilities": [0.5, 0.5],
                        "model_predictions": ["a", "b"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    predictions = load_prediction_summaries(path)

    assert len(predictions) == 1
    assert predictions[0].image_path == Path("unlabeled/a.jpg")


def test_entropy_disagreement_and_version_helpers() -> None:
    """Helper functions should be deterministic and bounded."""
    assert round(normalized_entropy([1 / 3, 1 / 3, 1 / 3]), 6) == 1.0
    assert normalized_entropy([1.0, 0.0, 0.0]) == 0.0
    assert disagreement_rate(["a", "b", "b"]) == pytest.approx(1 / 3)
    assert increment_dataset_version("v1") == "v2"
    assert increment_dataset_version("dataset") == "dataset_v2"
