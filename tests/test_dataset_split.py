"""Dataset split planner tests."""

from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.core.dataset_split import DatasetSplitPlanner


def _make_split_dataset(root: Path) -> Path:
    for split in ["train", "val"]:
        (root / "images" / split / "scene_a").mkdir(parents=True)
        (root / "labels" / split / "scene_a").mkdir(parents=True)
    (root / "images" / "train" / "scene_b").mkdir(parents=True)
    (root / "labels" / "train" / "scene_b").mkdir(parents=True)

    duplicated = b"same-frame"
    (root / "images" / "train" / "scene_a" / "same.jpg").write_bytes(duplicated)
    (root / "images" / "val" / "scene_a" / "same.jpg").write_bytes(duplicated)
    (root / "images" / "train" / "scene_b" / "small.jpg").write_bytes(b"small")

    (root / "labels" / "train" / "scene_a" / "same.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (root / "labels" / "val" / "scene_a" / "same.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (root / "labels" / "train" / "scene_b" / "small.txt").write_text("1 0.2 0.2 0.03 0.03\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "names:",
                "  - car",
                "  - cone",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_dataset_split_planner_detects_duplicates_leakage_and_scene_groups(tmp_path: Path) -> None:
    """Split planner should detect duplicate frames and train/val leakage."""
    data_yaml = _make_split_dataset(tmp_path / "dataset")

    plan = DatasetSplitPlanner().analyze(data_yaml)

    assert len(plan.samples) == 3
    assert len(plan.duplicates) == 1
    assert len(plan.leakage) == 1
    assert "scene_a" in plan.scene_distribution
    assert any(sample.has_small_object for sample in plan.samples)
    assert any("leakage" in item for item in plan.recommendations)
    assert plan.assignments


def test_dataset_split_plan_serializes_json(tmp_path: Path) -> None:
    """Split plans should be serializable for later reconstruction steps."""
    data_yaml = _make_split_dataset(tmp_path / "dataset")
    out_path = tmp_path / "split_plan.json"

    DatasetSplitPlanner().analyze(data_yaml).to_json(out_path)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["duplicates"]
    assert data["leakage"]
    assert data["assignments"]
