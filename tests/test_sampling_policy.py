"""Sampling policy tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.agents.sampling_policy import SamplingPolicyEngine
from yolo_agent.core.dataset_split import DatasetSplitPlanner
from yolo_agent.tools.dataset_stats import DatasetHealth, DatasetReport


def _report(problems: list[str] | None = None, small_ratio: float = 0.0) -> DatasetReport:
    return DatasetReport(
        data_yaml=Path("data.yaml"),
        dataset_root=Path("."),
        scene="generic",
        image_count=20,
        label_count=20,
        class_distribution={"major": 19, "minor": 1},
        object_size_ratio={"small": small_ratio, "medium": 1.0 - small_ratio, "large": 0.0},
        empty_label_images=0,
        dataset_health=DatasetHealth(score=50, problems=problems or []),
    )


def _make_leaky_dataset(root: Path) -> Path:
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "same.jpg").write_bytes(b"duplicate")
        (root / "labels" / split / "same.txt").write_text("0 0.5 0.5 0.04 0.04\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "names:",
                "  - object",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_sampling_policy_recommends_rebalancing_negatives_and_small_object_sampling() -> None:
    """Dataset and error signals should map to sampling actions."""
    engine = SamplingPolicyEngine()

    plan = engine.recommend(
        _report(["class_imbalance_long_tail", "missing_hard_backgrounds"], small_ratio=0.7),
        [DetectionErrorObservation(error_type="background_confusion", count=4, severity="high")],
    )

    action_types = {action.action_type for action in plan.actions}
    assert "long_tail_class_rebalancing" in action_types
    assert "hard_negative_sampling" in action_types
    assert "background_only_injection" in action_types
    assert "small_object_oversampling" in action_types
    assert "scene_balanced_split" in action_types
    assert "ensure_empty_label_files_for_background_images" in plan.required_checks


def test_sampling_policy_uses_split_plan_for_duplicate_and_leakage_actions(tmp_path: Path) -> None:
    """Split plan evidence should trigger duplicate filtering and leakage fixes."""
    split_plan = DatasetSplitPlanner().analyze(_make_leaky_dataset(tmp_path / "dataset"))

    plan = SamplingPolicyEngine().recommend(
        _report(["high_duplicate_frames", "train_val_leakage"]),
        split_plan=split_plan,
    )

    by_type = {action.action_type: action for action in plan.actions}
    assert by_type["train_val_leakage_fix"].priority == 10.0
    assert by_type["train_val_leakage_fix"].parameters["leakage_pairs"] == 1
    assert by_type["duplicate_frame_filtering"].parameters["duplicate_groups"] == 1
    assert "block_experiment_comparison_until_leakage_fixed" in plan.required_checks
