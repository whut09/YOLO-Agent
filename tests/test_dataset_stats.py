"""Dataset profiler tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import yolo_agent.tools.dataset_stats as dataset_stats_module
from yolo_agent.agents.data_stage_runner import DataStageRunner
from yolo_agent.cli import main
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.run_context import RunContext
from yolo_agent.tools.dataset_stats import DatasetProfiler


def _make_fake_yolo_dataset(root: Path) -> Path:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for name in ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]:
        (image_dir / name).write_bytes(b"")

    (label_dir / "img1.txt").write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.05 0.05",
                "1 0.3 0.3 0.08 0.08",
            ]
        ),
        encoding="utf-8",
    )
    (label_dir / "img2.txt").write_text("", encoding="utf-8")
    (label_dir / "img4.txt").write_text("0 0.4 0.4 0.04 0.04\n", encoding="utf-8")
    (label_dir / "img5.txt").write_text("", encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "scene: infrared_small_target",
                "train: images/train",
                "names:",
                "  - target",
                "  - clutter",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_dataset_profiler_computes_yolo_stats(tmp_path: Path) -> None:
    """Profiler should compute counts, distributions, and issue signals."""
    data_yaml = _make_fake_yolo_dataset(tmp_path / "dataset")

    report = DatasetProfiler().profile(data_yaml)

    assert report.image_count == 5
    assert report.label_count == 3
    assert report.class_distribution == {"target": 2, "clutter": 1}
    assert report.missing_label_files == 1
    assert report.empty_label_images == 2
    assert report.boxes_per_image["max"] == 2
    assert report.object_size_ratio["small"] == 1.0
    assert report.dataset_health.score < 70
    assert "severe_small_object_bias" in report.dataset_health.problems
    assert "annotation_noise" in report.dataset_health.problems
    assert "enable_small_object_recipe" in report.dataset_health.recommendations
    assert "Enable the small-object recipe" in report.recommendations[0]
    assert any("hard negative mining" in item for item in report.recommendations)


def test_dataset_profiler_emits_bounded_monotonic_progress(tmp_path: Path) -> None:
    data_yaml = _make_fake_yolo_dataset(tmp_path / "dataset")
    updates = []

    DatasetProfiler().profile(data_yaml, progress_callback=updates.append)

    assert [update.phase for update in updates] == [
        "discovering",
        "discovering",
        "reading_labels",
        "reading_labels",
        "health_checks",
        "health_checks",
        "writing",
    ]
    for phase in {update.phase for update in updates}:
        currents = [update.current for update in updates if update.phase == phase]
        assert currents == sorted(currents)
    assert updates[-1].status == "completed"
    assert updates[-1].images_discovered == 5
    assert updates[-1].labels_read == 4
    assert updates[-1].pid > 0
    assert updates[-1].updated_at >= updates[-1].started_at


def test_profile_stage_persists_atomic_completed_heartbeat(tmp_path: Path) -> None:
    data_yaml = _make_fake_yolo_dataset(tmp_path / "dataset")
    context = RunContext(
        run_id="profile-run",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=data_yaml,
    )

    result = DataStageRunner(context).profile_data()

    assert result.status == "completed"
    progress_path = context.artifact_path("dataset_profile_progress.json")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["phase"] == "writing"
    assert progress["current"] == progress["total"] == 2
    events = EventLog(context.run_dir / "events.jsonl").read()
    progress_events = [event for event in events if event.event_type == "stage_progress"]
    assert 1 <= len(progress_events) <= 10
    assert progress_events[-1].status == "completed"


def test_interrupted_profile_does_not_publish_json_contract(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    data_yaml = _make_fake_yolo_dataset(tmp_path / "dataset")
    out_prefix = tmp_path / "artifacts" / "dataset_report"

    def fail_write(path: Path, content: str) -> None:
        del path, content
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(dataset_stats_module, "_atomic_write_text", fail_write)

    with pytest.raises(OSError, match="simulated interrupted write"):
        dataset_stats_module.profile_dataset(data_yaml, out_prefix)

    assert not out_prefix.with_suffix(".json").exists()


def test_profile_dataset_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """The profile-data CLI should write JSON and Markdown reports."""
    data_yaml = _make_fake_yolo_dataset(tmp_path / "dataset")
    out_prefix = tmp_path / "runs" / "dataset_report"

    exit_code = main(["profile-data", "--data", str(data_yaml), "--out", str(out_prefix)])

    assert exit_code == 0
    json_path = out_prefix.with_suffix(".json")
    markdown_path = out_prefix.with_suffix(".md")
    assert json_path.exists()
    assert markdown_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["image_count"] == 5
    assert data["object_size_ratio"]["small"] == 1.0
    assert "dataset_health" in data
    assert "score" in data["dataset_health"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Dataset Report" in markdown
    assert "Dataset Health" in markdown


def test_dataset_health_detects_duplicate_and_train_val_leak(tmp_path: Path) -> None:
    """Health score should flag duplicate frames and train/val leakage heuristics."""
    root = tmp_path / "dataset"
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "same.jpg").write_bytes(b"duplicate")
        (root / "labels" / split / "same.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
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

    report = DatasetProfiler().profile(data_yaml)

    assert "high_duplicate_frames" in report.dataset_health.problems
    assert "train_val_leakage" in report.dataset_health.problems
    assert "deduplicate_near_duplicate_frames" in report.dataset_health.recommendations
    assert "fix_train_val_split_leakage" in report.dataset_health.recommendations
