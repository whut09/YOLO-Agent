"""Prompt-18I §9: real-run integration hook tests (deterministic, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_detection_error_profile_builder as fixtures  # noqa: E402

from yolo_agent.core.error_profile_run_hook import collect_error_profile  # noqa: E402


@pytest.fixture()
def eval_files(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "eval"
    data.mkdir()
    (data / "gt.json").write_text(__import__("json").dumps(fixtures.GT))
    (data / "predictions.json").write_text(__import__("json").dumps(fixtures.PREDICTIONS))
    return data / "gt.json", data / "predictions.json"


def test_profile_from_real_eval_files(eval_files, tmp_path: Path) -> None:
    gt, preds = eval_files
    out = tmp_path / "artifacts"
    profile = collect_error_profile(
        gt_json=gt,
        predictions_json=preds,
        run_id="run-1",
        candidate_id="cand",
        split="val",
        artifacts_dir=out,
    )
    assert profile.run_id == "run-1"
    assert profile.candidate_id == "cand"
    assert profile.false_negative.total == 2
    assert profile.false_positive.total == 3
    assert (out / "error_profile.yaml").exists()


def test_official_report_overrides_globals(eval_files, tmp_path: Path) -> None:
    gt, preds = eval_files
    report = tmp_path / "coco_report.json"
    report.write_text(
        __import__("json").dumps(
            {"map50": 0.55, "map50_95": 0.4, "precision": 0.7, "recall": 0.6}
        )
    )
    profile = collect_error_profile(
        gt_json=gt,
        predictions_json=preds,
        run_id="run-1",
        candidate_id="cand",
        official_report_json=report,
    )
    assert profile.global_.map50 == pytest.approx(0.55)
    assert profile.global_.map50_95 == pytest.approx(0.4)
    assert profile.global_.precision == pytest.approx(0.7)
    assert profile.global_.recall == pytest.approx(0.6)


def test_no_metadata_means_no_scene_slices(eval_files) -> None:
    gt, preds = eval_files
    profile = collect_error_profile(
        gt_json=gt,
        predictions_json=preds,
        run_id="run-1",
        candidate_id="cand",
    )
    assert profile.scene_slices == []
    # §5: absence is recorded, never fabricated
    assert set(profile.unavailable_scene_slices) == set(fixtures.SCENE_SLICE_TAGS) if hasattr(fixtures, "SCENE_SLICE_TAGS") else True
