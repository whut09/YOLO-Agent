"""Prompt-18I §7: round-artifact writer tests (deterministic, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_detection_error_profile_builder as fixtures  # noqa: E402

from yolo_agent.core.detection_error_delta import build_detection_error_delta  # noqa: E402
from yolo_agent.core.detection_error_profile import ErrorProfileSource  # noqa: E402
from yolo_agent.core.detection_error_profile_builder import (  # noqa: E402
    build_detection_error_profile,
)
from yolo_agent.core.error_decision_trace import (  # noqa: E402
    build_error_decision_trace,
)
from yolo_agent.core.error_profile_evidence_gate import (  # noqa: E402
    evaluate_error_profile_evidence,
)
from yolo_agent.core.error_root_cause import derive_root_cause_hypotheses  # noqa: E402
from yolo_agent.core.error_round_artifacts import (  # noqa: E402
    load_decision_trace_artifact,
    load_error_delta_artifact,
    load_error_profile_artifact,
    write_round_artifacts,
)


@pytest.fixture()
def round_artifacts(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "gt.json").write_text(__import__("json").dumps(fixtures.GT))
    (data_dir / "preds.json").write_text(__import__("json").dumps(fixtures.PREDICTIONS))
    candidate = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=data_dir / "gt.json",
            predictions_json=data_dir / "preds.json",
            run_id="run-1",
            candidate_id="cand",
            split="val",
        )
    )
    parent = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=data_dir / "gt.json",
            predictions_json=data_dir / "preds.json",
            run_id="run-1",
            candidate_id="parent",
            split="val",
        )
    )
    delta = build_detection_error_delta(candidate=candidate, parent=parent)
    gate = evaluate_error_profile_evidence(
        baseline_profile=parent, candidate_profile=candidate, delta=delta
    )
    trace = build_error_decision_trace(
        profile=candidate,
        gate=gate,
        delta=delta,
        hypotheses=derive_root_cause_hypotheses(candidate, delta=delta),
    )
    out_dir = tmp_path / "artifacts"
    written = write_round_artifacts(out_dir, profile=candidate, delta=delta, trace=trace)
    return out_dir, written, candidate, delta, trace


def test_all_three_artifacts_written(round_artifacts) -> None:
    out_dir, written, _, _, _ = round_artifacts
    assert set(written) == {"error_profile", "error_delta", "decision_trace"}
    for path in written.values():
        assert path.exists() and path.parent == out_dir


def test_roundtrip_preserves_models(round_artifacts) -> None:
    _, _, candidate, delta, trace = round_artifacts
    out_dir = round_artifacts[0]
    assert load_error_profile_artifact(out_dir / "error_profile.yaml") == candidate
    assert load_error_delta_artifact(out_dir / "error_delta.yaml") == delta
    assert load_decision_trace_artifact(out_dir / "decision_trace.yaml") == trace


def test_serialization_is_deterministic(round_artifacts, tmp_path: Path) -> None:
    out_dir, written, candidate, delta, trace = round_artifacts
    again = tmp_path / "again"
    write_round_artifacts(again, profile=candidate, delta=delta, trace=trace)
    for name, path in written.items():
        assert path.read_bytes() == (again / path.name).read_bytes(), name
