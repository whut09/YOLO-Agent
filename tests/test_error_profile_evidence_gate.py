"""Prompt-18I: evidence completeness gate tests."""

from __future__ import annotations

import json
from pathlib import Path


from yolo_agent.core.detection_error_delta import build_detection_error_delta
from yolo_agent.core.detection_error_profile import ErrorProfileSource
from yolo_agent.core.detection_error_profile_builder import (
    build_detection_error_profile,
)
from yolo_agent.core.error_profile_evidence_gate import (
    EVIDENCE_ONLY_ACTIONS,
    evaluate_error_profile_evidence,
)

GT = {
    "categories": [{"id": 1, "name": "person"}],
    "images": [{"id": 1}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [60, 60, 40, 40], "area": 1600},
    ],
}


def _profile(tmp_path: Path, name: str, predictions: list[dict], **kwargs: object):
    gt_path = tmp_path / "gt.json"  # shared GT: matched evaluation semantics
    if not gt_path.exists():
        gt_path.write_text(json.dumps(GT))
    (tmp_path / f"preds_{name}.json").write_text(json.dumps(predictions))
    return build_detection_error_profile(
        ErrorProfileSource(
            gt_json=gt_path,
            predictions_json=tmp_path / f"preds_{name}.json",
            run_id="run-1",
            candidate_id=name,
            **kwargs,  # type: ignore[arg-type]
        )
    )


FULL_PREDS = [
    {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
    {"image_id": 1, "category_id": 1, "bbox": [62, 62, 40, 40], "score": 0.7},
]


def test_complete_profiles_unlock_the_budget(tmp_path: Path) -> None:
    baseline = _profile(tmp_path, "base", FULL_PREDS)
    candidate = _profile(tmp_path, "cand", FULL_PREDS)
    delta = build_detection_error_delta(candidate, baseline)
    result = evaluate_error_profile_evidence(
        baseline_profile=baseline, candidate_profile=candidate, delta=delta
    )
    assert result.verdict == "complete"
    assert result.budget_request_allowed is True
    assert "next_round_experiment" in result.allowed_actions


def test_partial_evidence_restricts_to_evidence_actions(tmp_path: Path) -> None:
    baseline = _profile(tmp_path, "base", FULL_PREDS)
    candidate = _profile(tmp_path, "cand", FULL_PREDS)
    # Delta deliberately omitted: the profiles exist but were never compared.
    result = evaluate_error_profile_evidence(
        baseline_profile=baseline, candidate_profile=candidate, delta=None
    )
    assert result.verdict in {"partial", "insufficient"}
    assert result.budget_request_allowed is False
    assert set(result.allowed_actions) <= set(EVIDENCE_ONLY_ACTIONS) | {"collect_evidence"}


def test_insufficient_evidence_allows_only_collection(tmp_path: Path) -> None:
    result = evaluate_error_profile_evidence(
        baseline_profile=None, candidate_profile=None, delta=None
    )
    assert result.verdict == "insufficient"
    assert result.allowed_actions == ["collect_evidence"]
    assert result.budget_request_allowed is False


def test_mismatched_delta_blocks_budget(tmp_path: Path) -> None:
    baseline = _profile(tmp_path, "base", FULL_PREDS)
    candidate = _profile(
        tmp_path, "cand", FULL_PREDS, dataset_manifest_hash="different-dataset"
    )
    delta = build_detection_error_delta(candidate, baseline)
    assert delta.matched_evaluation is False
    result = evaluate_error_profile_evidence(
        baseline_profile=baseline, candidate_profile=candidate, delta=delta
    )
    assert result.verdict == "insufficient"
    assert any("not_matched" in reason for reason in result.reasons)
    assert result.budget_request_allowed is False


def test_repair_evaluation_is_evidence_only() -> None:
    assert "repair_evaluation" in EVIDENCE_ONLY_ACTIONS


def test_profile_with_new_fields_is_complete(tmp_path) -> None:
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [61, 61, 40, 40], "score": 0.7},
    ]
    profile = _profile(tmp_path, "full", predictions)
    gate = evaluate_error_profile_evidence(
        baseline_profile=profile, candidate_profile=profile, delta=None
    )
    assert gate.missing_sections == [], gate.missing_sections


def test_stripped_profile_is_not_complete(tmp_path) -> None:
    """mAP/precision/recall alone cannot pass as COMPLETE."""
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
    ]
    profile = _profile(tmp_path, "stripped", predictions)
    stripped = profile.model_copy(
        update={
            "localization": profile.localization.model_copy(
                update={"matched_iou_distribution": {}, "mean_matched_iou": None}
            ),
            "confidence": type(profile.confidence)(),
        }
    )
    gate = evaluate_error_profile_evidence(
        baseline_profile=stripped, candidate_profile=stripped, delta=None
    )
    assert "baseline.localization" in gate.missing_sections
    assert "candidate.confidence" in gate.missing_sections
    assert gate.verdict in {"partial", "insufficient"}
    assert not gate.budget_request_allowed
