"""Prompt-18I: evidence-linked root-cause attribution tests.

Pins the §2 contract (hypotheses cite real measured values, never derived
from a mAP movement alone) and the §6 action-family mapping, on the
deterministic synthetic COCO fixtures — no GPU, no training.
"""

from __future__ import annotations

import json
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
from yolo_agent.core.error_root_cause import (  # noqa: E402
    EvidenceLinkedHypothesis,
    derive_root_cause_hypotheses,
    fact_id_for_subject,
)


def _write(tmp_path: Path, name: str, payload) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture()
def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    return _write(tmp_path, "gt.json", fixtures.GT), _write(tmp_path, "preds.json", fixtures.PREDICTIONS)


def _profile(tmp_path: Path, preds, candidate_id: str = "cand"):
    gt = _write(tmp_path, "gt.json", fixtures.GT)
    pp = _write(tmp_path, "preds.json", preds)
    return build_detection_error_profile(
        ErrorProfileSource(
            gt_json=gt,
            predictions_json=pp,
            run_id="run-1",
            candidate_id=candidate_id,
            split="val",
        )
    )


def test_hypotheses_cite_real_measured_values(artifacts) -> None:
    """Every evidence link must quote a number that exists in the profile."""
    gt, preds = artifacts
    profile = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=gt, predictions_json=preds, run_id="run-1", candidate_id="cand", split="val"
        )
    )
    hypotheses = derive_root_cause_hypotheses(profile)
    assert hypotheses, "deterministic fixture must yield at least one hypothesis"
    for hypothesis in hypotheses:
        assert hypothesis.evidence, "hypothesis without evidence links is forbidden"
        for link in hypothesis.evidence:
            assert link.fact_ids, "evidence must reference ErrorFact ids"
            assert link.fact_ids[0].startswith(profile.profile_id)
            assert isinstance(link.value, float)


def test_empty_evidence_links_are_rejected() -> None:
    """Fail-closed: 'mAP dropped so probably small objects' is inexpressible."""
    with pytest.raises(Exception):
        EvidenceLinkedHypothesis(
            hypothesis_id="guess",
            pattern="small_object_feature_loss",
            description="mAP dropped so probably small objects",
            evidence=[],
            candidate_action_families=["sampling"],
        )


def test_background_fp_pattern_and_families(artifacts) -> None:
    """§6: background-FP pattern maps to the negative-evidence family set."""
    gt, preds = artifacts
    profile = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=gt, predictions_json=preds, run_id="run-1", candidate_id="cand", split="val"
        )
    )
    matches = [h for h in derive_root_cause_hypotheses(profile) if h.pattern == "background_false_positive"]
    assert len(matches) == 1
    hypothesis = matches[0]
    bg_share = hypothesis.evidence[0].value
    profile_bg = profile.false_positive.background_fp / profile.false_positive.total
    assert bg_share == pytest.approx(profile_bg, abs=1e-4)
    assert hypothesis.candidate_action_families == [
        "sampling",
        "augmentation",
        "classification_loss",
        "threshold",
        "postprocess",
    ]


def test_class_confusion_names_the_real_pair(artifacts) -> None:
    gt, preds = artifacts
    profile = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=gt, predictions_json=preds, run_id="run-1", candidate_id="cand", split="val"
        )
    )
    matches = [h for h in derive_root_cause_hypotheses(profile) if h.pattern == "class_confusion"]
    assert len(matches) == 1
    pair, count = profile.classification.top_confusion_pairs[0]
    true_cls, pred_cls = pair.split("->", 1)
    assert matches[0].hypothesis_id == f"rc:cand:class_confusion:{true_cls}->{pred_cls}"
    assert matches[0].evidence[0].value == float(count)


def test_small_object_pattern_requires_both_signals(tmp_path: Path) -> None:
    """§2: a single mAP drop must NOT yield a small-object hypothesis.

    The rule needs the structural pair (small-AP gap AND small-FN share);
    here a moderately low small AP with no FN dominance stays silent.
    """
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [100, 100, 100, 80], "score": 0.85},
        {"image_id": 2, "category_id": 1, "bbox": [52, 52, 40, 40], "score": 0.8},
        {"image_id": 2, "category_id": 2, "bbox": [202, 202, 90, 70], "score": 0.7},
    ]
    profile = _profile(tmp_path, preds)
    patterns = [h.pattern for h in derive_root_cause_hypotheses(profile)]
    assert "small_object_feature_loss" not in patterns


def test_delta_sharpens_confidence(tmp_path: Path, artifacts) -> None:
    gt, preds = artifacts
    parent = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=gt, predictions_json=preds, run_id="run-1", candidate_id="parent", split="val"
        )
    )
    candidate = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=gt, predictions_json=preds, run_id="run-1", candidate_id="cand", split="val"
        )
    )
    delta = build_detection_error_delta(candidate=candidate, parent=parent)
    hypotheses = derive_root_cause_hypotheses(candidate, delta=delta)
    bg = [h for h in hypotheses if h.pattern == "background_false_positive"][0]
    metrics = [link.metric for link in bg.evidence]
    assert "delta_background_fp" in metrics


def test_shared_fact_id_convention_is_deterministic(artifacts) -> None:
    gt, _ = artifacts
    profile = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=gt,
            predictions_json=Path(gt).with_name("preds.json"),
            run_id="run-1",
            candidate_id="cand",
            split="val",
        )
    )
    assert (
        fact_id_for_subject(profile, "global:map50")
        == f"{profile.profile_id}:global:map50"
    )
