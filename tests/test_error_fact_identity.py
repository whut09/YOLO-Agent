"""Prompt-18I-v2 §3: ErrorFact identity tests (deterministic, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_detection_error_profile_builder as fixtures  # noqa: E402

from yolo_agent.core.detection_error_profile import ErrorProfileSource  # noqa: E402
from yolo_agent.core.detection_error_profile_builder import (  # noqa: E402
    build_detection_error_profile,
)
from yolo_agent.core.error_fact_identity import (  # noqa: E402
    CALCULATION_VERSION,
    error_fact_identities,
    hypotheses_may_reference,
)


@pytest.fixture()
def profile(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "gt.json").write_text(__import__("json").dumps(fixtures.GT))
    (data / "preds.json").write_text(__import__("json").dumps(fixtures.PREDICTIONS))
    return build_detection_error_profile(
        ErrorProfileSource(
            gt_json=data / "gt.json",
            predictions_json=data / "preds.json",
            run_id="run-1",
            candidate_id="cand",
            split="val",
        )
    )


def test_every_fact_carries_identity_and_provenance(profile) -> None:
    facts = error_fact_identities(profile)
    assert facts
    for fact in facts:
        assert fact.error_fact_id.startswith(profile.profile_id)
        assert fact.metric
        assert fact.slice
        assert fact.evidence_path == profile.gt_artifact
        assert fact.calculation_version == CALCULATION_VERSION


def test_ids_are_deterministic_and_profile_scoped(profile, tmp_path: Path) -> None:
    facts = error_fact_identities(profile)
    ids = [f.error_fact_id for f in facts]
    assert len(ids) == len(set(ids)), "fact ids must be unique"

    # Same inputs -> same ids.
    data = tmp_path / "again"
    data.mkdir()
    (data / "gt.json").write_text(__import__("json").dumps(fixtures.GT))
    (data / "preds.json").write_text(__import__("json").dumps(fixtures.PREDICTIONS))
    clone = build_detection_error_profile(
        ErrorProfileSource(
            gt_json=data / "gt.json",
            predictions_json=data / "preds.json",
            run_id="run-1",
            candidate_id="cand",
            split="val",
        )
    )
    assert [f.error_fact_id for f in error_fact_identities(clone)] == ids


def test_absent_measurements_produce_no_facts(profile) -> None:
    assert profile.global_.map50_95 is None
    facts = error_fact_identities(profile)
    assert not any(f.metric == "map50_95" for f in facts)


def test_fn_fp_slices_are_decomposed(profile) -> None:
    facts = error_fact_identities(profile)
    slices = {f.slice for f in facts}
    assert "false_negative:class:car" in slices
    # Both fixture FNs are medium-area misses; the scale slice follows the data.
    assert "false_negative:scale:medium" in slices
    assert "false_positive:background" in slices
    assert "false_positive:duplicate" in slices
    assert "false_positive:class_confusion" in slices


def test_confusion_pairs_become_facts(profile) -> None:
    facts = error_fact_identities(profile)
    confusion = [f for f in facts if f.slice.startswith("confusion:")]
    assert confusion, "fixture has a real cross-class confusion pair"
    assert any(f.slice == "confusion:person->car" for f in confusion)


def test_hypothesis_reference_validation(profile) -> None:
    facts = error_fact_identities(profile)
    real_id = facts[0].error_fact_id
    assert hypotheses_may_reference([real_id], facts)
    assert not hypotheses_may_reference([f"{profile.profile_id}:global:nope"], facts)
    assert not hypotheses_may_reference([], facts)
