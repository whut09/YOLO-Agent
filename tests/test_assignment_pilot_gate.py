from __future__ import annotations

from pathlib import Path

import pytest

from tests.assignment_fixtures import assignment_recipes, run_one_shadow_batch
from yolo_agent.certification.assignment_pilot_gate import (
    AssignmentActivePilotMaterializer,
)


@pytest.mark.parametrize(
    ("recipe_id", "method"),
    [
        ("yolo26_tood_tal_assignment_shadow", "tood_tal"),
        ("yolo26_ota_assignment_shadow", "ota"),
        ("yolo26_dsla_assignment_shadow", "dsla"),
    ],
)
def test_active_assignment_recipe_requires_matching_shadow_and_control(
    recipe_id: str,
    method: str,
    tmp_path: Path,
) -> None:
    shadow_dir = tmp_path / method
    shadow_dir.mkdir()
    run_one_shadow_batch(shadow_dir, method)
    evidence = shadow_dir / f"assignment_{method}_shadow_evidence.json"
    recipe = next(item for item in assignment_recipes() if item.recipe_id == recipe_id)
    protocol_hash = f"protocol-{method}"

    decision = AssignmentActivePilotMaterializer().materialize(
        shadow_recipe=recipe,
        shadow_evidence_path=evidence,
        candidate_protocol_hash=protocol_hash,
        control_protocol_hash=protocol_hash,
        matched_control_available=True,
    )

    assert decision.allowed is True
    assert decision.active_recipe is not None
    assert decision.active_recipe.train_overrides[recipe.primary_changed_variable] == "active"
    assert decision.active_recipe.train_overrides["assignment.shadow_evidence_path"] == str(
        evidence.resolve()
    )
    assert "matched_control" in decision.active_recipe.promotion_requirements
    assert "ASHA_only" in decision.active_recipe.promotion_requirements
    assert decision.shadow_evidence_sha256


def test_active_assignment_recipe_rejects_missing_or_unmatched_control(
    tmp_path: Path,
) -> None:
    recipe = assignment_recipes()[0]
    missing = AssignmentActivePilotMaterializer().materialize(
        shadow_recipe=recipe,
        shadow_evidence_path=tmp_path / "missing.json",
        candidate_protocol_hash="candidate",
        control_protocol_hash="control",
        matched_control_available=False,
    )

    assert missing.allowed is False
    assert "matched_control_missing" in missing.blocked_by
    assert "matched_control_protocol_mismatch" in missing.blocked_by
    assert "shadow_evidence_missing" in missing.blocked_by


def test_active_assignment_recipe_rejects_shadow_protocol_mismatch(
    tmp_path: Path,
) -> None:
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    run_one_shadow_batch(shadow_dir, "tood_tal")
    recipe = assignment_recipes()[0]

    decision = AssignmentActivePilotMaterializer().materialize(
        shadow_recipe=recipe,
        shadow_evidence_path=shadow_dir / "assignment_tood_tal_shadow_evidence.json",
        candidate_protocol_hash="different-protocol",
        control_protocol_hash="different-protocol",
        matched_control_available=True,
    )

    assert decision.allowed is False
    assert "shadow_evidence_protocol_mismatch" in decision.blocked_by
