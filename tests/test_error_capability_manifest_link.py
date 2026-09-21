"""Prompt-18I-v2 §8: promotion record -> capability manifest link tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_agent.core.error_capability_manifest_link import (
    apply_promotion_to_manifest,
)
from yolo_agent.core.error_capability_promotion import ErrorLoopPromotionRecord


@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "capability_maturity.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "capabilities": [
                    {
                        "capability_id": "candidate_coco_error_facts",
                        "status": "incomplete",
                    },
                    {
                        "capability_id": "error_delta_next_round",
                        "status": "partial",
                    },
                    {
                        "capability_id": "asha_queue_control",
                        "status": "executable",
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _record(**overrides) -> ErrorLoopPromotionRecord:
    base = dict(
        run_id="run-1",
        baseline_facts_complete=True,
        candidate_facts_complete=True,
        error_delta_consumed=True,
        next_round_decision_consumed_delta=True,
        reasons=[],
    )
    base.update(overrides)
    record = ErrorLoopPromotionRecord(**base)
    if record.baseline_facts_complete and record.candidate_facts_complete and record.error_delta_consumed:
        record = record.model_copy(update={"facts_capability": "executable"})
    if record.next_round_decision_consumed_delta and record.error_delta_consumed:
        record = record.model_copy(update={"decision_capability": "executable"})
    return record


def test_incomplete_record_changes_nothing(manifest: Path) -> None:
    record = _record(
        baseline_facts_complete=False,
        reasons=["baseline profile facts are not complete"],
    )
    result = apply_promotion_to_manifest(record, manifest)
    assert result.upgraded == []
    assert result.rejected
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    statuses = {item["capability_id"]: item["status"] for item in data["capabilities"]}
    assert statuses["candidate_coco_error_facts"] == "incomplete"
    assert statuses["error_delta_next_round"] == "partial"


def test_complete_record_upgrades_only_the_two(manifest: Path) -> None:
    result = apply_promotion_to_manifest(_record(), manifest)
    assert set(result.upgraded) == {"candidate_coco_error_facts", "error_delta_next_round"}
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    statuses = {item["capability_id"]: item["status"] for item in data["capabilities"]}
    assert statuses["candidate_coco_error_facts"] == "executable"
    assert statuses["error_delta_next_round"] == "executable"
    assert statuses["asha_queue_control"] == "executable"
    # Ordering preserved.
    ids = [item["capability_id"] for item in data["capabilities"]]
    assert ids.index("candidate_coco_error_facts") < ids.index("asha_queue_control")


def test_already_promoted_is_idempotent(manifest: Path) -> None:
    apply_promotion_to_manifest(_record(), manifest)
    result = apply_promotion_to_manifest(_record(), manifest)
    assert result.upgraded == []
