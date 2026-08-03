"""Contract tests for paper auto-optimization acceptance reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperAutoOptimizationReport,
    PaperAutoOptimizationStage,
    PaperPairedDelta,
)


REQUIRED_STAGES = (
    "fresh_snapshot",
    "diagnosis",
    "method_profile",
    "certified_adapter",
    "matched_pilot_3_cohort",
    "candidate_control_post_eval",
    "complete_coco_error_facts",
    "paired_bootstrap_delta",
    "asha",
    "pilot_10",
    "policy_memory",
    "pilot_reproduced",
)


def _passed_report(tmp_path: Path) -> PaperAutoOptimizationReport:
    tracks = (
        ("sampling", "sampling.small_object"),
        ("auxiliary_loss", "loss.quality.correlation"),
        ("distillation", "distillation.yolo26_teacher_student"),
        ("model_graph", "head.p2_small_object"),
    )
    return PaperAutoOptimizationReport(
        acceptance_id="paper-auto",
        status="passed",
        execute_real_gpu=True,
        model="yolo26n.pt",
        device="0",
        research_snapshot_hash="snapshot",
        paper_ids=["paper-small-object"],
        adapter_hash="adapter",
        maturity="gpu_certified",
        runtime_payload_hash="payload",
        objective_hash="objective",
        protocol_hash="protocol",
        stages=[
            PaperAutoOptimizationStage(stage_id=stage, status="passed")
            for stage in REQUIRED_STAGES
        ],
        component_ids=[component_id for _, component_id in tracks],
        component_families=[family for family, _ in tracks],
        paired_deltas=[
            PaperPairedDelta(
                stage_id="pilot_3",
                track_id=family,  # type: ignore[arg-type]
                recipe_id=component_id,
                component_id=component_id,
                component_family=family,
                verified=True,
                protocol_match=True,
                ap_small_delta=0.01,
                target_recall_delta=0.02,
                false_negative_delta=1.0,
                result_hash=f"result-pilot-3-{family}",
            )
            for family, component_id in tracks
        ]
        + [
            PaperPairedDelta(
                stage_id="pilot_10",
                track_id="sampling",
                recipe_id="sampling.small_object",
                component_id="sampling.small_object",
                component_family="sampling",
                verified=True,
                protocol_match=True,
                ap_small_delta=0.02,
                target_recall_delta=0.02,
                false_negative_delta=1.0,
                result_hash="result-pilot-10-sampling",
            )
        ],
        asha_survivor="sampling.small_object",
        asha_survivors=["sampling.small_object"],
        policy_memory_path=tmp_path / "policy_memory.jsonl",
        pilot_reproduced=True,
        pilot_reproduced_component_ids=["sampling.small_object"],
    )


def test_passed_report_requires_complete_paired_state_machine(tmp_path: Path) -> None:
    report = _passed_report(tmp_path)
    path = tmp_path / "report.yaml"
    report.to_yaml(path, exclude_none=True, sort_keys=False)

    loaded = PaperAutoOptimizationReport.from_yaml(path)

    assert loaded.report_hash == report.report_hash
    assert loaded.pilot_reproduced is True


def test_passed_report_rejects_missing_stage(tmp_path: Path) -> None:
    payload = _passed_report(tmp_path).model_dump(mode="python", exclude={"report_hash"})
    payload["stages"] = payload["stages"][:-1]

    with pytest.raises(ValueError, match="missing stages"):
        PaperAutoOptimizationReport.model_validate(payload)


def test_recovery_report_requires_recovery_action() -> None:
    with pytest.raises(ValueError, match="requires evidence recovery actions"):
        PaperAutoOptimizationReport(
            acceptance_id="recovery",
            status="recovery",
            execute_real_gpu=True,
            model="yolo26n.pt",
            device="0",
        )


def test_sampling_only_report_cannot_claim_multi_mechanism_acceptance(
    tmp_path: Path,
) -> None:
    payload = _passed_report(tmp_path).model_dump(mode="python", exclude={"report_hash"})
    payload["paired_deltas"] = [
        item
        for item in payload["paired_deltas"]
        if item["component_family"] == "sampling"
    ]

    with pytest.raises(ValueError, match="four distinct pilot_3 component families"):
        PaperAutoOptimizationReport.model_validate(payload)
