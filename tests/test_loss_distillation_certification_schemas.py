from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.loss_distillation_schemas import (
    DistillationCpuReport,
    QualityLossCpuReport,
)


def _checks(*, distillation: bool) -> dict[str, bool]:
    common = {
        "atomic_recipe_verified": True,
        "trainer_bridge_called": True,
        "total_loss_changed": True,
        "student_backward": True,
        "zero_weight_native_equivalent": True,
        "exact_reproduction_false": True,
    }
    if distillation:
        common.update(
            {
                "teacher_no_grad": True,
                "teacher_frozen_eval": True,
                "student_inference_graph_unchanged": True,
                "method_profiles_only": True,
            }
        )
    else:
        common["paper_prior_not_local_evidence"] = True
    return common


def test_quality_report_requires_zero_weight_and_runtime_artifacts(
    tmp_path: Path,
) -> None:
    report = QualityLossCpuReport(
        component_id="loss.quality.correlation",
        recipe_id="yolo26_correlation_auxiliary_loss",
        status="passed",
        protocol_hash="protocol",
        runtime_payload_hash="a" * 64,
        zero_control_payload_hash="b" * 64,
        runtime_payload_path=tmp_path / "runtime.yaml",
        zero_control_payload_path=tmp_path / "zero.yaml",
        runtime_evidence_path=tmp_path / "evidence.json",
        zero_control_evidence_path=tmp_path / "zero-evidence.json",
        checks=_checks(distillation=False),
    )

    assert len(report.report_hash) == 64


def test_distillation_report_rejects_exact_reproduction_claim(tmp_path: Path) -> None:
    checks = _checks(distillation=True)
    checks["exact_reproduction_false"] = False

    with pytest.raises(ValueError, match="exact_reproduction_false"):
        DistillationCpuReport(
            status="passed",
            protocol_hash="protocol",
            runtime_payload_hash="a" * 64,
            zero_control_payload_hash="b" * 64,
            runtime_payload_path=tmp_path / "runtime.yaml",
            zero_control_payload_path=tmp_path / "zero.yaml",
            runtime_evidence_path=tmp_path / "evidence.json",
            zero_control_evidence_path=tmp_path / "zero-evidence.json",
            checks=checks,
        )
