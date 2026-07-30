"""Explicit hardware acceptance for the mini COCO evidence loop."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from yolo_agent.certification.runner import RealGpuAcceptanceSuite


@pytest.mark.real_gpu
def test_real_gpu_mini_coco_acceptance(tmp_path: Path) -> None:
    report = RealGpuAcceptanceSuite().run(
        workdir=tmp_path / "mini-gpu-certification",
        model=os.getenv("YOLO_AGENT_CERT_MODEL", "yolo26n.pt"),
        device=os.getenv("YOLO_AGENT_CERT_DEVICE", "0"),
        execute_real_gpu=True,
        recipe_id=os.getenv("YOLO_AGENT_CERT_RECIPE", "small_object_sampling"),
    )

    assert report.status == "passed", report.failures
    assert report.asha_survivor == os.getenv(
        "YOLO_AGENT_CERT_RECIPE", "small_object_sampling"
    )
    assert report.paired_result_hashes
    assert report.objective is not None and report.objective.passed
    assert report.objective.primary_metric == "ap_small"
    assert report.objective.target_metric_deltas["ap_small"] > 0
    assert report.objective.target_error_fact_deltas["false_negative/object"] > 0
    stages = {stage.stage_id: stage for stage in report.stages}
    assert stages["component_runtime_certification"].status == "passed"
    assert stages["runtime_adapter"].metrics["train_dataloader_hook_called"] is True
    assert stages["post_eval"].status == "passed"
    assert stages["paired_delta"].status == "passed"
    assert stages["asha_decision"].status == "passed"
    assert stages["pilot_10"].status == "passed"
    assert (tmp_path / "mini-gpu-certification" / "certification_report.yaml").is_file()
