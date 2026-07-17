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
    )

    assert report.status == "passed", report.failures
    assert report.asha_survivor
    assert report.paired_result_hashes
    assert (tmp_path / "mini-gpu-certification" / "certification_report.yaml").is_file()
