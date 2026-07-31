"""Opt-in CUDA acceptance tests; excluded from default pytest execution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from yolo_agent.certification.component_runner import ComponentCertificationRunner


pytestmark = [
    pytest.mark.real_gpu,
    pytest.mark.skipif(
        os.environ.get("YOLO_AGENT_RUN_REAL_GPU_COMPONENTS") != "1",
        reason="set YOLO_AGENT_RUN_REAL_GPU_COMPONENTS=1 for real CUDA training",
    ),
]


def test_sampling_real_gpu_train_checkpoint_and_resume(tmp_path: Path) -> None:
    model = Path("yolo26n.pt").resolve()
    if not model.is_file():
        pytest.skip("local yolo26n.pt is required; certification never downloads it")
    runner = ComponentCertificationRunner()
    registry = tmp_path / "registry.yaml"
    workdir = tmp_path / "sampling.small_object"

    cpu = runner.run(
        component_id="sampling.small_object",
        mode="cpu",
        workdir=workdir,
        registry_path=registry,
        model=str(model),
    )
    gpu = runner.run(
        component_id="sampling.small_object",
        mode="gpu",
        workdir=workdir,
        registry_path=registry,
        model=str(model),
        device="0",
        execute_gpu=True,
    )

    assert cpu.status == "passed"
    assert gpu.status == "passed", gpu.errors
    assert gpu.final_maturity == "gpu_certified"
    assert gpu.next_maturity == "pilot_reproduced"
