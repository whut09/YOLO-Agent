from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.component_runtime_backend import (
    build_component_runtime_launch,
)
from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry


class _MissingPayloadAdapter(ComponentAdapter):
    adapter_version = "1"
    source_commit = "test"
    strategy = "callback"

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        return AdapterValidationReport(ok=True)

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        return AdapterValidationReport(ok=True)

    def patch_model_config(self, config, context, *, dry_run=True):
        return config

    def patch_training_config(self, config, context, *, dry_run=True):
        return config

    def build_module(self, context):
        return None

    def load_pretrained_weights(self, module, weights, context):
        return WeightLoadResult(loaded=False)

    def smoke_test(self, context):
        return SmokeTestResult(passed=False)

    def expected_artifacts(self, context):
        return [ExpectedArtifact(name="unused", relative_path=Path("unused.json"))]

    def rollback_plan(self, context):
        return RollbackPlan()


def _base_command(tmp_path: Path) -> list[str]:
    return [
        "yolo",
        "detect",
        "train",
        "model=yolo26n.pt",
        f"data={tmp_path / 'data.yaml'}",
        "imgsz=640",
    ]


@pytest.mark.parametrize(
    ("component_id", "options", "artifact_name"),
    [
        (
            "sampling.small_object",
            {"dataset_manifest": "fixture", "seed": 1},
            "sampler_manifest",
        ),
        (
            "loss.quality.correlation",
            {"loss.correlation.weight": 0.2},
            "auxiliary_loss_correlation_evidence",
        ),
        (
            "distillation.yolo26_teacher_student",
            {"teacher": "yolo26s.pt", "student": "yolo26n.pt"},
            "distillation_evidence",
        ),
        (
            "head.p2_small_object",
            {"num_classes": 1, "audit_imgsz": 64},
            "p2_head_manifest",
        ),
    ],
)
def test_build_component_runtime_launch_uses_real_adapter_payload(
    tmp_path: Path,
    component_id: str,
    options: dict[str, object],
    artifact_name: str,
) -> None:
    launch = build_component_runtime_launch(
        component_id=component_id,
        base_command=_base_command(tmp_path),
        workspace=tmp_path / component_id,
        protocol_hash="protocol-1",
        options=options,
    )

    assert launch.payload.component_ids == [component_id]
    assert launch.command[:3] == [launch.command[0], "-m", launch.payload.runtime_entrypoint]
    assert launch.command[-len(launch.payload.base_command) :] == launch.payload.base_command
    assert launch.runtime_artifacts[artifact_name].parent == launch.payload_path.parent


def test_component_runtime_launch_rejects_adapter_without_payload(tmp_path: Path) -> None:
    registry = ComponentAdapterRegistry()
    registry.register("sampling.small_object", _MissingPayloadAdapter)

    with pytest.raises(RuntimeError, match="has no runtime payload"):
        build_component_runtime_launch(
            component_id="sampling.small_object",
            base_command=_base_command(tmp_path),
            workspace=tmp_path / "runtime",
            protocol_hash="protocol-1",
            options={},
            registry=registry,
        )
