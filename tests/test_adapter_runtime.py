"""Offline tests for the typed adapter runtime contract."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from yolo_agent.components.adapters import (
    AdapterContext,
    AdapterRuntimePayload,
    ExpectedArtifact,
    RollbackPlan,
    RuntimePluginReference,
)
from yolo_agent.components.adapters.audit_contract import (
    validate_audited_runtime_payload,
)
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationAdapter,
)
from yolo_agent.components.adapters.head.p2_head import P2HeadAdapter
from yolo_agent.components.adapters.inference.slicing import SlicingInferenceAdapter
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    SmallObjectSamplingAdapter,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.command_spec import CommandSpec


def _payload(tmp_path: Path) -> AdapterRuntimePayload:
    return AdapterRuntimePayload(
        component_ids=["dummy.component"],
        adapter_classes=["DummyAdapter"],
        adapter_versions={"dummy.component": "dummy.v1"},
        source_commits={"dummy.component": "local-test"},
        trainer_plugin=[
            RuntimePluginReference(
                reference="yolo_agent.components.adapters.dummy:DummyRuntimePlugin",
                required_hooks=["build_model"],
            )
        ],
        generated_config={"training_config": {"imgsz": 640, "amp": True}},
        changed_variables={"training.adapter_marker": "active"},
        expected_artifacts=[
            ExpectedArtifact(name="adapter_patch", relative_path=Path("adapter_patch.yaml"))
        ],
        rollback_plan=RollbackPlan(actions=["discard generated adapter patch"]),
        protocol_hash="protocol-1",
        base_command=[sys.executable, "-c", "print('runtime-ok')"],
        supports_amp=True,
        supports_ddp=True,
        supports_resume=True,
    )


def test_runtime_payload_serializes_and_restores_with_stable_hash(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    path = payload.write(tmp_path / "runtime.yaml")

    restored = AdapterRuntimePayload.read(path)

    assert restored == payload
    assert restored.payload_hash == payload.payload_hash
    assert restored.changed_variables == payload.changed_variables
    assert restored.supports_amp and restored.supports_ddp and restored.supports_resume


def test_runtime_payload_rejects_missing_plugin() -> None:
    payload = _payload(Path("."))
    broken = payload.model_copy(
        update={
            "trainer_plugin": [
                RuntimePluginReference(
                    reference="missing.adapter.module:Plugin",
                    required_hooks=["build_model"],
                )
            ]
        }
    )

    with pytest.raises(ImportError, match="runtime plugin is not importable"):
        broken.verify_imports()


def test_runtime_payload_rejects_importable_non_plugin() -> None:
    payload = _payload(Path("."))
    broken = payload.model_copy(
        update={
            "trainer_plugin": [
                RuntimePluginReference(
                    reference="pathlib:Path", required_hooks=["build_model"]
                )
            ]
        }
    )

    with pytest.raises(ImportError, match="no plugin_version"):
        broken.verify_imports()


def test_runtime_payload_rejects_missing_changed_variables() -> None:
    raw = _payload(Path(".")).model_dump(mode="python")
    raw["changed_variables"] = {}

    with pytest.raises(ValueError, match="at least one changed variable"):
        AdapterRuntimePayload.model_validate(raw)


def test_runtime_payload_composition_rejects_changed_variable_conflicts(
    tmp_path: Path,
) -> None:
    first = _payload(tmp_path).model_copy(
        update={"changed_variables": {"loss.weight": 0.1}}
    )
    second = first.model_copy(
        update={"changed_variables": {"loss.weight": 0.2}}
    )

    with pytest.raises(ValueError, match="changed variable conflict"):
        AdapterRuntimePayload.compose(
            [first, second],
            generated_config={},
            expected_artifacts=[],
            rollback_plan=RollbackPlan(),
        )


def test_command_spec_calls_runtime_entrypoint_and_preserves_training_args(tmp_path: Path) -> None:
    original = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project=tmp_path / "runs",
        name="candidate",
        epochs=3,
        imgsz=640,
        amp=True,
        resume="last.pt",
        device=[0, 1],
    )
    payload = _payload(tmp_path)
    path = payload.write(tmp_path / "runtime.yaml")

    wrapped = original.with_runtime_payload(
        path,
        runtime_entrypoint=payload.runtime_entrypoint,
        payload_hash=payload.payload_hash,
        protocol_hash=payload.protocol_hash,
    )

    assert wrapped.argv[:3] == [sys.executable, "-m", payload.runtime_entrypoint]
    assert "--" in wrapped.argv
    assert "amp=True" in wrapped.argv
    assert "resume=last.pt" in wrapped.argv
    assert "device=0,1" in wrapped.argv
    assert wrapped.metadata["adapter_runtime_protocol_hash"] == "protocol-1"
    assert wrapped.expected_artifacts["adapter_runtime_payload"] == path


def test_training_runtime_entrypoint_rejects_non_train_command(tmp_path: Path) -> None:
    from yolo_agent.adapters.ultralytics.runtime_entrypoint import run_payload

    payload = _payload(tmp_path)
    path = payload.write(tmp_path / "runtime.yaml")

    with pytest.raises(ValueError, match="refusing uninstrumented subprocess fallback"):
        run_payload(path, payload.base_command)


def test_wrapped_training_payload_cannot_execute_arbitrary_command(tmp_path: Path) -> None:
    from yolo_agent.adapters.ultralytics.runtime_entrypoint import run_payload

    marker = tmp_path / "entrypoint-ran.txt"
    original = CommandSpec(
        command=sys.executable,
        args=["-c", f"from pathlib import Path; Path(r'{marker}').write_text('ok')"],
        shell=False,
    )
    payload = _payload(tmp_path).model_copy(
        update={"base_command": list(original.argv)}
    )
    path = payload.write(tmp_path / "runtime.yaml")
    with pytest.raises(ValueError, match="requires a 'yolo detect train'"):
        run_payload(path, list(original.argv))
    assert not marker.exists()


def test_sahi_adapter_builds_inference_only_runtime_payload(tmp_path: Path) -> None:
    contract = ComponentContract(
        component_id="inference.sahi_slicing",
        display_name="SAHI",
        category="slicing",
        maturity="unit_tested",
    )
    context = AdapterContext(contract=contract, workspace=tmp_path)

    payload = SlicingInferenceAdapter().build_runtime_payload(
        context,
        protocol_hash="protocol-1",
        base_command=["yolo-agent", "advanced", "certify-sahi", "--execute"],
        generated_config={},
    )

    assert payload.inference_plugin
    assert not payload.trainer_plugin
    assert payload.changed_variables["inference.slicing_policy"]
    assert payload.inference_plugin[0].required_hooks == ["prepare_command"]
    checks = validate_audited_runtime_payload(payload, "inference.sahi_slicing")
    assert checks["audited_plugin_kind"] == "inference_plugin"


def test_audited_runtime_payload_rejects_changed_variable_drift(
    tmp_path: Path,
) -> None:
    contract = ComponentContract(
        component_id="inference.sahi_slicing",
        display_name="SAHI",
        category="slicing",
    )
    payload = SlicingInferenceAdapter().build_runtime_payload(
        AdapterContext(contract=contract, workspace=tmp_path),
        protocol_hash="protocol-1",
        base_command=["yolo-agent", "advanced", "certify-sahi", "--execute"],
        generated_config={},
    ).model_copy(update={"changed_variables": {"inference.other": True}})

    with pytest.raises(ValueError, match="canonical contract"):
        validate_audited_runtime_payload(payload, contract.component_id)


def test_distillation_claims_verified_loss_runtime(tmp_path: Path) -> None:
    contract = ComponentContract(
        component_id="distillation.yolo26_teacher_student",
        display_name="Distillation",
        category="distillation",
        maturity="smoke_passed",
    )
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        imgsz=640,
        workspace=tmp_path,
        options={
            "teacher": "yolo26s.pt",
            "student": "yolo26n.pt",
            "teacher_data": "coco.yaml",
            "student_data": "coco.yaml",
            "imgsz": 640,
        },
    )

    payload = YOLO26DistillationAdapter().build_runtime_payload(
        context,
        protocol_hash="protocol-1",
        base_command=[
            "yolo", "detect", "train", "model=yolo26n.pt", "data=coco.yaml",
            "imgsz=640",
        ],
        generated_config={},
    )

    assert payload.loss_plugin
    assert payload.supports_amp and payload.supports_ddp and payload.supports_resume
    payload.verify_imports()


def test_p2_head_claims_verified_model_graph_runtime(tmp_path: Path) -> None:
    contract = ComponentContract(
        component_id="head.p2_small_object",
        display_name="P2 Head",
        category="detection_head",
        maturity="smoke_passed",
    )
    context = AdapterContext(contract=contract, workspace=tmp_path)

    payload = P2HeadAdapter().build_runtime_payload(
        context,
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )

    assert payload.model_graph_plugin
    assert payload.supports_amp and payload.supports_ddp and payload.supports_resume
    assert {item.name for item in payload.expected_artifacts} == {
        "p2_head_manifest",
        "p2_model_yaml",
    }
    payload.verify_imports()


def test_small_object_sampling_claims_verified_dataloader_runtime(tmp_path: Path) -> None:
    contract = ComponentContract(
        component_id="sampling.small_object",
        display_name="Small Object Sampling",
        category="sampling",
        maturity="smoke_passed",
    )
    context = AdapterContext(contract=contract, workspace=tmp_path)

    payload = SmallObjectSamplingAdapter().build_runtime_payload(
        context,
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )

    assert payload.dataloader_plugin
    assert payload.supports_ddp and payload.supports_resume
    payload.verify_imports()
