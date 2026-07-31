"""CPU golden-path certification for YOLO26 teacher-student distillation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from yolo_agent.adapters.ultralytics.plugin_bridge import (
    PluginCriterionWrapper,
    UltralyticsTrainerPluginBridge,
)
from yolo_agent.adapters.ultralytics.plugin_context import (
    PluginRuntimeEvidence,
    runtime_evidence_path,
)
from yolo_agent.certification.loss_distillation_schemas import DistillationCpuReport
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationRuntimePlugin,
)
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)
from yolo_agent.recipes.schemas import AtomicRecipe, recipe_from_mapping


COMPONENT_ID = "distillation.yolo26_teacher_student"
RECIPE_ID = "yolo26n_distillation"


def run_distillation_cpu_fixture(
    *,
    runtime_payload_path: Path | str,
    workspace: Path | str,
) -> DistillationCpuReport:
    """Exercise teacher forward, student backward, and zero-weight equivalence."""
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload_path = Path(runtime_payload_path).resolve()
    payload = AdapterRuntimePayload.read(payload_path, verify_imports=True)
    _validate_payload(payload)
    report_path = root / "distillation_cpu_golden_path.yaml"
    zero_payload_path = root / "zero_weight_control" / "adapter_runtime_payload.yaml"
    zero_payload = _zero_weight_payload(payload)
    zero_payload.write(zero_payload_path)
    evidence_path = payload_path.parent / "distillation_evidence.json"
    zero_evidence_path = zero_payload_path.parent / "distillation_evidence.json"
    checks: dict[str, bool | str | int | float] = {}
    errors: list[str] = []
    try:
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel

        checks["atomic_recipe_verified"] = _atomic_recipe_verified()
        torch.manual_seed(23)
        student = DetectionModel("yolo26n.yaml", ch=3, nc=3, verbose=False)
        teacher = DetectionModel("yolo26s.yaml", ch=3, nc=3, verbose=False)
        zero_teacher = DetectionModel("yolo26s.yaml", ch=3, nc=3, verbose=False)
        student.args = get_cfg(overrides={"imgsz": 640})
        teacher.args = get_cfg(overrides={"imgsz": 640})
        zero_teacher.args = get_cfg(overrides={"imgsz": 640})
        student.train()
        trainer = SimpleNamespace(
            args=SimpleNamespace(
                imgsz=640,
                data=_payload_data(payload),
                resume=False,
            )
        )
        zero_trainer = SimpleNamespace(
            args=SimpleNamespace(
                imgsz=640,
                data=_payload_data(zero_payload),
                resume=False,
            )
        )
        bridge = UltralyticsTrainerPluginBridge(payload_path)
        zero_bridge = UltralyticsTrainerPluginBridge(zero_payload_path)
        plugin = _distillation_plugin(bridge)
        zero_plugin = _distillation_plugin(zero_bridge)
        plugin._teacher_loader = lambda _: teacher
        zero_plugin._teacher_loader = lambda _: zero_teacher
        bridge.install_model_hooks(student, trainer=trainer)
        wrapped = student.init_criterion()
        if not isinstance(wrapped, PluginCriterionWrapper):
            raise TypeError("distillation bridge did not install criterion wrapper")
        native_criterion = wrapped.criterion
        zero_criterion = zero_bridge.invoke_transform(
            "build_criterion",
            native_criterion,
            trainer=zero_trainer,
            model=student,
        )
        zero_wrapped = PluginCriterionWrapper(
            zero_criterion,
            zero_bridge,
            student,
            zero_trainer,
        )
        image = torch.rand(1, 3, 64, 64)
        batch = {
            "img": image,
            "batch_idx": torch.tensor([0, 0]),
            "cls": torch.tensor([[0.0], [1.0]]),
            "bboxes": torch.tensor(
                [
                    [0.35, 0.35, 0.25, 0.25],
                    [0.68, 0.65, 0.22, 0.28],
                ]
            ),
        }
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            predictions = student(image)
            native_loss, native_items = native_criterion(predictions, batch)
            zero_loss, zero_items = zero_wrapped(predictions, batch)
            active_loss, active_items = wrapped(predictions, batch)
        checks["zero_weight_native_equivalent"] = bool(
            torch.equal(zero_loss, native_loss)
            and torch.equal(zero_items, native_items)
        )
        checks["total_loss_changed"] = bool(
            not torch.allclose(active_loss, native_loss)
            and torch.equal(active_items, native_items)
        )
        student.zero_grad(set_to_none=True)
        active_loss.sum().backward()
        checks["student_backward"] = any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in student.parameters()
        )
        checks["teacher_no_grad"] = all(
            parameter.grad is None for parameter in teacher.parameters()
        )
        checks["teacher_frozen_eval"] = bool(
            not teacher.training
            and all(not parameter.requires_grad for parameter in teacher.parameters())
        )
        evidence = _read_json(evidence_path)
        zero_evidence = _read_json(zero_evidence_path)
        profiles = evidence.get("method_profiles", [])
        checks["method_profiles_only"] = bool(
            profiles
            and all(item.get("status") == "method_profile_only" for item in profiles)
        )
        checks["exact_reproduction_false"] = bool(
            profiles and all(item.get("exact_reproduction") is False for item in profiles)
        )
        checks["student_inference_graph_unchanged"] = bool(
            evidence.get("student_inference_graph_unchanged") is True
        )
        checks["trainer_bridge_called"] = bool(
            _compute_loss_calls(runtime_evidence_path(payload_path)) >= 1
            and _compute_loss_calls(runtime_evidence_path(zero_payload_path)) >= 1
            and evidence.get("compute_loss_calls", 0) >= 1
            and zero_evidence.get("compute_loss_calls", 0) >= 1
        )
        checks["teacher_forward_called"] = bool(
            evidence.get("teacher_forward_calls", 0) >= 1
            and zero_evidence.get("teacher_forward_calls", 0) >= 1
        )
        checks["shared_batch_tensor"] = evidence.get("shared_batch_tensor") is True
        checks["checkpoint_hashes_recorded"] = bool(
            len(str(evidence.get("teacher_checkpoint_sha256", ""))) == 64
            and len(str(evidence.get("student_checkpoint_sha256", ""))) == 64
        )
        checks["imgsz"] = 640
        failed = sorted(
            key
            for key, value in checks.items()
            if key != "imgsz" and value is not True
        )
        if failed:
            errors.append("failed distillation CPU checks: " + ", ".join(failed))
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(str(exc))

    report = DistillationCpuReport(
        status="failed" if errors else "passed",
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        zero_control_payload_hash=zero_payload.payload_hash,
        runtime_payload_path=payload_path,
        zero_control_payload_path=zero_payload_path if zero_payload_path.is_file() else None,
        runtime_evidence_path=evidence_path if evidence_path.is_file() else None,
        zero_control_evidence_path=(
            zero_evidence_path if zero_evidence_path.is_file() else None
        ),
        checks=checks,
        errors=errors,
    )
    report.to_yaml(report_path, exclude_none=True, sort_keys=False)
    return report


def _validate_payload(payload: AdapterRuntimePayload) -> None:
    if payload.component_ids != [COMPONENT_ID] or len(payload.loss_plugin) != 1:
        raise ValueError("distillation fixture requires one YOLO26 distillation plugin")


def _zero_weight_payload(payload: AdapterRuntimePayload) -> AdapterRuntimePayload:
    reference = payload.loss_plugin[0]
    options = dict(reference.options)
    weights = dict(options.get("weights", {}))
    weights.update({"logits": 0.0, "feature": 0.0, "localization": 0.0})
    options["weights"] = weights
    return payload.model_copy(
        update={
            "loss_plugin": [
                RuntimePluginReference(
                    reference=reference.reference,
                    options=options,
                    required_hooks=list(reference.required_hooks),
                )
            ]
        }
    )


def _distillation_plugin(
    bridge: UltralyticsTrainerPluginBridge,
) -> YOLO26DistillationRuntimePlugin:
    instances = [
        item.instance
        for item in bridge.plugins
        if isinstance(item.instance, YOLO26DistillationRuntimePlugin)
    ]
    if len(instances) != 1:
        raise ValueError("distillation bridge did not load exactly one runtime plugin")
    return instances[0]


def _payload_data(payload: AdapterRuntimePayload) -> str:
    return str(payload.loss_plugin[0].options["student_data"])


def _atomic_recipe_verified() -> bool:
    raw = yaml.safe_load(
        Path("configs/recipes/yolo26n_distillation.yaml").read_text(
            encoding="utf-8"
        )
    )
    recipe = recipe_from_mapping(raw)
    return bool(
        isinstance(recipe, AtomicRecipe)
        and recipe.recipe_id == RECIPE_ID
        and recipe.component_ids == [COMPONENT_ID]
        and recipe.train_overrides.get("imgsz") == 640
        and recipe.primary_changed_variable == "distillation"
        and recipe.fixed_variables.get("student_inference_graph") == "unchanged"
    )


def _compute_loss_calls(path: Path) -> int:
    evidence = PluginRuntimeEvidence.model_validate_json(
        path.read_text(encoding="utf-8-sig")
    )
    return sum(
        hooks.get("compute_loss", 0) for hooks in evidence.hook_call_counts.values()
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"runtime evidence must be a mapping: {path}")
    return value


__all__ = ["run_distillation_cpu_fixture"]
