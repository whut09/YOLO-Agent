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
from yolo_agent.components.distillation import DISTILLATION_COMPONENTS


COMPONENT_ID = "distillation.yolo26_teacher_student"
RECIPE_ID = "yolo26n_distillation"
DISTILLATION_RECIPE_IDS = {
    "distillation.logits": "yolo26_logits_distillation",
    "distillation.feature": "yolo26_feature_distillation",
    "distillation.localization": "yolo26_localization_distillation",
    "distillation.relation": "yolo26_relation_distillation",
    "distillation.attention": "yolo26_attention_distillation",
    "distillation.masked_feature": "yolo26_masked_feature_distillation",
    "distillation.quality_aware": "yolo26_quality_aware_distillation",
    "distillation.teacher_ensemble": "yolo26_teacher_ensemble_distillation",
}


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
    component_id = _validate_payload(payload)
    mechanism = payload.loss_plugin[0].options.get("mechanism")
    recipe_id = DISTILLATION_RECIPE_IDS.get(component_id, RECIPE_ID)
    report_name = (
        "distillation_cpu_golden_path.yaml"
        if mechanism is None
        else f"distillation_{mechanism}_cpu_golden_path.yaml"
    )
    report_path = root / report_name
    zero_payload_path = root / "zero_weight_control" / "adapter_runtime_payload.yaml"
    zero_payload = _zero_weight_payload(payload)
    zero_payload.write(zero_payload_path)
    evidence_name = (
        "distillation_evidence.json"
        if mechanism is None
        else f"distillation_{mechanism}_evidence.json"
    )
    evidence_path = payload_path.parent / evidence_name
    zero_evidence_path = zero_payload_path.parent / evidence_name
    checks: dict[str, bool | str | int | float] = {}
    errors: list[str] = []
    try:
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel

        checks["atomic_recipe_verified"] = _atomic_recipe_verified(component_id)
        torch.manual_seed(23)
        student = DetectionModel("yolo26n.yaml", ch=3, nc=3, verbose=False)
        teacher = DetectionModel("yolo26s.yaml", ch=3, nc=3, verbose=False)
        zero_teacher = DetectionModel("yolo26s.yaml", ch=3, nc=3, verbose=False)
        teacher_m = DetectionModel("yolo26m.yaml", ch=3, nc=3, verbose=False)
        zero_teacher_m = DetectionModel("yolo26m.yaml", ch=3, nc=3, verbose=False)
        student.args = get_cfg(overrides={"imgsz": 640})
        teacher.args = get_cfg(overrides={"imgsz": 640})
        zero_teacher.args = get_cfg(overrides={"imgsz": 640})
        teacher_m.args = get_cfg(overrides={"imgsz": 640})
        zero_teacher_m.args = get_cfg(overrides={"imgsz": 640})
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
        plugin._teacher_loader = (
            lambda path: teacher_m if path.name == "yolo26m.pt" else teacher
        )
        zero_plugin._teacher_loader = (
            lambda path: zero_teacher_m if path.name == "yolo26m.pt" else zero_teacher
        )
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
            not torch.equal(active_loss, native_loss)
            and torch.equal(active_items, native_items)
        )
        student.zero_grad(set_to_none=True)
        active_loss.sum().backward()
        plugin.on_model_serialize_start(context=bridge.context, trainer=trainer)
        plugin.on_model_serialize_end(context=bridge.context, trainer=trainer)
        bridge.context.persist()
        zero_bridge.context.persist()
        checks["student_backward"] = any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in student.parameters()
        )
        checks["teacher_no_grad"] = all(
            parameter.grad is None
            for active_teacher in plugin.teachers
            for parameter in active_teacher.parameters()
        )
        checks["teacher_frozen_eval"] = bool(
            all(not active_teacher.training for active_teacher in plugin.teachers)
            and all(
                not parameter.requires_grad
                for active_teacher in plugin.teachers
                for parameter in active_teacher.parameters()
            )
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
        checks["runtime_evidence_identity"] = bool(
            evidence.get("runtime_payload_hash") == payload.payload_hash
            and evidence.get("changed_variables") == payload.changed_variables
        )
        checks["evidence_total_loss_changed"] = bool(
            evidence.get("total_loss_changed") is True
            and evidence.get("total_loss_after") != evidence.get("native_loss_before")
        )
        checks["trainer_bridge_called"] = bool(
            _compute_loss_calls(runtime_evidence_path(payload_path)) >= 1
            and _compute_loss_calls(runtime_evidence_path(zero_payload_path)) >= 1
            and evidence.get("compute_loss_calls", 0) >= 1
            and zero_evidence.get("compute_loss_calls", 0) >= 1
        )
        checks["teacher_forward_called"] = bool(
            evidence.get("teacher_forward_calls", 0) >= len(plugin.teachers)
            and zero_evidence.get("teacher_forward_calls", 0)
            >= len(zero_plugin.teachers)
        )
        checks["shared_batch_tensor"] = evidence.get("shared_batch_tensor") is True
        options = payload.loss_plugin[0].options
        checks["teacher_checkpoint_bound"] = bool(
            evidence.get("teacher_checkpoint") == options.get("teacher")
            and evidence.get("teacher_checkpoint_sha256")
            == options.get("teacher_checkpoint_sha256")
        )
        checks["student_checkpoint_bound"] = bool(
            evidence.get("student_checkpoint") == options.get("student")
            and evidence.get("student_checkpoint_sha256")
            == options.get("student_checkpoint_sha256")
        )
        checks["dataset_protocol_bound"] = bool(
            evidence.get("dataset_hash") == options.get("dataset_hash")
            == options.get("teacher_dataset_hash")
            == options.get("student_dataset_hash")
            and evidence.get("teacher_dataset") == options.get("teacher_data")
            and evidence.get("student_dataset") == options.get("student_data")
        )
        checks["same_split_bound"] = bool(
            evidence.get("teacher_split") == evidence.get("student_split") == "train"
            and evidence.get("teacher_split") == options.get("teacher_split")
            and evidence.get("student_split") == options.get("student_split")
        )
        checks["loss_mode_bound"] = bool(
            evidence.get("loss_mode") == (options.get("mechanism") or "multi_term")
        )
        checks["student_only_export"] = bool(
            evidence.get("student_only_export") is True
            and evidence.get("teacher_exported") is False
        )
        checks["mechanism_identity"] = bool(
            evidence.get("component_id") == component_id
            and evidence.get("mechanism") == mechanism
        )
        checks["loss_contribution_recorded"] = bool(
            evidence.get("latest_loss_contribution")
            == evidence.get("latest_terms", {}).get("total")
        )
        checks["feature_hooks_verified"] = bool(
            evidence.get("feature_hooks_validated") is True
        )
        checks["teacher_ensemble_verified"] = bool(
            mechanism != "teacher_ensemble"
            or (
                len(evidence.get("teacher_checkpoints", [])) >= 2
                and len(evidence.get("teacher_checkpoint_sha256s", [])) >= 2
            )
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
        component_id=component_id,
        recipe_id=recipe_id,
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


def _validate_payload(payload: AdapterRuntimePayload) -> str:
    if len(payload.component_ids) != 1 or len(payload.loss_plugin) != 1:
        raise ValueError("distillation fixture requires one YOLO26 distillation plugin")
    component_id = payload.component_ids[0]
    if component_id != COMPONENT_ID and component_id not in DISTILLATION_COMPONENTS:
        raise ValueError("distillation fixture received an unsupported component")
    return component_id


def _zero_weight_payload(payload: AdapterRuntimePayload) -> AdapterRuntimePayload:
    reference = payload.loss_plugin[0]
    options = dict(reference.options)
    changed_variables = dict(payload.changed_variables)
    if options.get("mechanism") is not None:
        options["weight"] = 0.0
        changed_variables[str(options["changed_variable"])] = 0.0
    else:
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
            ],
            "changed_variables": changed_variables,
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


def _atomic_recipe_verified(component_id: str) -> bool:
    if component_id == COMPONENT_ID:
        raw = yaml.safe_load(
            Path("configs/recipes/yolo26n_distillation.yaml").read_text(
                encoding="utf-8"
            )
        )
        recipe = recipe_from_mapping(raw)
    else:
        raw = yaml.safe_load(
            Path("configs/recipes/yolo26_distillation_mechanisms.yaml").read_text(
                encoding="utf-8"
            )
        )
        recipe_id = DISTILLATION_RECIPE_IDS[component_id]
        recipe = next(
            recipe_from_mapping(item)
            for item in raw["recipes"]
            if item["recipe_id"] == recipe_id
        )
    return bool(
        isinstance(recipe, AtomicRecipe)
        and recipe.recipe_id == DISTILLATION_RECIPE_IDS.get(component_id, RECIPE_ID)
        and recipe.component_ids == [component_id]
        and recipe.train_overrides.get("imgsz") == 640
        and (
            recipe.primary_changed_variable == "distillation"
            if component_id == COMPONENT_ID
            else recipe.primary_changed_variable
            == DISTILLATION_COMPONENTS[component_id].changed_variable
        )
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
