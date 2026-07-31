"""CPU golden-path certification for additive YOLO26 quality losses."""

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
from yolo_agent.certification.loss_distillation_schemas import QualityLossCpuReport
from yolo_agent.components.adapters.losses.quality_alignment import LOSS_SPECS
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)
from yolo_agent.recipes.schemas import AtomicRecipe, recipe_from_mapping


QUALITY_RECIPE_IDS = {
    "loss.quality.correlation": "yolo26_correlation_auxiliary_loss",
    "loss.calibration.bpc": "yolo26_bpc_calibration_auxiliary_loss",
    "loss.quality.pseudo_iou": "yolo26_pseudo_iou_quality_auxiliary_loss",
}


def run_quality_loss_cpu_fixture(
    *,
    runtime_payload_path: Path | str,
    workspace: Path | str,
) -> QualityLossCpuReport:
    """Prove one runtime loss changes native total loss and backpropagates."""
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload_path = Path(runtime_payload_path).resolve()
    payload = AdapterRuntimePayload.read(payload_path, verify_imports=True)
    component_id = _quality_component(payload)
    spec = LOSS_SPECS[component_id]
    recipe_id = QUALITY_RECIPE_IDS[component_id]
    report_path = root / f"{spec.loss_name}_cpu_golden_path.yaml"
    zero_payload_path = root / "zero_weight_control" / "adapter_runtime_payload.yaml"
    zero_payload = _zero_weight_payload(payload)
    zero_payload.write(zero_payload_path)
    evidence_path = payload_path.parent / f"auxiliary_loss_{spec.loss_name}_evidence.json"
    zero_evidence_path = (
        zero_payload_path.parent / f"auxiliary_loss_{spec.loss_name}_evidence.json"
    )
    checks: dict[str, bool | str | int | float] = {}
    errors: list[str] = []
    try:
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel

        checks["atomic_recipe_verified"] = _atomic_recipe_verified(
            component_id,
            recipe_id,
        )
        torch.manual_seed(19)
        model = DetectionModel("yolo26n.yaml", ch=3, nc=3, verbose=False)
        model.args = get_cfg(overrides={"imgsz": 640})
        model.train()
        trainer = SimpleNamespace(loss_names=("box_loss", "cls_loss", "dfl_loss"))
        zero_trainer = SimpleNamespace(
            loss_names=("box_loss", "cls_loss", "dfl_loss")
        )
        bridge = UltralyticsTrainerPluginBridge(payload_path)
        zero_bridge = UltralyticsTrainerPluginBridge(zero_payload_path)
        bridge.install_model_hooks(model, trainer=trainer)
        wrapped = model.init_criterion()
        if not isinstance(wrapped, PluginCriterionWrapper):
            raise TypeError("quality loss bridge did not install criterion wrapper")
        native_criterion = wrapped.criterion
        zero_criterion = zero_bridge.invoke_transform(
            "build_criterion",
            native_criterion,
            trainer=zero_trainer,
            model=model,
        )
        zero_wrapped = PluginCriterionWrapper(
            zero_criterion,
            zero_bridge,
            model,
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
            predictions = model(image)
            if component_id == "loss.calibration.bpc":
                predictions = _force_confident_one2many_scores(predictions)
            native_loss, native_items = native_criterion(predictions, batch)
            zero_loss, zero_items = zero_wrapped(predictions, batch)
            active_loss, active_items = wrapped(predictions, batch)

        checks["zero_weight_native_equivalent"] = bool(
            torch.equal(zero_loss, native_loss)
            and torch.equal(zero_items[: native_items.numel()], native_items)
            and float(zero_items[-1]) == 0.0
        )
        checks["total_loss_changed"] = bool(
            not torch.allclose(active_loss, native_loss)
            and active_items.numel() == native_items.numel() + 1
            and float(active_items[-1].detach().float()) != 0.0
        )
        model.zero_grad(set_to_none=True)
        active_loss.sum().backward()
        checks["student_backward"] = any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        runtime_evidence = _read_json(evidence_path)
        zero_runtime_evidence = _read_json(zero_evidence_path)
        paper_prior = runtime_evidence.get("paper_prior", {})
        checks["paper_prior_not_local_evidence"] = bool(
            paper_prior.get("evidence_level") == "paper_prior"
            and not paper_prior.get("reported_delta")
        )
        checks["exact_reproduction_false"] = (
            paper_prior.get("exact_reproduction") is False
        )
        checks["native_assigner_preserved"] = bool(
            runtime_evidence.get("replaces_assigner") is False
        )
        checks["native_bbox_regression_preserved"] = bool(
            runtime_evidence.get("replaces_bbox_regression") is False
            and runtime_evidence.get("native_dfl_enabled") is False
        )
        checks["trainer_bridge_called"] = bool(
            _compute_loss_calls(runtime_evidence_path(payload_path)) >= 1
            and _compute_loss_calls(runtime_evidence_path(zero_payload_path)) >= 1
            and runtime_evidence.get("compute_loss_calls", 0) >= 1
            and zero_runtime_evidence.get("compute_loss_calls", 0) >= 1
        )
        checks["imgsz"] = 640
        failed = sorted(
            key
            for key, value in checks.items()
            if key != "imgsz" and value is not True
        )
        if failed:
            errors.append("failed quality loss CPU checks: " + ", ".join(failed))
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

    report = QualityLossCpuReport(
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


def _quality_component(payload: AdapterRuntimePayload) -> str:
    if len(payload.component_ids) != 1 or payload.component_ids[0] not in LOSS_SPECS:
        raise ValueError("quality loss fixture requires one supported loss component")
    return payload.component_ids[0]


def _zero_weight_payload(payload: AdapterRuntimePayload) -> AdapterRuntimePayload:
    if len(payload.loss_plugin) != 1:
        raise ValueError("quality loss fixture requires one loss plugin")
    reference = payload.loss_plugin[0]
    options = dict(reference.options)
    options["weight"] = 0.0
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


def _atomic_recipe_verified(component_id: str, recipe_id: str) -> bool:
    raw = yaml.safe_load(
        Path("configs/recipes/yolo26_quality_alignment.yaml").read_text(
            encoding="utf-8"
        )
    )
    recipes = [recipe_from_mapping(item) for item in raw["recipes"]]
    recipe = next((item for item in recipes if item.recipe_id == recipe_id), None)
    return bool(
        isinstance(recipe, AtomicRecipe)
        and recipe.component_ids == [component_id]
        and recipe.train_overrides.get("imgsz") == 640
        and recipe.primary_changed_variable == LOSS_SPECS[component_id].changed_variable
        and all(
            item.get("evidence_level") == "paper_prior"
            and item.get("local_evidence") is False
            for item in recipe.evidence_prior
        )
    )


def _compute_loss_calls(path: Path) -> int:
    evidence = PluginRuntimeEvidence.model_validate_json(
        path.read_text(encoding="utf-8-sig")
    )
    return sum(
        hooks.get("compute_loss", 0) for hooks in evidence.hook_call_counts.values()
    )


def _force_confident_one2many_scores(predictions: Any) -> Any:
    """Create a deterministic high-confidence BPC probe without changing targets."""
    container = predictions[1] if isinstance(predictions, tuple) else predictions
    if not isinstance(container, dict) or "one2many" not in container:
        raise ValueError("BPC certification requires YOLO26 one2many predictions")
    branch = container["one2many"]
    if not isinstance(branch, dict) or "scores" not in branch:
        raise ValueError("BPC certification predictions are missing scores")
    updated_branch = dict(branch)
    updated_branch["scores"] = branch["scores"] + 8.0
    updated_container = dict(container)
    updated_container["one2many"] = updated_branch
    if isinstance(predictions, tuple):
        values = list(predictions)
        values[1] = updated_container
        return tuple(values)
    return updated_container


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"runtime evidence must be a mapping: {path}")
    return value


__all__ = ["QUALITY_RECIPE_IDS", "run_quality_loss_cpu_fixture"]
