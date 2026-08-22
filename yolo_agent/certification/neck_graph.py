"""CPU golden-path certification for isolated YOLO26 neck graph plugins."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from yolo_agent.adapters.ultralytics.plugin_bridge import (
    UltralyticsTrainerPluginBridge,
)
from yolo_agent.certification.graph_assignment_schemas import GraphCpuReport
from yolo_agent.components.adapters.neck.common import YOLO26NeckManifest
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.recipes.schemas import AtomicRecipe, recipe_from_mapping


NECK_RECIPE_IDS = {
    "neck.multi_scale_fusion": "yolo26_generic_multi_scale_fusion",
    "neck.gold_gather_distribute": "yolo26_gold_gather_distribute_neck",
    "neck.rtmdet_large_kernel": "yolo26_rtmdet_large_kernel_neck",
    "neck.weighted_feature_pyramid": "yolo26_weighted_feature_pyramid",
    "neck.bidirectional_feature_fusion": "yolo26_bidirectional_feature_fusion",
    "neck.lightweight": "yolo26_lightweight_neck",
    "block.reparameterized_convolution": "yolo26_reparameterized_convolution",
    "attention.channel": "yolo26_channel_attention",
    "attention.spatial": "yolo26_spatial_attention",
    "neck.deformable_feature_aggregation": "yolo26_deformable_feature_aggregation",
    "feature_pyramid.multi_scale": "yolo26_feature_pyramid_multi_scale",
}


def run_neck_graph_cpu_fixture(
    *,
    runtime_payload_path: Path | str,
    workspace: Path | str,
) -> GraphCpuReport:
    """Exercise one real pre-Detect neck wrapper and validate its audit manifest."""
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload_path = Path(runtime_payload_path).resolve()
    payload = AdapterRuntimePayload.read(payload_path, verify_imports=True)
    component_id = _neck_component(payload)
    recipe_id = NECK_RECIPE_IDS[component_id]
    manifest_path = Path(
        str(payload.model_graph_plugin[0].options["manifest_path"])
    ).resolve()
    checkpoint_audit_path = root / f"{component_id.replace('.', '_')}_checkpoint.json"
    report_path = root / f"{component_id.replace('.', '_')}_cpu_golden_path.yaml"
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
        torch.manual_seed(31)
        model = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
        model.args = get_cfg(overrides={"imgsz": 640})
        checkpoint = root / "native_yolo26n.pt"
        torch.save(model.state_dict(), checkpoint)
        model.pt_path = checkpoint
        bridge = UltralyticsTrainerPluginBridge(payload_path)
        trainer = SimpleNamespace(args=get_cfg(overrides={"imgsz": 640}))
        transformed = bridge.invoke_transform("build_model", model, trainer=trainer)
        if transformed is not model:
            raise RuntimeError("neck runtime unexpectedly replaced the model object")
        manifest = YOLO26NeckManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8-sig")
        )
        image = torch.rand(1, 3, 64, 64)
        batch = {
            "img": image,
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
        model.train()
        predictions = model(image)
        checks["real_forward"] = bool(
            isinstance(predictions, dict)
            and set(predictions) == {"one2many", "one2one"}
            and len(predictions["one2many"]["feats"]) == 3
        )
        loss, _ = model.loss(batch)
        loss.sum().backward()
        checks["backward"] = any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if ".neck." in name
        )
        model.zero_grad(set_to_none=True)
        model.criterion = None
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            amp_loss, _ = model.loss(batch)
        amp_loss.sum().backward()
        checks["amp"] = bool(torch.isfinite(amp_loss).all())
        wrapper = model.model[-1]
        checks["native_loss_preserved"] = bool(
            model.end2end
            and wrapper.reg_max == 1
            and type(wrapper.dfl).__name__ == "Identity"
            and manifest.external_nms_added is False
        )
        checks["partial_checkpoint_audit"] = bool(
            manifest.checkpoint.loaded
            and manifest.checkpoint.partial
            and manifest.checkpoint.matched_keys
            and manifest.checkpoint.newly_initialized_keys
            and len(manifest.checkpoint.checkpoint_sha256) == 64
        )
        checks["export"] = manifest.export_dry_run is True
        checks["resource_guard"] = bool(
            manifest.resources.passed and all(manifest.resources.checks.values())
        )
        checks["matched_control_required"] = _matched_control_required(recipe_id)
        checks["trainer_bridge_called"] = bool(
            sum(
                hooks.get("build_model", 0)
                for hooks in bridge.context.evidence.hook_call_counts.values()
            )
            == 1
        )
        checks["paper_claim_not_local_evidence"] = bool(
            not manifest.exact_paper_reproduction
        )
        checks["mechanism_bound"] = bool(
            manifest.mechanism == payload.model_graph_plugin[0].options["kind"]
            and len(manifest.configuration_hash) == 64
        )
        checks["deformable_operator_verified"] = bool(
            component_id != "neck.deformable_feature_aggregation"
            or (
                manifest.dependency_available
                and manifest.operator_module == "torchvision.ops"
                and manifest.operator_class == "DeformConv2d"
                and manifest.operator_call_count > 0
            )
        )
        checks["training_deploy_equivalence"] = _repconv_deploy_equivalent(
            wrapper.neck,
            image,
        )
        checks["input_strides"] = str(manifest.input_strides)
        _write_json_atomic(
            checkpoint_audit_path,
            manifest.checkpoint.model_dump(mode="json"),
        )
        failed = sorted(
            key
            for key, value in checks.items()
            if key not in {"input_strides"} and value is not True
        )
        if failed:
            errors.append("failed neck graph CPU checks: " + ", ".join(failed))
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

    report = GraphCpuReport(
        component_id=component_id,
        recipe_id=recipe_id,
        status="failed" if errors else "passed",
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        runtime_payload_path=payload_path,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        checkpoint_audit_path=(
            checkpoint_audit_path if checkpoint_audit_path.is_file() else None
        ),
        checks=checks,
        errors=errors,
    )
    report.to_yaml(report_path, exclude_none=True, sort_keys=False)
    return report


def _neck_component(payload: AdapterRuntimePayload) -> str:
    if (
        len(payload.component_ids) != 1
        or payload.component_ids[0] not in NECK_RECIPE_IDS
        or len(payload.model_graph_plugin) != 1
    ):
        raise ValueError("neck fixture requires one supported neck graph plugin")
    return payload.component_ids[0]


def _recipes() -> list[AtomicRecipe]:
    paths = [
        Path("configs/recipes/yolo26_multi_scale_necks.yaml"),
        Path("configs/recipes/independent_paper_components.yaml"),
    ]
    recipes: list[AtomicRecipe] = []
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        recipes.extend(recipe_from_mapping(item) for item in raw["recipes"])
    return recipes


def _atomic_recipe_verified(component_id: str, recipe_id: str) -> bool:
    recipe = next((item for item in _recipes() if item.recipe_id == recipe_id), None)
    return bool(
        isinstance(recipe, AtomicRecipe)
        and recipe.component_ids == [component_id]
        and recipe.train_overrides.get("imgsz") == 640
        and recipe.primary_changed_variable
        == (
            "model.feature_pyramid"
            if component_id == "feature_pyramid.multi_scale"
            else "model.neck_plugin"
        )
    )


def _matched_control_required(recipe_id: str) -> bool:
    recipe = next(item for item in _recipes() if item.recipe_id == recipe_id)
    requirements = {
        *recipe.compatibility_requirements,
        *recipe.promotion_requirements,
    }
    return bool({"matched_pilot", "matched_control"} & requirements)


def _repconv_deploy_equivalent(neck: Any, image: Any) -> bool:
    import torch

    if not hasattr(neck, "switch_to_deploy"):
        return True
    neck.eval()
    channels = neck.input_contract.channels
    strides = neck.input_contract.strides
    features = [
        image.new_empty(1, channel, 64 // stride, 64 // stride).normal_()
        for channel, stride in zip(channels, strides, strict=True)
    ]
    deployed = copy.deepcopy(neck)
    with torch.no_grad():
        expected = neck(features)
        deployed.switch_to_deploy()
        actual = deployed(features)
    return all(
        torch.allclose(left, right, atol=1e-5, rtol=1e-4)
        for left, right in zip(expected, actual, strict=True)
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["NECK_RECIPE_IDS", "run_neck_graph_cpu_fixture"]
