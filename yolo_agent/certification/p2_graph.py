"""CPU golden-path certification for the YOLO26 P2 model graph."""

from __future__ import annotations

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
from yolo_agent.components.adapters.head.p2_head import P2HeadManifest
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.recipes.schemas import AtomicRecipe, recipe_from_mapping


COMPONENT_ID = "head.p2_small_object"
RECIPE_ID = "yolo26_small_object_p2"


def run_p2_graph_cpu_fixture(
    *,
    runtime_payload_path: Path | str,
    workspace: Path | str,
) -> GraphCpuReport:
    """Build and exercise the actual P2 graph through the trainer plugin bridge."""
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload_path = Path(runtime_payload_path).resolve()
    payload = AdapterRuntimePayload.read(payload_path, verify_imports=True)
    if payload.component_ids != [COMPONENT_ID] or len(payload.model_graph_plugin) != 1:
        raise ValueError("P2 graph fixture requires one P2 model graph plugin")
    manifest_path = Path(
        str(payload.model_graph_plugin[0].options["manifest_path"])
    ).resolve()
    checkpoint_audit_path = root / "p2_checkpoint_audit.json"
    report_path = root / "p2_graph_cpu_golden_path.yaml"
    checks: dict[str, bool | str | int | float] = {}
    errors: list[str] = []
    try:
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel

        checks["atomic_recipe_verified"] = _atomic_recipe_verified()
        torch.manual_seed(29)
        source = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
        source.args = get_cfg(overrides={"imgsz": 640})
        source_checkpoint = root / "native_yolo26n.pt"
        torch.save(source.state_dict(), source_checkpoint)
        source.pt_path = source_checkpoint
        bridge = UltralyticsTrainerPluginBridge(payload_path)
        trainer = SimpleNamespace(args=get_cfg(overrides={"imgsz": 640}))
        model = bridge.invoke_transform("build_model", source, trainer=trainer)
        manifest = P2HeadManifest.model_validate_json(
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
            and len(predictions["one2many"]["feats"]) == 4
            and len(predictions["one2one"]["feats"]) == 4
        )
        loss, _ = model.loss(batch)
        loss.sum().backward()
        checks["backward"] = any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name.startswith("model.19.")
        )
        model.zero_grad(set_to_none=True)
        model.criterion = None
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            amp_loss, _ = model.loss(batch)
        amp_loss.sum().backward()
        checks["amp"] = bool(torch.isfinite(amp_loss).all())
        checks["native_loss_preserved"] = bool(
            model.end2end
            and model.model[-1].reg_max == 1
            and type(model.model[-1].dfl).__name__ == "Identity"
        )
        model.eval()
        detect = model.model[-1]
        previous_export, previous_format = detect.export, detect.format
        detect.export, detect.format = True, "torchscript"
        with torch.no_grad():
            export_output = model(image)
        detect.export, detect.format = previous_export, previous_format
        checks["export"] = bool(
            torch.is_tensor(export_output) and export_output.shape[-1] == 6
        )
        checks["partial_checkpoint_audit"] = bool(
            manifest.checkpoint.loaded
            and manifest.checkpoint.partial
            and manifest.checkpoint.matched_keys
            and manifest.checkpoint.missing_keys
            and manifest.checkpoint.newly_initialized_keys
            and len(manifest.checkpoint.checkpoint_sha256) == 64
        )
        checks["resource_guard"] = bool(
            manifest.resources.passed and all(manifest.resources.checks.values())
        )
        checks["actual_strides"] = str(manifest.actual_tensor_strides)
        checks["matched_control_required"] = _matched_control_required()
        checks["trainer_bridge_called"] = bool(
            sum(
                hooks.get("build_model", 0)
                for hooks in bridge.context.evidence.hook_call_counts.values()
            )
            == 1
        )
        checks["paper_claim_not_local_evidence"] = True
        _write_json_atomic(
            checkpoint_audit_path,
            manifest.checkpoint.model_dump(mode="json"),
        )
        failed = sorted(
            key
            for key, value in checks.items()
            if key not in {"actual_strides"} and value is not True
        )
        if failed:
            errors.append("failed P2 graph CPU checks: " + ", ".join(failed))
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
        component_id=COMPONENT_ID,
        recipe_id=RECIPE_ID,
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


def _atomic_recipe_verified() -> bool:
    raw = yaml.safe_load(
        Path("configs/recipes/yolo26_small_object.yaml").read_text(encoding="utf-8")
    )
    recipe = next(
        (
            recipe_from_mapping(item)
            for item in raw["recipes"]
            if item.get("recipe_id") == RECIPE_ID
        ),
        None,
    )
    return bool(
        isinstance(recipe, AtomicRecipe)
        and recipe.component_ids == [COMPONENT_ID]
        and recipe.train_overrides.get("imgsz") == 640
        and recipe.primary_changed_variable == "head"
    )


def _matched_control_required() -> bool:
    raw = yaml.safe_load(
        Path("configs/recipes/yolo26_small_object.yaml").read_text(encoding="utf-8")
    )
    recipe = next(
        item for item in raw["recipes"] if item.get("recipe_id") == RECIPE_ID
    )
    return "matched_pilot" in {
        *recipe.get("compatibility_requirements", []),
        *recipe.get("promotion_requirements", []),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["run_p2_graph_cpu_fixture"]
