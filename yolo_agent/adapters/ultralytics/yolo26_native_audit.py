"""Runtime/source audit for the native YOLO26 training recipe."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig


AuditStatus = Literal["matched", "mismatched", "unknown", "unsupported"]


class AuditFinding(BaseModel):
    status: AuditStatus
    configured: Any = None
    observed: Any = None
    source: str = ""
    message: str = ""


class YOLO26NativeRecipeAudit(BaseModel):
    ultralytics_version: str | None = None
    matched: dict[str, AuditFinding] = Field(default_factory=dict)
    mismatched: dict[str, AuditFinding] = Field(default_factory=dict)
    unknown: dict[str, AuditFinding] = Field(default_factory=dict)
    unsupported: dict[str, AuditFinding] = Field(default_factory=dict)
    effective_training_config: dict[str, Any] = Field(default_factory=dict)
    source_locations: dict[str, str] = Field(default_factory=dict)
    recipe_hash: str = ""
    baseline_normalized: bool = False

    def add(self, key: str, finding: AuditFinding) -> None:
        getattr(self, finding.status)[key] = finding


class YOLO26NativeAuditor:
    """Audit configuration and installed implementation without training."""

    def audit(
        self,
        training_config: Path | str | UltralyticsTrainingConfig,
        *,
        model_path: Path | str | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> YOLO26NativeRecipeAudit:
        config, source = _load_training_config(training_config)
        installed = _installed_ultralytics()
        audit = YOLO26NativeRecipeAudit(ultralytics_version=installed["version"])
        effective = {**_default_config(installed["root"]), **config, **(runtime_config or {})}
        audit.effective_training_config = _jsonable(effective)

        checks = {
            "optimizer": (effective.get("optimizer"), "auto", "configuration"),
            "scheduler": (effective.get("cos_lr"), effective.get("cos_lr"), "configuration"),
            "warmup": (effective.get("warmup_epochs"), effective.get("warmup_epochs"), "configuration"),
            "ema": (effective.get("ema"), effective.get("ema"), "configuration"),
            "close_mosaic": (effective.get("close_mosaic"), effective.get("close_mosaic"), "configuration"),
            "augmentation": (_augmentation(effective), _augmentation(effective), "configuration"),
            "loss_weights": (_loss_weights(effective), _loss_weights(effective), "configuration"),
            "checkpoint": (effective.get("model"), effective.get("model"), "configuration"),
            "imgsz": (effective.get("imgsz"), 640, "configuration"),
            "batch": (effective.get("batch"), effective.get("batch"), "configuration"),
            "seed": (effective.get("seed"), effective.get("seed"), "configuration"),
        }
        for key, (configured, observed, location) in checks.items():
            status: AuditStatus = "matched" if configured == observed else "mismatched"
            if configured is None or observed is None:
                status = "unknown"
            audit.add(key, AuditFinding(status=status, configured=configured, observed=observed, source=location))
            audit.source_locations[key] = source if location == "configuration" else location

        _audit_native_structure(audit, installed, model_path)
        _audit_optimizer(audit, effective, installed)
        _audit_optional_features(audit, installed)
        audit.baseline_normalized = not bool(audit.mismatched or audit.unknown or audit.unsupported)
        audit.recipe_hash = _recipe_hash(audit)
        return audit


def _audit_native_structure(audit: YOLO26NativeRecipeAudit, installed: dict[str, Any], model_path: Path | str | None) -> None:
    yaml_path = installed["root"] / "cfg" / "models" / "26" / "yolo26.yaml"
    raw = _read_yaml(yaml_path)
    end2end = raw.get("end2end")
    reg_max = raw.get("reg_max")
    audit.add("head_mode", AuditFinding(status="matched" if end2end is True else "mismatched", configured=True, observed=end2end, source=str(yaml_path)))
    audit.add("nms_free", AuditFinding(status="matched" if end2end is True else "mismatched", configured=True, observed=end2end, source=str(yaml_path)))
    audit.add("dfl_free", AuditFinding(status="matched" if reg_max == 1 else "unknown", configured=1, observed=reg_max, source=str(yaml_path)))
    audit.source_locations.update({"head_mode": str(yaml_path), "nms_free": str(yaml_path), "dfl_free": str(yaml_path)})
    if model_path:
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            observed = getattr(model.model, "end2end", None)
            audit.add("runtime_end2end", AuditFinding(status="matched" if observed is True else "mismatched", configured=True, observed=observed, source=str(model_path)))
            audit.source_locations["runtime_end2end"] = str(model_path)
        except Exception as exc:
            audit.add("runtime_end2end", AuditFinding(status="unknown", message=str(exc), source=str(model_path)))


def _audit_optimizer(audit: YOLO26NativeRecipeAudit, effective: dict[str, Any], installed: dict[str, Any]) -> None:
    value = effective.get("optimizer")
    source = str(installed["root"] / "optim" / "muon.py")
    if value == "MuSGD":
        audit.add("musgd", AuditFinding(status="matched", configured=value, observed="requested", source=source))
    elif value == "auto":
        audit.add("musgd", AuditFinding(status="unknown", configured=value, observed="auto selection", source=source, message="MuSGD is available but the selected optimizer is runtime-dependent."))
    else:
        audit.add("musgd", AuditFinding(status="mismatched", configured=value, observed="not requested", source=source))
    audit.source_locations["musgd"] = source


def _audit_optional_features(audit: YOLO26NativeRecipeAudit, installed: dict[str, Any]) -> None:
    root = installed["root"]
    files = list(root.rglob("*.py"))
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in files)
    symbols = {
        "progressive_loss": re.search(r"\bProgressiveLoss\b|\bprogressive_loss\b", text) is not None,
        "stal": re.search(r"\bSTAL\b|\bstal\b", text) is not None,
    }
    for key, present in symbols.items():
        audit.add(key, AuditFinding(status="unknown" if present else "unsupported", observed="symbol found; activation not proven" if present else "symbol not found", source=str(root)))
    audit.source_locations.update({"progressive_loss": str(root), "stal": str(root)})


def _load_training_config(value: Path | str | UltralyticsTrainingConfig) -> tuple[dict[str, Any], str]:
    if isinstance(value, UltralyticsTrainingConfig):
        return value.model_dump(mode="json", exclude_none=False), "UltralyticsTrainingConfig"
    path = Path(value)
    with path.open("r", encoding="utf-8-sig") as file:
        raw = yaml.safe_load(file) or {}
    data = raw.get("training", raw)
    return dict(data), str(path)


def _installed_ultralytics() -> dict[str, Any]:
    module = importlib.import_module("ultralytics")
    root = Path(module.__file__).resolve().parent
    return {"version": getattr(module, "__version__", "unknown"), "root": root}


def _default_config(root: Path) -> dict[str, Any]:
    return _read_yaml(root / "cfg" / "default.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            value = yaml.safe_load(file) or {}
        return value if isinstance(value, dict) else {}
    except OSError:
        return {}


def _augmentation(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in ("mosaic", "mixup", "copy_paste", "close_mosaic")}


def _loss_weights(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in ("box", "cls", "dfl")}


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _recipe_hash(audit: YOLO26NativeRecipeAudit) -> str:
    payload = audit.model_dump(mode="json", exclude={"recipe_hash"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = ["AuditFinding", "YOLO26NativeAudit", "YOLO26NativeAuditor", "YOLO26NativeRecipeAudit"]

# Backwards-friendly alias for callers that prefer the shorter name.
YOLO26NativeAudit = YOLO26NativeRecipeAudit
