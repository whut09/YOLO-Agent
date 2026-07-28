"""Runtime context, compatibility audit, and evidence for trainer plugins."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.core.yaml_io import YAMLModelMixin


SUPPORTED_ULTRALYTICS_SERIES = {(8, 4)}
EXPECTED_METHOD_PARAMETERS: dict[str, tuple[str, ...]] = {
    "DetectionTrainer.__init__": ("self", "cfg", "overrides", "_callbacks"),
    "DetectionTrainer.get_model": ("self", "cfg", "weights", "verbose"),
    "DetectionTrainer.build_dataset": ("self", "img_path", "mode", "batch"),
    "DetectionTrainer.get_dataloader": (
        "self",
        "dataset_path",
        "batch_size",
        "rank",
        "mode",
    ),
    "DetectionTrainer.get_validator": ("self",),
    "DetectionTrainer.save_model": ("self",),
    "DetectionTrainer.resume_training": ("self", "ckpt"),
    "DetectionTrainer.preprocess_batch": ("self", "batch"),
    "DetectionTrainer.add_callback": ("self", "event", "callback"),
    "DetectionTrainer.run_callbacks": ("self", "event"),
    "DetectionTrainer.train": ("self",),
    "DetectionModel.init_criterion": ("self",),
    "DetectionModel.loss": ("self", "batch", "preds"),
    "Model.train": ("self", "trainer", "kwargs"),
    "YOLO.__init__": ("self", "model", "task", "verbose"),
}


class UltralyticsRuntimeAudit(BaseModel):
    """Installed Ultralytics API contract required by the plugin bridge."""

    schema_version: str = "ultralytics_plugin_api.v1"
    version: str
    compatible: bool
    method_parameters: dict[str, list[str]] = Field(default_factory=dict)
    signature_hash: str
    blocked_by: list[str] = Field(default_factory=list)
    source_locations: dict[str, str] = Field(default_factory=dict)


class RuntimePluginDescriptor(BaseModel):
    """Auditable identity for one loaded runtime plugin."""

    reference: str
    class_name: str
    module: str
    version: str
    source_hash: str
    hooks: list[str] = Field(default_factory=list)


class PluginRuntimeEvidence(BaseModel, YAMLModelMixin):
    """Persisted plugin identities, hook counts, and failures."""

    schema_version: str = "ultralytics_plugin_runtime.v1"
    payload_hash: str
    protocol_hash: str
    ultralytics_version: str
    signature_hash: str
    compatible: bool
    rank: int = -1
    plugins: list[RuntimePluginDescriptor] = Field(default_factory=list)
    hook_call_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UltralyticsPluginContext(BaseModel):
    """Mutable runtime state shared by plugin hooks in one trainer process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payload: AdapterRuntimePayload
    payload_path: Path
    audit: UltralyticsRuntimeAudit
    evidence_path: Path
    evidence: PluginRuntimeEvidence
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def record_call(self, reference: str, hook: str) -> None:
        """Increment a successful or attempted hook invocation and persist it."""
        with self._lock:
            self._merge_existing_locked()
            counts = self.evidence.hook_call_counts.setdefault(reference, {})
            counts[hook] = counts.get(hook, 0) + 1
            self._persist_locked()

    def record_failure(self, reference: str, hook: str, error: Exception | str) -> None:
        """Persist a plugin failure without erasing previous invocation evidence."""
        message = f"{reference}:{hook}:{error}"
        with self._lock:
            self._merge_existing_locked()
            self.evidence.failures.append(message)
            self._persist_locked()

    def persist(self) -> Path:
        """Persist current runtime evidence atomically."""
        with self._lock:
            return self._persist_locked()

    def _persist_locked(self) -> Path:
        self.evidence.updated_at = datetime.now(timezone.utc)
        target = self.evidence_path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        self.evidence.to_json(temporary, exclude_none=True, sort_keys=True, encoding="utf-8")
        temporary.replace(target)
        return target

    def _merge_existing_locked(self) -> None:
        if not self.evidence_path.is_file():
            return
        try:
            existing = PluginRuntimeEvidence.model_validate_json(
                self.evidence_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, ValueError):
            return
        if (
            existing.payload_hash != self.evidence.payload_hash
            or existing.protocol_hash != self.evidence.protocol_hash
        ):
            return
        for reference, hooks in existing.hook_call_counts.items():
            current = self.evidence.hook_call_counts.setdefault(reference, {})
            for hook, count in hooks.items():
                current[hook] = max(current.get(hook, 0), count)
        for failure in existing.failures:
            if failure not in self.evidence.failures:
                self.evidence.failures.append(failure)


def audit_installed_ultralytics(
    *,
    version: str | None = None,
    trainer_class: type[Any] | None = None,
    model_class: type[Any] | None = None,
) -> UltralyticsRuntimeAudit:
    """Inspect the installed classes and fail closed on unreviewed API changes."""
    if trainer_class is None or model_class is None:
        try:
            from ultralytics import YOLO
            from ultralytics.engine.model import Model
            from ultralytics.models.yolo.detect.train import DetectionTrainer
            from ultralytics.nn.tasks import DetectionModel
        except ImportError as exc:
            raise ImportError("Ultralytics trainer plugins require the ultralytics package") from exc
        trainer_class = trainer_class or DetectionTrainer
        model_class = model_class or DetectionModel
    else:
        from ultralytics import YOLO
        from ultralytics.engine.model import Model
    installed_version = version or importlib.metadata.version("ultralytics")
    blocked: list[str] = []
    series = _version_series(installed_version)
    if series not in SUPPORTED_ULTRALYTICS_SERIES:
        blocked.append(f"unsupported_ultralytics_version:{installed_version}")
    classes = {
        "DetectionTrainer": trainer_class,
        "DetectionModel": model_class,
        "Model": Model,
        "YOLO": YOLO,
    }
    observed: dict[str, list[str]] = {}
    locations: dict[str, str] = {}
    for name, expected in EXPECTED_METHOD_PARAMETERS.items():
        class_name, method_name = name.split(".", 1)
        owner = classes[class_name]
        method = getattr(owner, method_name, None)
        if method is None:
            observed[name] = []
            blocked.append(f"missing_ultralytics_method:{name}")
            continue
        parameters = list(inspect.signature(method).parameters)
        observed[name] = parameters
        locations[name] = f"{method.__module__}.{owner.__qualname__}.{method_name}"
        if tuple(parameters) != expected:
            blocked.append(
                f"ultralytics_signature_mismatch:{name}:"
                f"expected={','.join(expected)}:actual={','.join(parameters)}"
            )
    signature_hash = hashlib.sha256(
        json.dumps(observed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return UltralyticsRuntimeAudit(
        version=installed_version,
        compatible=not blocked,
        method_parameters=observed,
        signature_hash=signature_hash,
        blocked_by=blocked,
        source_locations=locations,
    )


def runtime_evidence_path(payload_path: Path | str) -> Path:
    """Return a rank-safe evidence path next to the immutable payload."""
    source = Path(payload_path).resolve()
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))
    suffix = "" if rank in {-1, 0} else f".rank{rank}"
    return source.parent / f"plugin_runtime_evidence{suffix}.json"


def plugin_source_hash(implementation: Any) -> str:
    """Hash plugin source rather than trusting a class name as implementation identity."""
    try:
        source = inspect.getsource(implementation).encode("utf-8")
    except (OSError, TypeError):
        module = inspect.getmodule(implementation)
        module_path = Path(str(getattr(module, "__file__", "")))
        source = module_path.read_bytes() if module_path.is_file() else repr(implementation).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _version_series(version: str) -> tuple[int, int]:
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid Ultralytics version: {version}") from exc


__all__ = [
    "EXPECTED_METHOD_PARAMETERS",
    "PluginRuntimeEvidence",
    "RuntimePluginDescriptor",
    "SUPPORTED_ULTRALYTICS_SERIES",
    "UltralyticsPluginContext",
    "UltralyticsRuntimeAudit",
    "audit_installed_ultralytics",
    "plugin_source_hash",
    "runtime_evidence_path",
]
