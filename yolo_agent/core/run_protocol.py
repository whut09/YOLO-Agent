"""Versioned execution protocol shared by runs, plans, ASHA, and evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from yolo_agent.core.matched_baseline import stable_identity_hash
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.yaml_io import YAMLModelMixin

if TYPE_CHECKING:
    from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig


RUN_PROTOCOL_SCHEMA_VERSION = "run_protocol.v1"


class RunProtocolVersion(BaseModel, YAMLModelMixin):
    """Immutable identity for one comparable training/evaluation protocol."""

    schema_version: str = RUN_PROTOCOL_SCHEMA_VERSION
    model: str
    dataset_version: str
    dataset_manifest_sha256: str
    subset_manifest_sha256: str
    imgsz: int = Field(ge=1)
    epochs: int = Field(ge=1)
    seed: int | str
    batch_policy: dict[str, Any]
    batch_policy_hash: str
    ultralytics_version: str
    eval_protocol: dict[str, Any]
    eval_protocol_hash: str
    code_version: str
    profile: str
    protocol_hash: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_protocol_hash(self) -> "RunProtocolVersion":
        expected = self.semantic_hash()
        if self.protocol_hash and self.protocol_hash != expected:
            raise ValueError("run protocol hash does not match its semantic payload")
        self.protocol_hash = expected
        return self

    def semantic_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"protocol_hash", "created_at"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_run_protocol_version(
    *,
    model: str,
    context: RunContext,
    training_config: "UltralyticsTrainingConfig",
    profile: str,
    seed: int | str | None = None,
    epochs: int | None = None,
    fraction: float | None = None,
    code_version: str | None = None,
    ultralytics_version: str | None = None,
) -> RunProtocolVersion:
    """Build a protocol identity from the effective profile and fixed evaluation contract."""
    budget = training_config.budget_profiles[profile]  # type: ignore[index]
    effective_seed = seed if seed is not None else (budget.seeds[0] if budget.seeds else context.seed)
    dataset_manifest = str(context.dataset_manifest_sha256 or "missing")
    effective_epochs = epochs if epochs is not None else budget.epochs
    effective_fraction = fraction if fraction is not None else budget.fraction
    subset_manifest = stable_identity_hash(
        {
            "schema": "training_subset.v1",
            "dataset_manifest_sha256": dataset_manifest,
            "split": "train",
            "fraction": effective_fraction,
            "seed": str(effective_seed),
        }
    )
    batch_policy = {
        "schema": "batch_policy.v1",
        "requested_batch": budget.batch,
        "auto_tuning_enabled": training_config.batch_tuning.enabled,
        "candidate_batches": training_config.batch_tuning.candidate_batches,
        "auto_expand_candidates": training_config.batch_tuning.auto_expand_candidates,
        "max_candidate_batch": training_config.batch_tuning.max_candidate_batch,
    }
    effective_overrides = {**training_config.overrides, **budget.overrides}
    eval_protocol = {
        "schema": "coco_post_eval.v1",
        "task": training_config.task,
        "split": training_config.coco_post_eval.split,
        "imgsz": training_config.coco_post_eval.imgsz,
        "enabled": training_config.coco_post_eval.enabled,
        "profiles": sorted(training_config.coco_post_eval.profiles),
        "save_json": training_config.coco_post_eval.save_json,
        "plots": training_config.coco_post_eval.plots,
        "conf": training_config.coco_post_eval.conf,
        "iou": training_config.coco_post_eval.iou,
        "profile_val": budget.val,
        "profile_quick_val": budget.quick_val,
        "effective_save_json": bool(effective_overrides.get("save_json", False)),
    }
    return RunProtocolVersion(
        model=model,
        dataset_version=context.dataset_version,
        dataset_manifest_sha256=dataset_manifest,
        subset_manifest_sha256=subset_manifest,
        imgsz=training_config.imgsz,
        epochs=effective_epochs,
        seed=effective_seed,
        batch_policy=batch_policy,
        batch_policy_hash=stable_identity_hash(batch_policy),
        ultralytics_version=ultralytics_version or installed_ultralytics_version(),
        eval_protocol=eval_protocol,
        eval_protocol_hash=stable_identity_hash(eval_protocol),
        code_version=code_version or current_code_version(),
        profile=profile,
    )


def current_code_version(root: Path | str = ".") -> str:
    """Return the git commit plus a dirty marker, with a package fallback."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return f"{commit}+dirty" if dirty else commit
    except (OSError, subprocess.SubprocessError):
        try:
            return f"package:{importlib.metadata.version('yolo-agent')}"
        except importlib.metadata.PackageNotFoundError:
            return "package:unknown"


def installed_ultralytics_version() -> str:
    try:
        return importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


__all__ = [
    "RUN_PROTOCOL_SCHEMA_VERSION",
    "RunProtocolVersion",
    "build_run_protocol_version",
    "current_code_version",
    "installed_ultralytics_version",
]
