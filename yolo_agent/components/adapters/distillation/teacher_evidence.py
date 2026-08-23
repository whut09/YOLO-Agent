"""Fail-closed teacher/student checkpoint identity and protocol evidence.

This module intentionally does not load a detector for certification.  It
only resolves local files and reads lightweight checkpoint metadata on CPU.
The training plugin remains responsible for loading the frozen teacher and
checking the model graph at runtime.
"""

from __future__ import annotations

from hashlib import sha256
import json
import pickle
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.adapters.distillation.protocol import (
    resolve_local_checkpoint,
    sha256_file,
)


CheckpointRole = Literal["teacher", "student"]
EvidenceDisposition = Literal["runtime_ready", "evidence_recovery", "blocked_runtime"]


class CheckpointMetadata(BaseModel):
    """Metadata that can be verified without constructing a model."""

    model_config = ConfigDict(extra="forbid")

    architecture: str | None = None
    dataset_hash: str | None = None
    split: str | None = None
    imgsz: int | None = None
    source: Literal["checkpoint", "sidecar", "filename", "missing"] = "missing"


class CheckpointIdentity(BaseModel):
    """Resolved identity used in payloads and execution fingerprints."""

    model_config = ConfigDict(extra="forbid")

    role: CheckpointRole
    path: str
    sha256: str | None = None
    architecture: str | None = None
    dataset_hash: str | None = None
    split: str | None = None
    imgsz: int | None = None
    metadata_source: str = "missing"
    metadata_verified: bool = False
    test_only: bool = False

    @property
    def identity_hash(self) -> str:
        payload = self.model_dump(mode="json")
        return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class CheckpointResolution(BaseModel):
    """Result of resolving one checkpoint, including all admission blockers."""

    model_config = ConfigDict(extra="forbid")

    identity: CheckpointIdentity
    disposition: EvidenceDisposition
    reason_codes: list[str] = Field(default_factory=list)
    recovery_action: str

    @property
    def ready(self) -> bool:
        return self.disposition == "runtime_ready" and not self.reason_codes


def resolve_checkpoint_identity(
    value: Path | str,
    *,
    role: CheckpointRole,
    workspace: Path | str | None = None,
    expected_sha256: str | None = None,
    expected_architecture: str | None = None,
    expected_dataset_hash: str | None = None,
    expected_split: str | None = None,
    expected_imgsz: int | None = None,
    require_metadata: bool = True,
) -> CheckpointResolution:
    """Resolve and validate a local checkpoint without downloading it.

    Missing metadata is tolerated only by callers explicitly performing a
    CPU test fixture.  It is represented as ``test_only`` and never grants
    runtime readiness.
    """

    requested = Path(value).expanduser()
    try:
        path = resolve_local_checkpoint(requested, workspace=workspace)
    except FileNotFoundError:
        identity = CheckpointIdentity(role=role, path=str(requested), test_only=True)
        return CheckpointResolution(
            identity=identity,
            disposition="evidence_recovery",
            reason_codes=[f"{role}_checkpoint_missing:{value}"],
            recovery_action=f"provide a local frozen {role} checkpoint at {value}",
        )

    digest = sha256_file(path)
    metadata = _read_metadata(path)
    architecture = metadata.architecture or _architecture_from_name(path)
    identity = CheckpointIdentity(
        role=role,
        path=str(path.resolve()),
        sha256=digest,
        architecture=architecture,
        dataset_hash=metadata.dataset_hash,
        split=metadata.split,
        imgsz=metadata.imgsz,
        metadata_source=metadata.source,
        metadata_verified=metadata.source in {"checkpoint", "sidecar"},
        test_only=metadata.source == "filename",
    )
    reasons: list[str] = []
    if expected_sha256 and digest != expected_sha256:
        reasons.append(f"{role}_checkpoint_sha256_mismatch")
    if role == "student" and architecture != "yolo26n":
        reasons.append("student_architecture_not_yolo26n")
    if role == "teacher" and not architecture:
        reasons.append("teacher_architecture_missing")
    if expected_architecture and architecture and architecture != expected_architecture:
        reasons.append(f"{role}_architecture_mismatch")
    if require_metadata and not identity.metadata_verified:
        reasons.append(f"{role}_checkpoint_metadata_missing")
    if expected_dataset_hash and metadata.dataset_hash and metadata.dataset_hash != expected_dataset_hash:
        reasons.append(f"{role}_dataset_hash_mismatch")
    if expected_dataset_hash and not metadata.dataset_hash and require_metadata:
        reasons.append(f"{role}_dataset_hash_missing")
    if expected_split and metadata.split and metadata.split != expected_split:
        reasons.append(f"{role}_split_mismatch")
    if expected_split and not metadata.split and require_metadata:
        reasons.append(f"{role}_split_missing")
    if expected_imgsz is not None and metadata.imgsz is not None and metadata.imgsz != expected_imgsz:
        reasons.append(f"{role}_imgsz_mismatch")
    if expected_imgsz is not None and metadata.imgsz is None and require_metadata:
        reasons.append(f"{role}_imgsz_missing")
    if reasons:
        disposition: EvidenceDisposition = "evidence_recovery"
    else:
        disposition = "runtime_ready"
    return CheckpointResolution(
        identity=identity,
        disposition=disposition,
        reason_codes=list(dict.fromkeys(reasons)),
        recovery_action=_recovery_action(role, reasons),
    )


def resolve_teacher_checkpoint(
    value: Path | str,
    *,
    workspace: Path | str | None = None,
    expected_sha256: str | None = None,
    expected_dataset_hash: str | None = None,
    expected_split: str = "train",
    expected_imgsz: int = 640,
    require_metadata: bool = True,
) -> CheckpointResolution:
    """Resolve a teacher with the protocol required by YOLO26 distillation."""

    expected_architecture = _architecture_from_name(Path(value))
    return resolve_checkpoint_identity(
        value,
        role="teacher",
        workspace=workspace,
        expected_sha256=expected_sha256,
        expected_architecture=expected_architecture,
        expected_dataset_hash=expected_dataset_hash,
        expected_split=expected_split,
        expected_imgsz=expected_imgsz,
        require_metadata=require_metadata,
    )


def resolve_student_checkpoint(
    value: Path | str,
    *,
    workspace: Path | str | None = None,
    expected_sha256: str | None = None,
    expected_dataset_hash: str | None = None,
    expected_split: str = "train",
    expected_imgsz: int = 640,
    require_metadata: bool = True,
) -> CheckpointResolution:
    """Resolve the fixed YOLO26n student checkpoint."""

    return resolve_checkpoint_identity(
        value,
        role="student",
        workspace=workspace,
        expected_sha256=expected_sha256,
        expected_architecture="yolo26n",
        expected_dataset_hash=expected_dataset_hash,
        expected_split=expected_split,
        expected_imgsz=expected_imgsz,
        require_metadata=require_metadata,
    )


def _read_metadata(path: Path) -> CheckpointMetadata:
    for candidate in (
        path.with_suffix(path.suffix + ".metadata.json"),
        path.with_suffix(path.suffix + ".metadata.yaml"),
        path.with_suffix(".metadata.json"),
        path.with_suffix(".metadata.yaml"),
    ):
        if not candidate.is_file():
            continue
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8")) if candidate.suffix == ".json" else yaml.safe_load(candidate.read_text(encoding="utf-8"))
            return _metadata_from_mapping(raw, source="sidecar")
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
    try:
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError):
        raw = None
    metadata = _metadata_from_mapping(raw, source="checkpoint")
    if metadata.source != "missing":
        return metadata
    return CheckpointMetadata(
        architecture=_architecture_from_name(path),
        source="filename",
    )


def _metadata_from_mapping(raw: Any, *, source: Literal["checkpoint", "sidecar"]) -> CheckpointMetadata:
    if not isinstance(raw, dict):
        return CheckpointMetadata()
    candidates = [raw]
    for key in ("metadata", "meta", "args", "train_args"):
        value = raw.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    model = raw.get("model")
    if isinstance(model, dict):
        candidates.append(model)
    architecture = _first(candidates, "architecture", "arch", "model_name", "model_family")
    dataset_hash = _first(candidates, "dataset_hash", "data_hash", "dataset_manifest_hash")
    split = _first(candidates, "split", "dataset_split")
    imgsz = _first(candidates, "imgsz", "image_size", "image_size_train")
    if architecture is None and dataset_hash is None and split is None and imgsz is None:
        return CheckpointMetadata()
    return CheckpointMetadata(
        architecture=str(architecture) if architecture is not None else None,
        dataset_hash=str(dataset_hash) if dataset_hash is not None else None,
        split=str(split) if split is not None else None,
        imgsz=int(imgsz) if imgsz is not None else None,
        source=source,
    )


def _first(mappings: list[dict[str, Any]], *keys: str) -> Any:
    for mapping in mappings:
        for key in keys:
            if key in mapping and mapping[key] not in (None, ""):
                return mapping[key]
    return None


def _architecture_from_name(path: Path) -> str | None:
    stem = path.stem.lower()
    for scale in ("n", "s", "m", "l", "x"):
        if stem.endswith(scale) and "yolo26" in stem:
            return f"yolo26{scale}"
    return None


def _recovery_action(role: CheckpointRole, reasons: list[str]) -> str:
    if any("missing" in reason for reason in reasons):
        return f"recover {role} checkpoint metadata and protocol evidence, then rerun readiness"
    if any("mismatch" in reason for reason in reasons):
        return f"replace or rebind the {role} checkpoint so SHA-256 and dataset protocol match"
    return f"provide verifiable frozen {role} checkpoint evidence before ASHA admission"


__all__ = [
    "CheckpointIdentity",
    "CheckpointMetadata",
    "CheckpointResolution",
    "EvidenceDisposition",
    "resolve_checkpoint_identity",
    "resolve_student_checkpoint",
    "resolve_teacher_checkpoint",
]
