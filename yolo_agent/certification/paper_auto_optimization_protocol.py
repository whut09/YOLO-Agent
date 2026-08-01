"""Matched candidate/control protocol checks for paper acceptance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperProtocolIdentity,
)


class PaperProtocolMatch(BaseModel):
    """Strict comparison result for one matched candidate/control pair."""

    model_config = ConfigDict(extra="forbid")

    matched: bool
    protocol_hash: str | None = None
    mismatched_fields: dict[str, tuple[Any, Any]] = Field(default_factory=dict)


def build_paper_protocol_identity(
    *,
    data_yaml: Path,
    protocol_hash: str,
    objective_hash: str,
    epochs: int,
    seed: int,
    ultralytics_version: str | None = None,
) -> PaperProtocolIdentity:
    """Build one fixed-640 protocol shared by candidate and control."""
    dataset_hash = hash_files(data_yaml.parent)
    return PaperProtocolIdentity(
        dataset_manifest_hash=dataset_hash,
        subset_manifest_hash=dataset_hash,
        seed=seed,
        epochs=epochs,
        batch_policy_hash=hash_payload({"batch": 4, "device": "single_gpu"}),
        ultralytics_version=(
            ultralytics_version or importlib.metadata.version("ultralytics")
        ),
        eval_protocol_hash=hash_payload(
            {
                "protocol": "mini-coco-post-eval",
                "imgsz": 640,
                "conf": 0.001,
                "iou": 0.7,
            }
        ),
        objective_hash=objective_hash,
        protocol_hash=protocol_hash,
    )


def compare_paper_protocols(
    control: PaperProtocolIdentity | None,
    candidate: PaperProtocolIdentity | None,
) -> PaperProtocolMatch:
    """Fail closed unless every fairness field and hash is identical."""
    if control is None or candidate is None:
        return PaperProtocolMatch(
            matched=False,
            mismatched_fields={
                "protocol_identity": (
                    "present" if control is not None else "missing",
                    "present" if candidate is not None else "missing",
                )
            },
        )
    mismatches: dict[str, tuple[Any, Any]] = {}
    control_payload = control.model_dump(mode="json")
    candidate_payload = candidate.model_dump(mode="json")
    for field in sorted(control_payload):
        if control_payload[field] != candidate_payload[field]:
            mismatches[field] = (control_payload[field], candidate_payload[field])
    return PaperProtocolMatch(
        matched=not mismatches,
        protocol_hash=control.protocol_hash if not mismatches else None,
        mismatched_fields=mismatches,
    )


def hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def hash_files(root: Path) -> str:
    values = [
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return hash_payload({"files": values})


__all__ = [
    "PaperProtocolMatch",
    "build_paper_protocol_identity",
    "compare_paper_protocols",
    "hash_files",
    "hash_payload",
]
