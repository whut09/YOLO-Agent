"""Stable identity for code that controls optimization certification."""

from __future__ import annotations

import hashlib
from pathlib import Path


CERTIFIED_MODULES = (
    "adapters/ultralytics/coco_post_eval.py",
    "agents/asha_scheduler.py",
    "core/matched_baseline.py",
    "core/paired_experiment.py",
    "core/pilot_evidence.py",
    "certification/runner.py",
)


def certification_code_hash() -> str:
    """Hash the executable evidence and scheduling implementation."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in CERTIFIED_MODULES:
        path = package_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["CERTIFIED_MODULES", "certification_code_hash"]
