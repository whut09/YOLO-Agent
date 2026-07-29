"""Stable identities for implementation work rather than paper records."""

from __future__ import annotations

import hashlib
import json

from yolo_agent.research.component_aliases import normalize_component_id


def implementation_fingerprint(
    *,
    component_id: str,
    insertion_point: str,
    required_runtime_hook: str | None,
    detector_family: str = "yolo26",
) -> str:
    """Deduplicate the same engineering task proposed by multiple papers."""
    payload = {
        "component_id": normalize_component_id(component_id),
        "insertion_point": normalize_component_id(insertion_point),
        "required_runtime_hook": normalize_component_id(required_runtime_hook or "none"),
        "detector_family": normalize_component_id(detector_family),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def component_family(component_id: str, category: str) -> str:
    """Return a broad implementation family used only for cooldown."""
    normalized_category = normalize_component_id(category)
    if normalized_category in {"feature_pyramid", "neck", "attention"}:
        return "multi_scale_features"
    if normalized_category in {"detection_head", "positive_sample_selection"}:
        return "detection_head"
    if normalized_category in {"assigner", "matching"}:
        return "assignment"
    if normalized_category in {"slicing", "tta", "nms"}:
        return "inference_policy"
    if normalized_category:
        return normalized_category
    return normalize_component_id(component_id).split("_", 1)[0] or "unknown"


__all__ = ["component_family", "implementation_fingerprint"]
