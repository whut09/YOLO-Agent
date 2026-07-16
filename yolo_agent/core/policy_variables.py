"""Classify fixed protocol values separately from experimental changes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PolicyVariableClassification(BaseModel):
    """Effective command overrides and the subset that changes the experiment."""

    fixed_variables: dict[str, Any] = Field(default_factory=dict)
    effective_overrides: dict[str, Any] = Field(default_factory=dict)
    changed_variables: dict[str, Any] = Field(default_factory=dict)


def classify_policy_variables(
    *,
    components: list[str],
    train_overrides: dict[str, Any],
    action_domain: str,
    action_id: str | None,
    scale: str,
    declared_fixed_variables: dict[str, Any] | None = None,
    baseline_protocol: dict[str, Any] | None = None,
) -> PolicyVariableClassification:
    """Separate protocol constraints from true ablation variables.

    A value remains an effective command override even when it equals the
    baseline protocol. Such a value is fixed, not an experimental change.
    """
    declared = dict(declared_fixed_variables or {})
    baseline = {key: value for key, value in (baseline_protocol or {}).items() if value is not None}
    # The measured baseline protocol is authoritative over proposal claims.
    fixed = {**declared, **baseline}
    effective = dict(train_overrides)
    changed: dict[str, Any] = {}

    component_groups = {
        "bbox_loss": "loss.bbox.",
        "head_component": "head.",
        "assigner": "assigner.",
        "neck_component": "neck.",
        "augmentation_policy": "augmentation.",
        "optimizer": "optimizer.",
    }
    for variable, prefix in component_groups.items():
        values = [component for component in components if component.startswith(prefix)]
        if values:
            changed[variable] = values

    tracked_overrides = {
        "imgsz",
        "augmentation_policy",
        "postprocess",
        "data_action",
        "label_action",
        "training_action",
        "postprocess_action",
        "augmentation_action",
        "evidence_action",
    }
    for key in tracked_overrides:
        if key not in effective:
            continue
        value = effective[key]
        if key in fixed and _same_value(value, fixed[key]):
            continue
        changed[key] = value

    if action_domain != "model" and action_id is not None:
        action_key = f"{action_domain}_action"
        if action_key not in fixed or not _same_value(action_id, fixed[action_key]):
            changed.setdefault(action_key, action_id)
    if scale not in {"", "baseline", "n"}:
        if "model_scale" not in fixed or not _same_value(scale, fixed["model_scale"]):
            changed["model_scale"] = scale

    return PolicyVariableClassification(
        fixed_variables=fixed,
        effective_overrides=effective,
        changed_variables=changed,
    )


def _same_value(left: Any, right: Any) -> bool:
    """Compare protocol scalars without treating numeric spellings as changes."""
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right
