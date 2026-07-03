"""Learnable-style augmentation policy selection from data and error profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.resources import ResourcePaths
from yolo_agent.tools.dataset_stats import DatasetReport
from yolo_agent.utils import dedupe_list


class AugmentationPolicyAction(BaseModel):
    """Merged augmentation action sets."""

    enable: list[str] = Field(default_factory=list)
    reduce: list[str] = Field(default_factory=list)
    disable: list[str] = Field(default_factory=list)
    add: list[str] = Field(default_factory=list)
    set_params: dict[str, Any] = Field(default_factory=dict)


class AugmentationPolicyResult(BaseModel):
    """Selected augmentation policy for a dataset/error profile."""

    actions: AugmentationPolicyAction
    matched_rules: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class AugmentationPolicyRule(BaseModel):
    """Configurable rule that maps profile conditions to augmentation actions."""

    conditions: dict[str, Any] = Field(default_factory=dict)
    enable: list[str] = Field(default_factory=list)
    reduce: list[str] = Field(default_factory=list)
    disable: list[str] = Field(default_factory=list)
    add: list[str] = Field(default_factory=list)
    set_params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    risks: list[str] = Field(default_factory=list)


class AugmentationPolicyEngine:
    """Select augmentation policy from dataset and error profiles."""

    def __init__(self, rules: dict[str, AugmentationPolicyRule]) -> None:
        self.rules = rules

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "AugmentationPolicyEngine":
        """Load augmentation policy rules from YAML."""
        rule_path = Path(path) if path is not None else default_augmentation_policy_path()
        with rule_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Augmentation policy YAML must contain a mapping: {rule_path}")
        raw_rules = data.get("rules", {})
        if not isinstance(raw_rules, dict):
            raise ValueError("Augmentation policy YAML requires a 'rules' mapping.")
        return cls(
            {
                str(rule_id): AugmentationPolicyRule.model_validate(rule)
                for rule_id, rule in raw_rules.items()
                if isinstance(rule, dict)
            }
        )

    def recommend(
        self,
        dataset_report: DatasetReport,
        error_profile: list[DetectionErrorObservation] | None = None,
    ) -> AugmentationPolicyResult:
        """Recommend augmentation actions from current evidence."""
        observations = error_profile or []
        result = AugmentationPolicyResult(actions=AugmentationPolicyAction())

        for rule_id, rule in self.rules.items():
            if not _matches_rule(rule, dataset_report, observations):
                continue
            result.matched_rules.append(rule_id)
            result.actions.enable.extend(rule.enable)
            result.actions.reduce.extend(rule.reduce)
            result.actions.disable.extend(rule.disable)
            result.actions.add.extend(rule.add)
            result.actions.set_params.update(rule.set_params)
            if rule.rationale:
                result.rationale.append(rule.rationale)
            result.risks.extend(rule.risks)

        result.actions.enable = dedupe_list(result.actions.enable)
        result.actions.reduce = dedupe_list(result.actions.reduce)
        result.actions.disable = dedupe_list(result.actions.disable)
        result.actions.add = dedupe_list(result.actions.add)
        result.rationale = dedupe_list(result.rationale)
        result.risks = dedupe_list(result.risks)
        return result


def default_augmentation_policy_path() -> Path:
    """Return bundled augmentation policy rules."""
    return ResourcePaths.AUGMENTATION_POLICIES


def _matches_rule(
    rule: AugmentationPolicyRule,
    dataset_report: DatasetReport,
    observations: list[DetectionErrorObservation],
) -> bool:
    conditions = rule.conditions
    if "min_small_object_ratio" in conditions:
        small_ratio = dataset_report.object_size_ratio.get("small", 0.0)
        if small_ratio <= float(conditions["min_small_object_ratio"]):
            return False

    scenes = conditions.get("scenes")
    if isinstance(scenes, list) and dataset_report.scene not in {str(scene) for scene in scenes}:
        return False

    error_types = conditions.get("error_types")
    if isinstance(error_types, list):
        matched = [
            observation
            for observation in observations
            if observation.error_type in {str(error_type) for error_type in error_types}
        ]
        if not matched:
            return False
        min_error_count = conditions.get("min_error_count")
        if min_error_count is not None and sum(observation.count for observation in matched) < int(min_error_count):
            return False
        min_severity = conditions.get("min_severity")
        if min_severity is not None and not _has_min_severity(matched, str(min_severity)):
            return False

    health_problems = conditions.get("health_problems")
    if isinstance(health_problems, list):
        problems = set(dataset_report.dataset_health.problems)
        if not problems.intersection(str(problem) for problem in health_problems):
            return False

    return True


def _has_min_severity(observations: list[DetectionErrorObservation], severity: str) -> bool:
    order = {"low": 0, "medium": 1, "high": 2}
    threshold = order.get(severity, 0)
    return any(order[observation.severity] >= threshold for observation in observations)
