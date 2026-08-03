"""Matched paired-delta attribution for coupled recipe ablations."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import stdev
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


CoupledArm = Literal["A", "B", "A+B"]
CoupledEffectKind = Literal["single", "combined_total", "interaction"]
ContributionConfidence = Literal["possible", "confirmed"]


class CoupledArmObservation(BaseModel):
    """One current-node paired delta against an exact matched control."""

    model_config = ConfigDict(extra="forbid")

    arm: CoupledArm
    node_id: str
    matched_control_node_id: str
    seed: int
    protocol_hash: str
    metric_deltas: dict[str, float] = Field(min_length=1)
    paired_result_verified: bool = False
    evidence_role: str = "current_observation"
    inheritance_depth: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _identity_complete(self) -> "CoupledArmObservation":
        if not self.node_id.strip() or not self.matched_control_node_id.strip():
            raise ValueError("coupled observation requires node and matched control IDs")
        if not self.protocol_hash.strip():
            raise ValueError("coupled observation requires protocol_hash")
        return self


class CoupledContributionEffect(BaseModel):
    effect_id: str
    effect_kind: CoupledEffectKind
    component_ids: list[str]
    metric_name: str
    seed_count: int = Field(ge=1)
    mean_delta: float
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    confidence: ContributionConfidence
    direction: Literal["positive", "negative", "neutral", "uncertain"]
    reason: str


class CoupledContributionReport(BaseModel):
    schema_version: str = "coupled_contribution_report.v1"
    recipe_id: str
    component_a: str
    component_b: str
    effects: list[CoupledContributionEffect] = Field(default_factory=list)
    complete_seeds: list[int] = Field(default_factory=list)
    incomplete_seeds: dict[int, list[str]] = Field(default_factory=dict)
    rejected_observations: dict[str, str] = Field(default_factory=dict)

    def to_yaml(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CoupledContributionReport":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(payload)


class CoupledContributionAnalyzer:
    """Attribute A, B, combined, and interaction effects from matched evidence."""

    def analyze(
        self,
        *,
        recipe_id: str,
        component_a: str,
        component_b: str,
        observations: list[CoupledArmObservation],
        confirmed_seed_count: int = 3,
    ) -> CoupledContributionReport:
        accepted: list[CoupledArmObservation] = []
        rejected: dict[str, str] = {}
        for index, observation in enumerate(observations):
            reason = _rejection_reason(observation)
            if reason is None:
                accepted.append(observation)
            else:
                rejected[f"{observation.node_id}:{index}"] = reason

        by_seed: dict[int, dict[CoupledArm, CoupledArmObservation]] = defaultdict(dict)
        duplicate_seeds: dict[int, set[CoupledArm]] = defaultdict(set)
        for observation in accepted:
            if observation.arm in by_seed[observation.seed]:
                duplicate_seeds[observation.seed].add(observation.arm)
                continue
            by_seed[observation.seed][observation.arm] = observation

        incomplete: dict[int, list[str]] = {}
        complete: dict[int, dict[CoupledArm, CoupledArmObservation]] = {}
        required_arms: tuple[CoupledArm, ...] = ("A", "B", "A+B")
        for seed, arms in sorted(by_seed.items()):
            reasons: list[str] = []
            missing = [arm for arm in required_arms if arm not in arms]
            if missing:
                reasons.append("missing_arms:" + ",".join(missing))
            if duplicate_seeds.get(seed):
                reasons.append(
                    "duplicate_arms:" + ",".join(sorted(duplicate_seeds[seed]))
                )
            controls = {item.matched_control_node_id for item in arms.values()}
            if len(controls) > 1:
                reasons.append("matched_control_mismatch")
            protocols = {item.protocol_hash for item in arms.values()}
            if len(protocols) > 1:
                reasons.append("protocol_hash_mismatch")
            common_metrics = set.intersection(
                *(set(item.metric_deltas) for item in arms.values())
            ) if arms else set()
            if not common_metrics:
                reasons.append("no_common_metrics")
            if reasons:
                incomplete[seed] = reasons
            else:
                complete[seed] = arms

        effects: list[CoupledContributionEffect] = []
        metrics = sorted(
            set.intersection(
                *(
                    set(item.metric_deltas)
                    for arms in complete.values()
                    for item in arms.values()
                )
            )
        ) if complete else []
        definitions: tuple[
            tuple[str, CoupledEffectKind, list[str]], ...
        ] = (
            ("A", "single", [component_a]),
            ("B", "single", [component_b]),
            ("A+B", "combined_total", [component_a, component_b]),
            ("interaction", "interaction", [component_a, component_b]),
        )
        for metric in metrics:
            values_by_effect = _effect_values(complete, metric)
            for effect_name, effect_kind, component_ids in definitions:
                effects.append(
                    _summarize_effect(
                        recipe_id=recipe_id,
                        effect_name=effect_name,
                        effect_kind=effect_kind,
                        component_ids=component_ids,
                        metric_name=metric,
                        values=values_by_effect[effect_name],
                        confirmed_seed_count=confirmed_seed_count,
                    )
                )
        return CoupledContributionReport(
            recipe_id=recipe_id,
            component_a=component_a,
            component_b=component_b,
            effects=effects,
            complete_seeds=sorted(complete),
            incomplete_seeds=incomplete,
            rejected_observations=rejected,
        )


def _rejection_reason(observation: CoupledArmObservation) -> str | None:
    if not observation.paired_result_verified:
        return "paired_result_not_verified"
    if observation.evidence_role != "current_observation":
        return "current_observation_required"
    if observation.inheritance_depth != 0:
        return "inherited_evidence_forbidden"
    return None


def _effect_values(
    complete: dict[int, dict[CoupledArm, CoupledArmObservation]],
    metric_name: str,
) -> dict[str, list[float]]:
    result = {"A": [], "B": [], "A+B": [], "interaction": []}
    for arms in complete.values():
        delta_a = arms["A"].metric_deltas[metric_name]
        delta_b = arms["B"].metric_deltas[metric_name]
        delta_combined = arms["A+B"].metric_deltas[metric_name]
        result["A"].append(delta_a)
        result["B"].append(delta_b)
        result["A+B"].append(delta_combined)
        result["interaction"].append(delta_combined - delta_a - delta_b)
    return result


def _summarize_effect(
    *,
    recipe_id: str,
    effect_name: str,
    effect_kind: CoupledEffectKind,
    component_ids: list[str],
    metric_name: str,
    values: list[float],
    confirmed_seed_count: int,
) -> CoupledContributionEffect:
    mean = sum(values) / len(values)
    interval = _cross_seed_interval(values)
    excludes_zero = interval is not None and (
        interval[0] > 0.0 or interval[1] < 0.0
    )
    confirmed = len(values) >= confirmed_seed_count and excludes_zero
    if confirmed:
        reason = "multi_seed_confidence_interval_excludes_zero"
        direction: Literal["positive", "negative", "neutral", "uncertain"] = (
            "positive" if mean > 0.0 else "negative"
        )
    elif len(values) < confirmed_seed_count:
        reason = f"insufficient_repeated_seeds:{len(values)}/{confirmed_seed_count}"
        direction = "neutral" if mean == 0.0 else "uncertain"
    else:
        reason = "cross_seed_confidence_interval_includes_zero"
        direction = "neutral" if mean == 0.0 else "uncertain"
    return CoupledContributionEffect(
        effect_id=f"{recipe_id}:{effect_name}:{metric_name}",
        effect_kind=effect_kind,
        component_ids=component_ids,
        metric_name=metric_name,
        seed_count=len(values),
        mean_delta=mean,
        confidence_interval_low=interval[0] if interval is not None else None,
        confidence_interval_high=interval[1] if interval is not None else None,
        confidence="confirmed" if confirmed else "possible",
        direction=direction,
        reason=reason,
    )


def _cross_seed_interval(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    standard_error = stdev(values) / (len(values) ** 0.5)
    critical = _student_t_critical(len(values) - 1)
    return mean - critical * standard_error, mean + critical * standard_error


def _student_t_critical(degrees_of_freedom: int) -> float:
    values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    for upper in sorted(values):
        if degrees_of_freedom <= upper:
            return values[upper]
    return 1.96


__all__ = [
    "ContributionConfidence",
    "CoupledArm",
    "CoupledArmObservation",
    "CoupledContributionEffect",
    "CoupledContributionAnalyzer",
    "CoupledContributionReport",
    "CoupledEffectKind",
]
