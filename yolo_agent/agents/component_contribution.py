"""Component ablation matrix and contribution evaluation."""

from __future__ import annotations

from pathlib import Path
from statistics import stdev
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.candidate_generator import CandidateConfig

if TYPE_CHECKING:
    from yolo_agent.core.experiment_graph import Evidence


class AblationMatrixNode(BaseModel):
    """One node in an automatic component ablation matrix."""

    node_id: str
    candidate_config: CandidateConfig
    added_component: str | None = None
    parent_id: str | None = None
    component_set: list[str] = Field(default_factory=list)


class AblationMatrix(BaseModel):
    """Baseline plus cumulative component combinations."""

    baseline_id: str
    nodes: list[AblationMatrixNode] = Field(default_factory=list)

    def to_yaml(self, path: Path | str) -> None:
        """Serialize ablation matrix YAML."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.model_dump(mode="json"), file, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AblationMatrix":
        """Load ablation matrix YAML."""
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Ablation matrix YAML must contain a mapping: {input_path}")
        return cls.model_validate(data)


class ComponentContribution(BaseModel):
    """Measured contribution of one component relative to its parent node."""

    component: str
    parent_id: str
    candidate_id: str
    deltas: dict[str, float] = Field(default_factory=dict)
    matched_control_hashes: list[str] = Field(default_factory=list)
    paired_seed_count: int = 0
    confidence: str = "possible"
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    confirmation_metric: str | None = None
    confidence_reason: str = "insufficient_repeated_seed_evidence"
    image_bootstrap_direction: str | None = None


class ComponentContributionReport(BaseModel):
    """Contribution report for an ablation matrix."""

    contributions: list[ComponentContribution] = Field(default_factory=list)
    missing_metrics: list[str] = Field(default_factory=list)


class ComponentContributionPlanner:
    """Create cumulative ablation matrices and evaluate component deltas."""

    def build_matrix(
        self,
        baseline: CandidateConfig,
        components: list[str],
    ) -> AblationMatrix:
        """Build baseline, +A, +A+B, +A+B+C cumulative matrix."""
        nodes = [
            AblationMatrixNode(
                node_id=baseline.candidate_id,
                candidate_config=baseline,
                component_set=list(baseline.components),
            )
        ]
        current_components = list(baseline.components)
        parent_id: str | None = baseline.candidate_id
        for component in components:
            if component not in current_components:
                current_components.append(component)
            candidate = baseline.model_copy(
                update={
                    "candidate_id": _matrix_candidate_id(baseline, current_components),
                    "components": list(current_components),
                    "expected_effect": [f"Evaluate cumulative contribution of {component}."],
                }
            )
            nodes.append(
                AblationMatrixNode(
                    node_id=candidate.candidate_id,
                    candidate_config=candidate,
                    added_component=component,
                    parent_id=parent_id,
                    component_set=list(current_components),
                )
            )
            parent_id = candidate.candidate_id
        return AblationMatrix(baseline_id=baseline.candidate_id, nodes=nodes)

    def evaluate(
        self,
        matrix: AblationMatrix,
        metrics_by_candidate: dict[str, dict[str, float | int]],
    ) -> ComponentContributionReport:
        """Compute metric deltas for each added component."""
        contributions: list[ComponentContribution] = []
        missing: list[str] = []
        for node in matrix.nodes:
            if node.added_component is None or node.parent_id is None:
                continue
            current_metrics = metrics_by_candidate.get(node.candidate_config.candidate_id)
            parent_metrics = metrics_by_candidate.get(node.parent_id)
            if current_metrics is None or parent_metrics is None:
                missing.append(node.candidate_config.candidate_id)
                continue
            if _is_inference_policy_metrics(current_metrics):
                # Slicing/TTA/threshold experiments are Pareto alternatives,
                # not evidence of contribution by a training component.
                continue
            contributions.append(
                ComponentContribution(
                    component=node.added_component,
                    parent_id=node.parent_id,
                    candidate_id=node.candidate_config.candidate_id,
                    deltas=_metric_deltas(parent_metrics, current_metrics),
                )
            )
        return ComponentContributionReport(contributions=contributions, missing_metrics=missing)

    def evaluate_evidence(
        self,
        matrix: AblationMatrix,
        evidence: "Evidence",
        *,
        protocol_hash: str,
        dataset_manifest_sha256: str,
        split: str,
        seed_by_candidate: dict[str, int | str],
    ) -> ComponentContributionReport:
        """Evaluate contribution only from repeated exact matched-control pairs."""
        from yolo_agent.core.evidence_selector import EvidenceSelector, select_metric_evidence
        from yolo_agent.core.matched_baseline import paired_metric_delta

        contributions: list[ComponentContribution] = []
        missing: list[str] = []
        for node in matrix.nodes:
            if node.added_component is None or node.parent_id is None:
                continue
            candidate_id = node.candidate_config.candidate_id
            selected = select_metric_evidence(
                evidence.metric_records,
                EvidenceSelector(
                    current_run_id=evidence.run_id,
                    current_run_only=True,
                    current_node_only=[node.node_id],
                    inherited_context=False,
                    baseline_reference=False,
                    same_protocol_hash=protocol_hash,
                    same_dataset_manifest=dataset_manifest_sha256,
                    same_split=split,
                    same_seed=seed_by_candidate.get(candidate_id),
                    candidate_id=candidate_id,
                    verified=True,
                ),
            ).records
            paired = [
                delta
                for record in selected
                for _, delta in [paired_metric_delta(record, evidence.metric_records)]
                if delta is not None
            ]
            if not paired:
                missing.append(candidate_id)
                continue
            by_metric: dict[str, list[float]] = {}
            for delta in paired:
                by_metric.setdefault(delta.metric_name, []).append(delta.paired_delta)
            seeds = {delta.match_key.seed for delta in paired}
            hashes = sorted({delta.match_key_hash for delta in paired})
            confirmation_metric, interval = _confirmation_interval(paired)
            confirmed = len(seeds) >= 3 and interval is not None and interval[0] > 0.0
            bootstrap_direction = _current_bootstrap_direction(
                evidence.metric_records, candidate_id=candidate_id, node_id=node.node_id
            )
            contributions.append(
                ComponentContribution(
                    component=node.added_component,
                    parent_id=node.parent_id,
                    candidate_id=candidate_id,
                    deltas={name: sum(values) / len(values) for name, values in by_metric.items()},
                    matched_control_hashes=hashes,
                    paired_seed_count=len(seeds),
                    confidence="confirmed" if confirmed else "possible",
                    confidence_interval_low=interval[0] if interval is not None else None,
                    confidence_interval_high=interval[1] if interval is not None else None,
                    confirmation_metric=confirmation_metric,
                    confidence_reason=(
                        "three_or_more_paired_seeds_with_positive_confidence_interval"
                        if confirmed
                        else "paired_seed_confidence_interval_not_strictly_positive"
                        if len(seeds) >= 3
                        else f"insufficient_repeated_seeds:{len(seeds)}/3"
                    ),
                    image_bootstrap_direction=bootstrap_direction,
                )
            )
        return ComponentContributionReport(contributions=contributions, missing_metrics=missing)


def _confirmation_interval(paired: list[Any]) -> tuple[str | None, tuple[float, float] | None]:
    """Return a conservative 95% cross-seed CI for the primary accuracy effect."""
    preferred = ("map50_95", "coco_ap50_95", "ap_small", "map50")
    metric_name = next((name for name in preferred if any(item.metric_name == name for item in paired)), None)
    if metric_name is None:
        return None, None
    by_seed: dict[str, list[float]] = {}
    for item in paired:
        if item.metric_name == metric_name:
            by_seed.setdefault(item.match_key.seed, []).append(item.effect_delta)
    values = [sum(seed_values) / len(seed_values) for seed_values in by_seed.values()]
    if len(values) < 2:
        return metric_name, None
    mean = sum(values) / len(values)
    standard_error = stdev(values) / (len(values) ** 0.5)
    critical = _student_t_critical(len(values) - 1)
    return metric_name, (mean - critical * standard_error, mean + critical * standard_error)


def _student_t_critical(degrees_of_freedom: int) -> float:
    values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
              15: 2.131, 20: 2.086, 30: 2.042}
    for upper in sorted(values):
        if degrees_of_freedom <= upper:
            return values[upper]
    return 1.96


def _current_bootstrap_direction(
    records: list[Any], *, candidate_id: str, node_id: str,
) -> str | None:
    matches = [
        item for item in records
        if item.candidate_id == candidate_id and item.node_id == node_id
        and item.metric_name == "bootstrap/diagnostic_map50_direction"
        and item.validator == "paired_image_bootstrap"
        and item.evidence_role == "current_observation" and item.inheritance_depth == 0
    ]
    if not matches:
        return None
    return str(max(matches, key=lambda item: item.created_at).value)


def _matrix_candidate_id(baseline: CandidateConfig, components: list[str]) -> str:
    suffix = "_plus_" + "_".join(_short_component_name(component) for component in components)
    return f"{baseline.candidate_id}{suffix}".replace(".", "_").replace("-", "_")


def _short_component_name(component: str) -> str:
    return component.split(".")[-1]


def _metric_deltas(
    parent_metrics: dict[str, float | int],
    current_metrics: dict[str, float | int],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key, value in current_metrics.items():
        parent_value = parent_metrics.get(key)
        if isinstance(value, (int, float)) and isinstance(parent_value, (int, float)):
            deltas[key] = float(value) - float(parent_value)
    return deltas


def _is_inference_policy_metrics(metrics: dict[str, float | int]) -> bool:
    return bool(metrics.get("inference_policy_changed")) or any(
        str(name).startswith("sliced_") for name in metrics
    )
