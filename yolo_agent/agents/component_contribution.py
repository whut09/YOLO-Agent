"""Component ablation matrix and contribution evaluation."""

from __future__ import annotations

from pathlib import Path
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
            contributions.append(
                ComponentContribution(
                    component=node.added_component,
                    parent_id=node.parent_id,
                    candidate_id=candidate_id,
                    deltas={name: sum(values) / len(values) for name, values in by_metric.items()},
                    matched_control_hashes=hashes,
                    paired_seed_count=len(seeds),
                    confidence="confirmed" if len(seeds) >= 3 else "possible",
                )
            )
        return ComponentContributionReport(contributions=contributions, missing_metrics=missing)


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
