"""Component ablation matrix and contribution evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.candidate_generator import CandidateConfig


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
            contributions.append(
                ComponentContribution(
                    component=node.added_component,
                    parent_id=node.parent_id,
                    candidate_id=node.candidate_config.candidate_id,
                    deltas=_metric_deltas(parent_metrics, current_metrics),
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

