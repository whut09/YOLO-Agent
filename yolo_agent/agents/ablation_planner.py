"""Ablation planning for single-variable experiment comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.candidate_generator import CandidateConfig, CandidatePlan
from yolo_agent.core.yaml_io import YAMLModelMixin


AblationVariable = Literal[
    "model_scale",
    "neck_component",
    "head_component",
    "bbox_loss",
    "assigner",
    "imgsz",
    "augmentation_policy",
    "recipe_component",
]


class AblationNode(BaseModel):
    """A valid single-variable ablation candidate."""

    node_id: str
    candidate_config: CandidateConfig
    parent_id: str
    changed_variables: dict[AblationVariable, Any]


class InvalidAblationCandidate(BaseModel):
    """Candidate rejected by ablation constraints."""

    candidate_id: str
    reason: str
    changed_variables: dict[AblationVariable, Any] = Field(default_factory=dict)


class AblationPlan(BaseModel, YAMLModelMixin):
    """A scientifically constrained ablation plan."""

    baseline_id: str
    nodes: list[AblationNode] = Field(default_factory=list)
    invalid_candidates: list[InvalidAblationCandidate] = Field(default_factory=list)


class AblationPlanner:
    """Create single-variable ablation comparisons from candidates."""

    def plan(self, candidates: list[CandidateConfig]) -> AblationPlan:
        """Build an ablation plan from candidate configs."""
        baseline = _find_baseline(candidates)
        if baseline is None:
            raise ValueError("Ablation planning requires a baseline candidate.")

        nodes: list[AblationNode] = []
        invalid: list[InvalidAblationCandidate] = []
        for candidate in candidates:
            if candidate.candidate_id == baseline.candidate_id:
                continue

            changed_variables = _changed_variables(baseline, candidate)
            if not changed_variables:
                invalid.append(
                    InvalidAblationCandidate(
                        candidate_id=candidate.candidate_id,
                        reason="Candidate does not change any tracked ablation variable.",
                    )
                )
                continue

            if len(changed_variables) > 1:
                invalid.append(
                    InvalidAblationCandidate(
                        candidate_id=candidate.candidate_id,
                        reason="Candidate changes multiple primary variables and must be split before ablation.",
                        changed_variables=changed_variables,
                    )
                )
                continue

            nodes.append(
                AblationNode(
                    node_id=f"ablate_{candidate.candidate_id}",
                    candidate_config=candidate,
                    parent_id=baseline.candidate_id,
                    changed_variables=changed_variables,
                )
            )

        return AblationPlan(
            baseline_id=baseline.candidate_id,
            nodes=nodes,
            invalid_candidates=invalid,
        )


def create_ablation_plan(plan_path: Path | str, out_path: Path | str) -> AblationPlan:
    """Load a candidate plan, create an ablation plan, and write it."""
    candidate_plan = CandidatePlan.from_yaml(plan_path)
    ablation_plan = AblationPlanner().plan(candidate_plan.candidates)
    ablation_plan.to_yaml(out_path)
    return ablation_plan


def _find_baseline(candidates: list[CandidateConfig]) -> CandidateConfig | None:
    for candidate in candidates:
        if "baseline" in candidate.candidate_id:
            return candidate
    for candidate in candidates:
        if not candidate.components and not candidate.train_overrides:
            return candidate
    return None


def _changed_variables(
    baseline: CandidateConfig,
    candidate: CandidateConfig,
) -> dict[AblationVariable, Any]:
    changed: dict[AblationVariable, Any] = {}

    if candidate.scale != baseline.scale:
        changed["model_scale"] = {"from": baseline.scale, "to": candidate.scale}

    _compare_component_group(changed, "neck_component", baseline, candidate, ("neck.",))
    _compare_component_group(changed, "head_component", baseline, candidate, ("head.",))
    _compare_component_group(changed, "bbox_loss", baseline, candidate, ("loss.bbox.",))
    _compare_component_group(changed, "assigner", baseline, candidate, ("assigner.",))
    _compare_component_group(changed, "augmentation_policy", baseline, candidate, ("augmentation.",))

    baseline_imgsz = baseline.train_overrides.get("imgsz")
    candidate_imgsz = candidate.train_overrides.get("imgsz")
    if candidate_imgsz != baseline_imgsz and candidate_imgsz is not None:
        changed["imgsz"] = {"from": baseline_imgsz, "to": candidate_imgsz}

    baseline_aug = baseline.train_overrides.get("augmentation_policy")
    candidate_aug = candidate.train_overrides.get("augmentation_policy")
    if candidate_aug != baseline_aug and candidate_aug is not None:
        changed["augmentation_policy"] = {"from": baseline_aug, "to": candidate_aug}

    return changed


def _compare_component_group(
    changed: dict[AblationVariable, Any],
    variable: AblationVariable,
    baseline: CandidateConfig,
    candidate: CandidateConfig,
    prefixes: tuple[str, ...],
) -> None:
    baseline_values = _components_with_prefixes(baseline.components, prefixes)
    candidate_values = _components_with_prefixes(candidate.components, prefixes)
    if candidate_values != baseline_values:
        changed[variable] = {"from": baseline_values, "to": candidate_values}


def _components_with_prefixes(components: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [component for component in components if component.startswith(prefixes)]
