"""Materialize active assignment pilots only from matched shadow evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    ASSIGNMENT_SPECS,
    AssignmentActivationGate,
)
from yolo_agent.components.adapters.assigners import yolo26_assignment
from yolo_agent.recipes.schemas import AtomicRecipe


class AssignmentActivePilotDecision(BaseModel):
    """Auditable boundary between shadow observation and active assignment."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    execution_class: Literal["blocked", "active_pilot_candidate"]
    component_id: str
    shadow_recipe_id: str
    active_recipe: AtomicRecipe | None = None
    shadow_evidence_path: Path | None = None
    shadow_evidence_sha256: str | None = None
    matched_protocol_hash: str | None = None
    blocked_by: list[str] = Field(default_factory=list)


class AssignmentActivePilotMaterializer:
    """Create an active AtomicRecipe only after all deterministic gates pass."""

    def materialize(
        self,
        *,
        shadow_recipe: AtomicRecipe,
        shadow_evidence_path: Path | str,
        candidate_protocol_hash: str,
        control_protocol_hash: str,
        matched_control_available: bool,
        imgsz: int = 640,
        minimum_shadow_batches: int = 1,
        maximum_conflict_rate: float = 1.0,
    ) -> AssignmentActivePilotDecision:
        component_id = (
            shadow_recipe.component_ids[0]
            if len(shadow_recipe.component_ids) == 1
            else "unknown"
        )
        spec = ASSIGNMENT_SPECS.get(component_id)
        blocked: list[str] = []
        if spec is None:
            blocked.append("assignment_component_not_supported")
        if imgsz != 640 or shadow_recipe.train_overrides.get("imgsz", 640) != 640:
            blocked.append("fixed_imgsz_640_required")
        if not matched_control_available:
            blocked.append("matched_control_missing")
        if not candidate_protocol_hash or candidate_protocol_hash != control_protocol_hash:
            blocked.append("matched_control_protocol_mismatch")
        if "matched_control" not in shadow_recipe.compatibility_requirements:
            blocked.append("recipe_matched_control_requirement_missing")
        if spec is not None:
            mode = shadow_recipe.train_overrides.get(spec.changed_variable)
            if mode != "shadow":
                blocked.append("assignment_recipe_is_not_shadow")
            activation = AssignmentActivationGate().evaluate(
                shadow_evidence_path,
                component_id=component_id,
                method=spec.method,
                assignment_path="one_to_many",
                minimum_batches=minimum_shadow_batches,
                maximum_conflict_rate=maximum_conflict_rate,
                protocol_hash=candidate_protocol_hash,
                runtime_plugin_sha256=yolo26_assignment._sha256(
                    Path(yolo26_assignment.__file__)
                ),
                changed_variable=spec.changed_variable,
            )
            blocked.extend(activation.blocked_by)
        else:
            activation = None
        blocked = list(dict.fromkeys(blocked))
        if blocked or spec is None or activation is None:
            return AssignmentActivePilotDecision(
                allowed=False,
                execution_class="blocked",
                component_id=component_id,
                shadow_recipe_id=shadow_recipe.recipe_id,
                shadow_evidence_path=Path(shadow_evidence_path),
                blocked_by=blocked,
            )
        evidence_path = Path(activation.evidence_path or shadow_evidence_path).resolve()
        active_overrides = dict(shadow_recipe.train_overrides)
        active_overrides.update(
            {
                spec.changed_variable: "active",
                "assignment.shadow_evidence_path": str(evidence_path),
                "assignment.shadow_payload_hash": activation.runtime_payload_hash,
                "imgsz": 640,
                "profile": "pilot",
            }
        )
        active = shadow_recipe.model_copy(
            update={
                "recipe_id": shadow_recipe.recipe_id.removesuffix("_shadow")
                + "_active_pilot",
                "train_overrides": active_overrides,
                "compatibility_requirements": list(
                    dict.fromkeys(
                        [
                            *shadow_recipe.compatibility_requirements,
                            "shadow_evidence_passed",
                            "same_protocol_hash",
                        ]
                    )
                ),
                "promotion_requirements": list(
                    dict.fromkeys(
                        [
                            *shadow_recipe.promotion_requirements,
                            "matched_control",
                            "ASHA_only",
                        ]
                    )
                ),
                "maturity": "smoke_passed",
            }
        )
        return AssignmentActivePilotDecision(
            allowed=True,
            execution_class="active_pilot_candidate",
            component_id=component_id,
            shadow_recipe_id=shadow_recipe.recipe_id,
            active_recipe=active,
            shadow_evidence_path=evidence_path,
            shadow_evidence_sha256=activation.evidence_sha256,
            matched_protocol_hash=candidate_protocol_hash,
        )


__all__ = [
    "AssignmentActivePilotDecision",
    "AssignmentActivePilotMaterializer",
]
