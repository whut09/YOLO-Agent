"""Evidence-bound templates for narrowly justified coupled paper recipes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.research.method_profiles import PaperMethodProfile
from yolo_agent.recipes.schemas import CoupledRecipe


CoupledExecutionTrack = Literal["training", "inference"]
CouplingEvidenceKind = Literal["method_profile", "local_diagnosis"]
CoupledLibraryDecision = Literal["materialized", "rejected"]


class CoupledRecipeTemplate(BaseModel):
    """An allow-listed mechanism pair without an executable coupling reason."""

    model_config = ConfigDict(extra="forbid")

    template_id: str
    version: str = "v1.0.0"
    execution_track: CoupledExecutionTrack
    component_a_ids: list[str] = Field(min_length=1)
    component_b_ids: list[str] = Field(min_length=1)
    changed_variable_a: str
    changed_variable_b: str
    target_error_facts: list[dict[str, Any]] = Field(default_factory=list)
    target_metrics: list[str] = Field(default_factory=list)
    compatibility_requirements: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    guard_metrics_a: list[str] = Field(default_factory=list)
    guard_metrics_b: list[str] = Field(default_factory=list)
    attribution_excluded_metrics_a: list[str] = Field(default_factory=list)
    attribution_excluded_metrics_b: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_template(self) -> "CoupledRecipeTemplate":
        if not self.template_id.strip():
            raise ValueError("coupled recipe template requires template_id")
        if set(self.component_a_ids) & set(self.component_b_ids):
            raise ValueError("coupled template A and B component sets must be disjoint")
        if self.changed_variable_a == self.changed_variable_b:
            raise ValueError("coupled template requires two distinct changed variables")
        if not self.target_error_facts:
            raise ValueError("coupled template requires target error facts")
        return self

    def match(self, component_ids: list[str]) -> tuple[str, str] | None:
        """Return canonical A/B ordering only for one exact two-component pair."""
        unique = list(dict.fromkeys(component_ids))
        if len(unique) != 2:
            return None
        first = next((item for item in unique if item in self.component_a_ids), None)
        second = next((item for item in unique if item in self.component_b_ids), None)
        return (first, second) if first and second else None

    @property
    def template_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class CoupledRecipeTemplateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "evidence_bound_coupled_recipe_templates.v1"
    templates: list[CoupledRecipeTemplate] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_templates(self) -> "CoupledRecipeTemplateConfig":
        ids = [item.template_id for item in self.templates]
        if len(ids) != len(set(ids)):
            raise ValueError("coupled recipe template IDs must be unique")
        return self

    @classmethod
    def from_yaml(
        cls, path: Path | str = Path("configs/coupled_recipe_templates.yaml")
    ) -> "CoupledRecipeTemplateConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(payload)


class CouplingEvidence(BaseModel):
    """Explicit paper or local evidence authorizing one mechanism pair."""

    model_config = ConfigDict(extra="forbid")

    evidence_kind: CouplingEvidenceKind
    source_id: str
    component_ids: list[str]
    reason: str
    source_locations: list[str] = Field(min_length=1)
    paper_ids: list[str] = Field(default_factory=list)
    error_fact_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    verified: bool = False
    evidence_hash: str = ""

    @model_validator(mode="after")
    def _validate_evidence(self) -> "CouplingEvidence":
        self.component_ids = list(dict.fromkeys(self.component_ids))
        if len(self.component_ids) != 2:
            raise ValueError("coupling evidence must bind exactly two components")
        if not self.reason.strip() or self.reason.strip().lower() == "unknown":
            raise ValueError("coupling evidence requires an explicit reason")
        if self.evidence_kind == "method_profile" and not self.paper_ids:
            raise ValueError("MethodProfile coupling evidence requires paper_ids")
        if self.evidence_kind == "local_diagnosis" and not self.error_fact_ids:
            raise ValueError("local diagnosis coupling evidence requires error facts")
        expected = self.calculate_hash()
        if self.evidence_hash and self.evidence_hash != expected:
            raise ValueError("coupling evidence hash mismatch")
        self.evidence_hash = expected
        return self

    def calculate_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"evidence_hash"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


class LocalCouplingDiagnosis(BaseModel):
    """Local diagnosis assertion that two mechanisms address distinct causes."""

    model_config = ConfigDict(extra="forbid")

    diagnosis_id: str
    component_ids: list[str]
    reason: str
    error_fact_ids: list[str] = Field(min_length=1)
    source_location: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    verified: bool = False

    @model_validator(mode="after")
    def _validate_diagnosis(self) -> "LocalCouplingDiagnosis":
        self.component_ids = list(dict.fromkeys(self.component_ids))
        if len(self.component_ids) != 2:
            raise ValueError("local coupling diagnosis must bind exactly two components")
        if not self.reason.strip():
            raise ValueError("local coupling diagnosis requires a reason")
        if not self.source_location.strip():
            raise ValueError("local coupling diagnosis requires source_location")
        return self


class CoupledRecipeLibraryResult(BaseModel):
    """Auditable materialization result; rejection never returns a recipe."""

    model_config = ConfigDict(extra="forbid")

    decision: CoupledLibraryDecision
    template_id: str | None = None
    template_hash: str | None = None
    execution_track: CoupledExecutionTrack | None = None
    component_ids: list[str] = Field(default_factory=list)
    evidence_hash: str | None = None
    recipe: CoupledRecipe | None = None
    blocked_by: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_result(self) -> "CoupledRecipeLibraryResult":
        if self.decision == "materialized" and self.recipe is None:
            raise ValueError("materialized coupled library result requires recipe")
        if self.decision == "materialized" and not self.template_hash:
            raise ValueError("materialized coupled library result requires template_hash")
        if self.decision == "rejected" and not self.blocked_by:
            raise ValueError("rejected coupled library result requires blocked_by")
        return self


class EvidenceBoundCoupledRecipeLibrary:
    """Materialize only allow-listed, explicitly evidenced two-component recipes."""

    library_version = "evidence_bound_coupled_recipe_library.v1"

    def __init__(self, config: CoupledRecipeTemplateConfig | None = None) -> None:
        self.config = config or CoupledRecipeTemplateConfig.from_yaml()

    def materialize(
        self,
        *,
        component_ids: list[str],
        evidence: CouplingEvidence,
        template_id: str | None = None,
        recipe_id: str | None = None,
    ) -> CoupledRecipeLibraryResult:
        unique = list(dict.fromkeys(component_ids))
        blocked: list[str] = []
        if len(unique) != 2:
            blocked.append("exactly_two_components_required")
        if not evidence.verified:
            blocked.append("verified_coupling_evidence_required")
        if set(unique) != set(evidence.component_ids):
            blocked.append("coupling_evidence_component_mismatch")
        matches = [
            item
            for item in self.config.templates
            if item.match(unique) is not None
            and (template_id is None or item.template_id == template_id)
        ]
        if not matches:
            blocked.append("allowlisted_complementary_mechanism_pair_required")
        if len(matches) > 1:
            blocked.append("ambiguous_coupled_recipe_template")
        if blocked:
            return CoupledRecipeLibraryResult(
                decision="rejected",
                component_ids=unique,
                evidence_hash=evidence.evidence_hash,
                blocked_by=blocked,
            )
        template = matches[0]
        matched = template.match(unique)
        if matched is None:  # guarded above; retained for static safety
            raise RuntimeError("matched coupled template became unavailable")
        component_a, component_b = matched
        recipe = self._build_recipe(
            template,
            component_a=component_a,
            component_b=component_b,
            evidence=evidence,
            recipe_id=recipe_id,
        )
        return CoupledRecipeLibraryResult(
            decision="materialized",
            template_id=template.template_id,
            template_hash=template.template_hash,
            execution_track=template.execution_track,
            component_ids=[component_a, component_b],
            evidence_hash=evidence.evidence_hash,
            recipe=recipe,
        )

    def _build_recipe(
        self,
        template: CoupledRecipeTemplate,
        *,
        component_a: str,
        component_b: str,
        evidence: CouplingEvidence,
        recipe_id: str | None,
    ) -> CoupledRecipe:
        arm_a = {
            "name": "A",
            "components": [component_a],
            "changed_variables": {template.changed_variable_a: component_a},
            "guard_metrics": template.guard_metrics_a,
            "attribution_excluded_metrics": template.attribution_excluded_metrics_a,
        }
        arm_b = {
            "name": "B",
            "components": [component_b],
            "changed_variables": {template.changed_variable_b: component_b},
            "guard_metrics": template.guard_metrics_b,
            "attribution_excluded_metrics": template.attribution_excluded_metrics_b,
        }
        combined = {
            "name": "A+B",
            "components": [component_a, component_b],
            "changed_variables": {
                template.changed_variable_a: component_a,
                template.changed_variable_b: component_b,
            },
            "guard_metrics": sorted(
                set(template.guard_metrics_a) | set(template.guard_metrics_b)
            ),
            "attribution_excluded_metrics": sorted(
                set(template.attribution_excluded_metrics_a)
                & set(template.attribution_excluded_metrics_b)
            ),
        }
        stable_id = recipe_id or _recipe_id(
            template.template_id, component_a, component_b
        )
        source_ids = list(evidence.paper_ids) or [
            f"local_diagnosis:{evidence.source_id}"
        ]
        inference_actions = (
            [component_a, component_b]
            if template.execution_track == "inference"
            else []
        )
        data_actions = [
            item
            for item in (component_a, component_b)
            if template.execution_track == "training"
            and (item.startswith("sampling.") or item.startswith("augmentation."))
        ]
        fixed_variables: dict[str, Any] = {"imgsz": 640}
        if template.execution_track == "inference":
            fixed_variables.update(
                {"training_recipe": "unchanged", "checkpoint": "unchanged"}
            )
        return CoupledRecipe(
            recipe_id=stable_id,
            version=template.version,
            target_error_facts=list(template.target_error_facts),
            target_metrics=list(template.target_metrics),
            component_ids=[component_a, component_b],
            train_overrides={"imgsz": 640},
            data_actions=data_actions,
            inference_actions=inference_actions,
            fixed_variables=fixed_variables,
            primary_changed_variable=template.changed_variable_a,
            coupled_variables=[
                template.changed_variable_a,
                template.changed_variable_b,
            ],
            coupling_reason=evidence.reason,
            coupling_source_papers=source_ids,
            internal_ablation_plan=[
                {"name": "baseline", "components": [], "changed_variables": {}},
                arm_a,
                arm_b,
                combined,
            ],
            compatibility_requirements=list(template.compatibility_requirements),
            conflicts=[
                *template.conflicts,
                "automatic_component_bundle_forbidden",
            ],
            expected_effects={
                "evidence_bound_pair": True,
                "interaction_requires_ablation": True,
                "coupled_template_hash": template.template_hash,
            },
            evidence_prior=[
                {
                    "evidence_kind": evidence.evidence_kind,
                    "source_id": evidence.source_id,
                    "evidence_hash": evidence.evidence_hash,
                    "confidence": evidence.confidence,
                    "local_effect": False,
                }
            ],
            implementation_risk="high",
            training_cost={
                "training_changed": template.execution_track == "training"
            },
            inference_cost={
                "inference_policy_changed": template.execution_track == "inference"
            },
            stop_conditions=[
                "minimum_internal_ablation_incomplete",
                "matched_control_missing",
                "target_error_not_improved",
            ],
            promotion_requirements=[
                "baseline_A_B_A_plus_B_complete",
                "matched_control_per_arm",
                "verified_paired_delta",
            ],
            maturity="adapter_implemented",
        )


def coupling_evidence_from_method_profile(
    profile: PaperMethodProfile,
    component_ids: list[str],
) -> CouplingEvidence:
    """Read only an explicit MethodProfile coupling statement."""
    unique = list(dict.fromkeys(component_ids))
    if len(unique) != 2 or not set(unique).issubset(profile.canonical_component_ids):
        raise ValueError("MethodProfile does not bind the requested component pair")
    reason = _explicit_profile_reason(profile)
    if reason is None:
        raise ValueError("MethodProfile has no explicit coupling_reason")
    return CouplingEvidence(
        evidence_kind="method_profile",
        source_id=profile.profile_id,
        component_ids=unique,
        reason=reason,
        source_locations=list(profile.source_locations),
        paper_ids=[profile.paper_id],
        confidence=_profile_coupling_confidence(profile),
        verified=True,
    )


def coupling_evidence_from_diagnosis(
    diagnosis: LocalCouplingDiagnosis,
) -> CouplingEvidence:
    """Convert a verified local diagnosis without treating it as paper evidence."""
    if not diagnosis.verified:
        raise ValueError("local coupling diagnosis must be verified")
    return CouplingEvidence(
        evidence_kind="local_diagnosis",
        source_id=diagnosis.diagnosis_id,
        component_ids=list(diagnosis.component_ids),
        reason=diagnosis.reason,
        source_locations=[diagnosis.source_location],
        error_fact_ids=list(diagnosis.error_fact_ids),
        confidence=diagnosis.confidence,
        verified=True,
    )


def _explicit_profile_reason(profile: PaperMethodProfile) -> str | None:
    for source in (profile.paper_parameters, profile.protocol_constraints):
        value = source.get("coupling_reason")
        if isinstance(value, str) and value.strip() and value.strip().lower() != "unknown":
            return value.strip()
    return None


def _profile_coupling_confidence(profile: PaperMethodProfile) -> float:
    values = []
    for source in (profile.paper_parameters, profile.protocol_constraints):
        value = source.get("coupling_confidence")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return min(max(values[0], 0.0), 1.0) if values else 0.5


def _recipe_id(template_id: str, component_a: str, component_b: str) -> str:
    values = [template_id, component_a, component_b]
    return "coupled__" + "__".join(
        value.replace(".", "_").replace("-", "_") for value in values
    )


__all__ = [
    "CoupledExecutionTrack",
    "CoupledLibraryDecision",
    "CoupledRecipeTemplate",
    "CoupledRecipeTemplateConfig",
    "CoupledRecipeLibraryResult",
    "CouplingEvidence",
    "CouplingEvidenceKind",
    "LocalCouplingDiagnosis",
    "EvidenceBoundCoupledRecipeLibrary",
    "coupling_evidence_from_diagnosis",
    "coupling_evidence_from_method_profile",
]
