"""Evidence-bound templates for narrowly justified coupled paper recipes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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


__all__ = [
    "CoupledExecutionTrack",
    "CoupledLibraryDecision",
    "CoupledRecipeTemplate",
    "CoupledRecipeTemplateConfig",
    "CouplingEvidence",
    "CouplingEvidenceKind",
]
