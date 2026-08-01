"""Structured, offline-only evidence for paper method adaptation."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MethodEvidenceSource = Literal[
    "title",
    "summary",
    "note",
    "harness_hint",
    "category",
    "official_code_metadata",
    "cached_readme",
    "cached_config",
]
MethodEvidenceConfidence = Literal["low", "medium", "high"]
MethodEvidenceField = Literal[
    "method_family",
    "canonical_mechanism",
    "insertion_point",
    "changed_variable",
    "detector_family",
    "component_type",
    "training_only",
    "inference_changed",
    "compatibility_constraint",
    "required_runtime_hook",
]
PaperComponentType = Literal[
    "loss",
    "head",
    "neck",
    "data",
    "assigner",
    "inference",
    "distillation",
    "attention",
    "other",
]


class PaperMethodEvidenceObservation(BaseModel):
    """One source-grounded method field extracted from local text or metadata."""

    model_config = ConfigDict(extra="forbid")

    field_name: MethodEvidenceField
    value: str | bool
    source: MethodEvidenceSource
    source_location: str
    confidence: MethodEvidenceConfidence
    authorizes_method_profile: bool = False
    evidence_level: Literal["paper_prior"] = "paper_prior"

    @model_validator(mode="after")
    def validate_authorization(self) -> "PaperMethodEvidenceObservation":
        if not self.source_location.strip():
            raise ValueError("paper method evidence requires source_location")
        if isinstance(self.value, str) and not self.value.strip():
            raise ValueError("paper method evidence requires a non-empty value")
        if self.source in {"title", "harness_hint", "category"}:
            if self.authorizes_method_profile:
                raise ValueError(
                    f"{self.source} evidence cannot authorize a method profile"
                )
        if self.confidence == "low" and self.authorizes_method_profile:
            raise ValueError("low-confidence evidence cannot authorize a method profile")
        return self


class PaperMethodEvidenceProfile(BaseModel):
    """Aggregated method boundary inferred exclusively from permitted offline inputs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_method_evidence.v1"
    paper_id: str
    method_families: list[str] = Field(default_factory=list)
    canonical_mechanisms: list[str] = Field(default_factory=list)
    insertion_points: list[str] = Field(default_factory=list)
    changed_variables: list[str] = Field(default_factory=list)
    detector_families: list[str] = Field(default_factory=list)
    component_types: list[PaperComponentType] = Field(default_factory=list)
    training_only: bool | None = None
    inference_changed: bool | None = None
    compatibility_constraints: list[str] = Field(default_factory=list)
    required_runtime_hooks: list[str] = Field(default_factory=list)
    observations: list[PaperMethodEvidenceObservation] = Field(default_factory=list)
    authorizes_method_profile: bool = False
    authorization_reasons: list[str] = Field(default_factory=list)
    evidence_hash: str = ""

    @model_validator(mode="after")
    def validate_aggregation(self) -> "PaperMethodEvidenceProfile":
        if not self.paper_id.strip():
            raise ValueError("paper method evidence profile requires paper_id")
        observed = {
            (item.field_name, str(item.value).lower() if isinstance(item.value, bool) else item.value)
            for item in self.observations
        }
        aggregate_fields: tuple[tuple[MethodEvidenceField, list[str]], ...] = (
            ("method_family", self.method_families),
            ("canonical_mechanism", self.canonical_mechanisms),
            ("insertion_point", self.insertion_points),
            ("changed_variable", self.changed_variables),
            ("detector_family", self.detector_families),
            ("component_type", list(self.component_types)),
            ("compatibility_constraint", self.compatibility_constraints),
            ("required_runtime_hook", self.required_runtime_hooks),
        )
        for field_name, values in aggregate_fields:
            missing = [value for value in values if (field_name, value) not in observed]
            if missing:
                raise ValueError(
                    f"aggregated {field_name} values lack observations: {missing}"
                )
        for field_name, value in (
            ("training_only", self.training_only),
            ("inference_changed", self.inference_changed),
        ):
            if value is not None and (field_name, str(value).lower()) not in observed:
                raise ValueError(f"aggregated {field_name} lacks an observation")
        if self.authorizes_method_profile and not any(
            item.authorizes_method_profile for item in self.observations
        ):
            raise ValueError("profile authorization requires authorizing evidence")
        return self

    def with_hash(self) -> "PaperMethodEvidenceProfile":
        payload = self.model_dump(mode="json", exclude={"evidence_hash"})
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self.model_copy(update={"evidence_hash": digest})


__all__ = [
    "MethodEvidenceConfidence",
    "MethodEvidenceField",
    "MethodEvidenceSource",
    "PaperComponentType",
    "PaperMethodEvidenceObservation",
    "PaperMethodEvidenceProfile",
]
