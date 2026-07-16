"""Strict schemas and validation for LLM-extracted paper components."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yolo_agent.research.schemas import ComponentCategory, ComponentTaxonomy, PaperRecord


UnknownBool = bool | Literal["unknown"]
CategoryOrUnknown = ComponentCategory | Literal["unknown"]


class SourceLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    location: str

    @field_validator("paper_id", "location")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source paper_id and location must not be empty")
        return value.strip()


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    paper_id: str
    source_location: str
    evidence_level: Literal["paper_claim"]

    @field_validator("claim", "paper_id", "source_location")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim provenance fields must not be empty")
        return value.strip()


class ExtractedComponent(BaseModel):
    """A non-executable component description extracted from a paper."""

    model_config = ConfigDict(extra="forbid")
    component_id: str
    name: str
    component_category: CategoryOrUnknown = "unknown"
    insertion_point: str = "unknown"
    required_inputs: list[str] = Field(default_factory=lambda: ["unknown"])
    produced_outputs: list[str] = Field(default_factory=lambda: ["unknown"])
    claimed_effects: list[ExtractedClaim]
    target_error_types: list[str] = Field(default_factory=lambda: ["unknown"])
    coupling_dependencies: list[str] = Field(default_factory=lambda: ["unknown"])
    incompatible_components: list[str] = Field(default_factory=lambda: ["unknown"])
    training_only: UnknownBool = "unknown"
    inference_only: UnknownBool = "unknown"
    implementation_notes: list[str] = Field(default_factory=lambda: ["unknown"])
    evidence_level: Literal["paper_claim"]
    uncertainties: list[str] = Field(default_factory=lambda: ["unknown"])
    source_locations: list[SourceLocation]

    @field_validator("component_id", "name", "insertion_point")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("component identity and insertion_point must not be empty")
        return value.strip()

    @field_validator("required_inputs", "produced_outputs", "target_error_types", "coupling_dependencies", "incompatible_components", "implementation_notes", "uncertainties")
    @classmethod
    def _unknown_when_empty(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        return normalized or ["unknown"]

    @model_validator(mode="after")
    def _validate_contract(self) -> "ExtractedComponent":
        if self.training_only is True and self.inference_only is True:
            raise ValueError("a component cannot be both training_only and inference_only")
        if not self.claimed_effects:
            raise ValueError("each extracted component requires at least one sourced claim")
        if not self.source_locations:
            raise ValueError("each extracted component requires source_locations")
        return self


class ComponentExtractionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "component_extraction.v1"
    extracted_components: list[ExtractedComponent] = Field(default_factory=list)


class ComponentExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["used", "skipped", "failed"]
    paper_id: str
    provider: str
    model: str
    prompt_sha256: str | None = None
    bundle: ComponentExtractionBundle | None = None
    warnings: list[str] = Field(default_factory=list)
    raw_text: str = ""

    @property
    def extracted_components(self) -> list[ExtractedComponent]:
        return self.bundle.extracted_components if self.bundle else []


class ComponentExtractor:
    """Parse and ground an LLM response against supplied paper inputs."""

    def parse(self, raw_text: str, *, paper: PaperRecord, taxonomy: ComponentTaxonomy) -> tuple[ComponentExtractionBundle | None, list[str]]:
        text = raw_text.strip()
        fence = chr(96) * 3
        if text.startswith(fence) and text.endswith(fence):
            text = text[len(fence): -len(fence)].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, [f"component_json_parse_failed:{exc}"]
        try:
            bundle = ComponentExtractionBundle.model_validate(payload)
        except ValueError as exc:
            return None, [f"component_schema_validation_failed:{exc}"]
        warnings: list[str] = []
        allowed_categories = set(taxonomy.categories)
        for component in bundle.extracted_components:
            if component.component_category != "unknown" and component.component_category not in allowed_categories:
                return None, [f"unknown_component_category:{component.component_category}"]
            for claim in component.claimed_effects:
                if claim.paper_id != paper.paper_id:
                    return None, [f"claim_wrong_paper_id:{component.component_id}:{claim.paper_id}"]
            for location in component.source_locations:
                if location.paper_id != paper.paper_id:
                    return None, [f"source_wrong_paper_id:{component.component_id}:{location.paper_id}"]
            if "unknown" in component.uncertainties:
                warnings.append(f"component_has_unknowns:{component.component_id}")
        return bundle, warnings


__all__ = ["ComponentExtractionBundle", "ComponentExtractionResult", "ComponentExtractor", "ExtractedClaim", "ExtractedComponent", "SourceLocation"]
