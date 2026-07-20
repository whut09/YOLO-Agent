"""Core schemas for paper intelligence and research evidence.

Paper claims are deliberately separate from local experiment evidence. These
models describe research metadata only; they do not authorize an experiment or
turn a paper claim into a trusted metric.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from yolo_agent.core.yaml_io import YAMLModelMixin


RESEARCH_SCHEMA_VERSION = "research.v1"

EvidenceLevel = Literal[
    "paper_claim",
    "paper_prior",
    "official_code_available",
    "externally_reproduced",
    "locally_smoke_tested",
    "locally_pilot_reproduced",
    "locally_full_reproduced",
    "confirmed_multi_seed",
]

BenchmarkEvidenceLevel = EvidenceLevel

ComponentCategory = Literal[
    "backbone",
    "neck",
    "detection_head",
    "feature_pyramid",
    "attention",
    "convolution_block",
    "optimizer",
    "lr_schedule",
    "loss_schedule",
    "assigner",
    "matching",
    "positive_sample_selection",
    "bbox_regression_loss",
    "classification_loss",
    "quality_estimation",
    "distillation",
    "augmentation",
    "sampling",
    "label_quality",
    "active_learning",
    "threshold",
    "slicing",
    "tta",
    "calibration",
    "ensemble",
    "nms",
    "pretraining",
    "domain_adaptation",
]

Applicability = Literal[
    "direct_adapter_candidate",
    "recipe_idea_only",
    "separate_detector_family",
    "incompatible",
    "insufficient_information",
]


class ResearchSchema(BaseModel, YAMLModelMixin):
    """Common schema metadata shared by research records."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = RESEARCH_SCHEMA_VERSION


class PaperBenchmark(ResearchSchema):
    """A benchmark value reported by a paper or reproduction source."""

    dataset: str
    split: str = "val"
    model: str
    metric_name: str
    value: float
    imgsz: int | None = Field(default=None, ge=1)
    latency_ms: float | None = Field(default=None, ge=0.0)
    hardware: str | None = None
    training_epochs: int | None = Field(default=None, ge=0)
    source_location: str | None = None
    evidence_level: BenchmarkEvidenceLevel
    verified: bool = False

    @field_validator("dataset", "model", "metric_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset, model, and metric_name must not be empty")
        return value.strip()


class PaperComponentClaim(ResearchSchema):
    """A structured claim about one component from a paper."""

    component_id: str
    paper_id: str = "unknown"
    component_category: ComponentCategory | None = None
    claimed_effect: str
    evidence_level: Literal["paper_claim"]
    target_metrics: list[str] = Field(default_factory=list)
    target_error_types: list[str] = Field(default_factory=list)
    reported_delta: dict[str, float | str] = Field(default_factory=dict)
    baseline: str | None = None
    experiment_conditions: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("component_id", "claimed_effect")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("component_id and claimed_effect must not be empty")
        return value.strip()


class PaperProvenance(ResearchSchema):
    """Source and import history for a normalized paper record."""

    source_repository: str
    source_commit: str = "unknown"
    source_path: str
    source_record_hash: str
    imported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importer_version: str
    original_category: str | None = None
    original_applicability: Applicability | None = None
    original_harness_hints: list[str] = Field(default_factory=list)
    original_component_ids: list[str] = Field(default_factory=list)
    original_note_path: str | None = None
    abstract_source: Literal["abstract", "summary", "unknown"] = "unknown"
    history: list[dict[str, Any]] = Field(default_factory=list)


class PaperRecord(ResearchSchema):
    """Normalized metadata for one research paper."""

    paper_id: str
    doi: str | None = None
    title: str
    abstract: str = ""
    year: int = Field(ge=1900, le=2100)
    published_at: datetime | None = None
    updated_at: datetime | None = None
    authors: list[str] = Field(default_factory=list)
    task_families: list[str] = Field(default_factory=list)
    detector_family: str | None = None
    source_url: str | None = None
    paper_url: str | None = None
    official_code_url: str | None = None
    code_license: str | None = None
    framework: str | None = None
    datasets: list[str] = Field(default_factory=list)
    benchmarks: list[PaperBenchmark] = Field(default_factory=list)
    training_budget: dict[str, Any] = Field(default_factory=dict)
    claimed_effects: list[PaperComponentClaim] = Field(default_factory=list)
    component_ids: list[str] = Field(default_factory=list)
    component_categories: list[ComponentCategory] = Field(default_factory=list)
    applicability: Applicability = "insufficient_information"
    source: str = "manual"
    ingestion_version: str = "manual.v1"
    evidence_level: Literal["paper_claim", "paper_prior"] = "paper_claim"
    provenance: PaperProvenance | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("paper_id", "title", "source", "ingestion_version")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("paper_id, title, source, and ingestion_version must not be empty")
        return value.strip()


class ComponentTaxonomy(ResearchSchema):
    """Configured research component taxonomy."""

    categories: dict[ComponentCategory, list[str]] = Field(default_factory=dict)

    @field_validator("categories")
    @classmethod
    def _validate_categories(
        cls,
        value: dict[ComponentCategory, list[str]],
    ) -> dict[ComponentCategory, list[str]]:
        normalized: dict[ComponentCategory, list[str]] = {}
        for category, components in value.items():
            normalized[category] = [item.strip() for item in components if item.strip()]
        return normalized
