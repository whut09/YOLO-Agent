"""Validated scoring and diversity policy for implementation planning."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PaperAdapterPlanningPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_adapter_planning_policy.v1"
    current_year: int = Field(default=2026, ge=2000, le=2100)
    freshness_window_years: int = Field(default=8, ge=1)
    family_cooldown_rounds: int = Field(default=2, ge=0)
    diagnosis_match_weight: float = 30.0
    ap_small_priority_weights: dict[str, float] = Field(default_factory=lambda: {
        "sampling.small_object": 25.0,
        "head.p2_small_object": 20.0,
        "distillation.yolo26_teacher_student": 15.0,
        "inference.sahi_slicing": 12.0,
    })
    compatibility_weight: float = 15.0
    runtime_hook_weight: float = 12.0
    mechanism_confidence_weight: float = 12.0
    paper_coverage_log_weight: float = 12.0
    paper_coverage_max_weight: float = 36.0
    official_code_weight: float = 6.0
    known_license_weight: float = 4.0
    unknown_license_penalty: float = -5.0
    freshness_max_weight: float = 5.0
    local_positive_max_weight: float = 20.0
    local_negative_max_penalty: float = -40.0
    implementation_cost_weights: dict[str, float] = Field(default_factory=lambda: {
        "low": 10.0, "medium": 3.0, "high": -8.0, "unknown": -3.0,
    })
    deployment_cost_weights: dict[str, float] = Field(default_factory=lambda: {
        "low": 4.0, "medium": 0.0, "high": -6.0, "unknown": -1.0,
    })

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PaperAdapterPlanningPolicy":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
        return cls.model_validate(payload)


__all__ = ["PaperAdapterPlanningPolicy"]
