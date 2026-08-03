"""Certified component tracks used by paper auto-optimization acceptance."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


PaperAcceptanceTrackId = Literal[
    "sampling",
    "auxiliary_loss",
    "distillation",
    "model_graph",
]


class PaperAcceptanceRecipe(BaseModel):
    """One atomic, diagnosis-bound mechanism in the acceptance cohort."""

    model_config = ConfigDict(extra="forbid")

    track_id: PaperAcceptanceTrackId
    recipe_id: str
    component_id: str
    component_family: str
    changed_variable: str
    adapter_options: dict[str, Any] = Field(default_factory=dict)
    primary_metric: str
    target_metrics: list[str] = Field(default_factory=list)
    target_error_facts: list[dict[str, Any]] = Field(min_length=1)
    max_map_regression: float = Field(default=0.01, ge=0.0)
    max_latency_regression: float = Field(default=0.05, ge=0.0)
    max_model_size_regression: float = Field(default=0.05, ge=0.0)


PAPER_ACCEPTANCE_RECIPES = (
    PaperAcceptanceRecipe(
        track_id="sampling",
        recipe_id="sampling.small_object",
        component_id="sampling.small_object",
        component_family="sampling",
        changed_variable="data.sampling_policy",
        adapter_options={
            "small_object_boost": 2.0,
            "class_balance": True,
            "rare_class_boost": 1.5,
            "fn_heavy_class_ids": [0],
            "target_class_ids": [0],
            "max_oversampling_ratio": 3.0,
        },
        primary_metric="ap_small",
        target_metrics=["ap_small", "per_class_ar/object"],
        target_error_facts=[
            {
                "fact_type": "false_negative_heavy_class",
                "subject": "object",
                "class_name": "object",
                "metric_name": "false_negative",
            }
        ],
    ),
    PaperAcceptanceRecipe(
        track_id="auxiliary_loss",
        recipe_id="loss.quality.correlation",
        component_id="loss.quality.correlation",
        component_family="auxiliary_loss",
        changed_variable="loss.correlation.weight",
        adapter_options={"loss.correlation.weight": 0.2},
        primary_metric="map50_95",
        target_metrics=["map50_95", "per_class_ap/object"],
        target_error_facts=[
            {
                "fact_type": "localization_heavy_class",
                "subject": "object",
                "class_name": "object",
                "metric_name": "localization_error",
            }
        ],
    ),
    PaperAcceptanceRecipe(
        track_id="distillation",
        recipe_id="distillation.yolo26_teacher_student",
        component_id="distillation.yolo26_teacher_student",
        component_family="distillation",
        changed_variable="loss.distillation",
        adapter_options={
            "teacher": "yolo26s.pt",
            "student": "yolo26n.pt",
            "imgsz": 640,
            "logits": True,
            "feature": True,
            "localization": True,
        },
        primary_metric="map50_95",
        target_metrics=["map50_95", "per_class_ar/object"],
        target_error_facts=[
            {
                "fact_type": "false_negative_heavy_class",
                "subject": "object",
                "class_name": "object",
                "metric_name": "false_negative",
            }
        ],
    ),
    PaperAcceptanceRecipe(
        track_id="model_graph",
        recipe_id="head.p2_small_object",
        component_id="head.p2_small_object",
        component_family="model_graph",
        changed_variable="model.p2_head",
        adapter_options={
            "imgsz": 640,
            "num_classes": 1,
            "audit_imgsz": 64,
        },
        primary_metric="ap_small",
        target_metrics=["ap_small", "per_class_ar/object"],
        target_error_facts=[
            {
                "fact_type": "false_negative_heavy_class",
                "subject": "object",
                "class_name": "object",
                "metric_name": "false_negative",
            }
        ],
        max_latency_regression=0.20,
        max_model_size_regression=0.20,
    ),
)


def acceptance_recipe(component_id: str) -> PaperAcceptanceRecipe:
    for recipe in PAPER_ACCEPTANCE_RECIPES:
        if recipe.component_id == component_id:
            return recipe
    raise KeyError(f"paper acceptance recipe is not defined: {component_id}")


__all__ = [
    "PAPER_ACCEPTANCE_RECIPES",
    "PaperAcceptanceRecipe",
    "PaperAcceptanceTrackId",
    "acceptance_recipe",
]
