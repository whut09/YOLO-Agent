"""Deterministic implementation-track classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from yolo_agent.agents.paper_adapter_planning.schemas import (
    AdapterImplementationEstimate,
    ImplementationTrack,
)
from yolo_agent.research.component_aliases import ResolvedComponentAlias, normalize_component_id
from yolo_agent.research.schemas import PaperRecord


class TrackClassification(BaseModel):
    track: ImplementationTrack
    reasons: list[str] = Field(default_factory=list)


def classify_implementation_track(
    paper: PaperRecord,
    mapping: ResolvedComponentAlias,
    estimate: AdapterImplementationEstimate,
) -> TrackClassification:
    if _is_separate_detector_track(paper):
        return TrackClassification(
            track="separate_detector_family",
            reasons=["detector_family_requires_separate_track"],
        )
    if paper.applicability == "incompatible" or mapping.yolo26_compatibility == "incompatible":
        return TrackClassification(
            track="incompatible",
            reasons=["yolo26_component_incompatible"],
        )
    if mapping.executable:
        return TrackClassification(
            track="ready_to_materialize",
            reasons=["verified_adapter_smoke_passed"],
        )
    if (
        estimate.requires_shadow_evaluation
        or mapping.category in {"assigner", "matching"}
    ) and mapping.implementation_status == "adapter_implemented":
        return TrackClassification(
            track="shadow_evaluation_queue",
            reasons=["adapter_requires_shadow_evidence_before_active_use"],
        )
    if mapping.implementation_status == "adapter_implemented":
        return TrackClassification(
            track="shadow_evaluation_queue",
            reasons=["implemented_adapter_requires_smoke_or_shadow_evidence"],
        )
    return TrackClassification(
        track="implementation_queue",
        reasons=[
            "implementation_request_required",
            f"catalog_applicability_is_prior:{paper.applicability}",
        ],
    )


def _is_separate_detector_track(paper: PaperRecord) -> bool:
    if paper.applicability == "separate_detector_family":
        return True
    text = " ".join(
        [
            paper.title,
            paper.detector_family or "",
            *paper.task_families,
            (paper.provenance.original_category or "") if paper.provenance else "",
        ]
    )
    normalized = normalize_component_id(text)
    return any(
        token in normalized
        for token in (
            "detr",
            "open_vocabulary",
            "open_world",
            "grounded_detection",
            "vision_language",
        )
    )


__all__ = ["TrackClassification", "classify_implementation_track"]
