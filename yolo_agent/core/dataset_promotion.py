"""Dataset promotion policy for active-learning reviewed labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_serializer

from yolo_agent.agents.active_learning import LabelingManifest
from yolo_agent.core.artifact_manifest import sha256_file
from yolo_agent.core.dataset_versioning import DatasetVersionManifest


ReviewStatus = Literal["accepted", "rejected", "needs_fix"]
PromotionDecision = Literal["promoted", "rejected", "needs_more_review"]


class ReviewedLabelSample(BaseModel):
    """One human-reviewed active-learning sample."""

    image_path: Path
    status: ReviewStatus = "accepted"
    labels_path: Path | None = None
    notes: list[str] = Field(default_factory=list)

    @field_serializer("image_path", "labels_path")
    def serialize_path(self, value: Path | None) -> str | None:
        """Serialize review paths portably."""
        return value.as_posix() if value is not None else None


class ReviewedLabels(BaseModel):
    """Reviewed labels returned by a labeling workflow."""

    dataset_version: str
    next_dataset_version: str | None = None
    samples: list[ReviewedLabelSample] = Field(default_factory=list)


class DatasetPromotionPolicyConfig(BaseModel):
    """Thresholds for deciding whether a dataset version can be promoted."""

    min_reviewed_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    max_rejected_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    allow_empty_manifest: bool = False


class DatasetPromotionResult(BaseModel):
    """Auditable decision for active-learning dataset promotion."""

    decision: PromotionDecision
    dataset_version: str
    next_dataset_version: str
    required_samples: int
    reviewed_samples: int = 0
    accepted_samples: int = 0
    rejected_samples: int = 0
    needs_fix_samples: int = 0
    reviewed_ratio: float = 0.0
    rejected_ratio: float = 0.0
    parent_manifest_sha256: str | None = None
    promoted: bool = False
    reasons: list[str] = Field(default_factory=list)


class DatasetPromotionPolicy:
    """Decide whether reviewed active-learning samples can create dataset_vNext."""

    def __init__(self, config: DatasetPromotionPolicyConfig | None = None) -> None:
        self.config = config or DatasetPromotionPolicyConfig()

    def evaluate(
        self,
        labeling_manifest: LabelingManifest,
        reviewed_labels: ReviewedLabels | None,
        parent_manifest: DatasetVersionManifest | None,
        parent_manifest_path: Path | str | None = None,
    ) -> DatasetPromotionResult:
        """Evaluate promotion readiness from manifest, reviews, and parent dataset manifest."""
        required = len(labeling_manifest.samples)
        parent_sha = sha256_file(parent_manifest_path) if parent_manifest_path is not None and Path(parent_manifest_path).is_file() else None
        reasons: list[str] = []
        if parent_manifest is None:
            reasons.append("Missing parent dataset manifest.")
            return _result(
                decision="rejected",
                manifest=labeling_manifest,
                required=required,
                parent_sha=parent_sha,
                reasons=reasons,
            )
        if parent_manifest.version != labeling_manifest.dataset_version:
            reasons.append(
                f"Parent manifest version {parent_manifest.version} does not match labeling manifest {labeling_manifest.dataset_version}."
            )
            return _result(
                decision="rejected",
                manifest=labeling_manifest,
                required=required,
                parent_sha=parent_sha,
                reasons=reasons,
            )
        if required == 0 and not self.config.allow_empty_manifest:
            reasons.append("Labeling manifest contains no samples to promote.")
            return _result(
                decision="rejected",
                manifest=labeling_manifest,
                required=required,
                parent_sha=parent_sha,
                reasons=reasons,
            )
        if reviewed_labels is None:
            reasons.append("Missing reviewed_labels; waiting for label review.")
            return _result(
                decision="needs_more_review",
                manifest=labeling_manifest,
                required=required,
                parent_sha=parent_sha,
                reasons=reasons,
            )
        if reviewed_labels.dataset_version != labeling_manifest.dataset_version:
            reasons.append(
                f"Reviewed labels version {reviewed_labels.dataset_version} does not match {labeling_manifest.dataset_version}."
            )
            return _result(
                decision="rejected",
                manifest=labeling_manifest,
                required=required,
                parent_sha=parent_sha,
                reasons=reasons,
            )
        if reviewed_labels.next_dataset_version and reviewed_labels.next_dataset_version != labeling_manifest.next_dataset_version:
            reasons.append(
                f"Reviewed labels next version {reviewed_labels.next_dataset_version} does not match {labeling_manifest.next_dataset_version}."
            )
            return _result(
                decision="rejected",
                manifest=labeling_manifest,
                required=required,
                parent_sha=parent_sha,
                reasons=reasons,
            )

        manifest_paths = {sample.image_path.as_posix() for sample in labeling_manifest.samples}
        reviewed_by_path = {
            sample.image_path.as_posix(): sample
            for sample in reviewed_labels.samples
            if sample.image_path.as_posix() in manifest_paths
        }
        reviewed = len(reviewed_by_path)
        accepted = sum(sample.status == "accepted" for sample in reviewed_by_path.values())
        rejected = sum(sample.status == "rejected" for sample in reviewed_by_path.values())
        needs_fix = sum(sample.status == "needs_fix" for sample in reviewed_by_path.values())
        reviewed_ratio = reviewed / required if required else 1.0
        rejected_ratio = rejected / reviewed if reviewed else 0.0

        if reviewed_ratio < self.config.min_reviewed_ratio:
            reasons.append(
                f"Reviewed ratio {reviewed_ratio:.3g} is below required {self.config.min_reviewed_ratio:.3g}."
            )
        if needs_fix:
            reasons.append(f"{needs_fix} reviewed samples still need label fixes.")
        if reasons:
            return DatasetPromotionResult(
                decision="needs_more_review",
                dataset_version=labeling_manifest.dataset_version,
                next_dataset_version=labeling_manifest.next_dataset_version,
                required_samples=required,
                reviewed_samples=reviewed,
                accepted_samples=accepted,
                rejected_samples=rejected,
                needs_fix_samples=needs_fix,
                reviewed_ratio=reviewed_ratio,
                rejected_ratio=rejected_ratio,
                parent_manifest_sha256=parent_sha,
                promoted=False,
                reasons=reasons,
            )
        if rejected_ratio > self.config.max_rejected_ratio:
            reasons.append(
                f"Rejected ratio {rejected_ratio:.3g} exceeds allowed {self.config.max_rejected_ratio:.3g}."
            )
            decision: PromotionDecision = "rejected"
        else:
            reasons.append("All promotion gates passed.")
            decision = "promoted"

        return DatasetPromotionResult(
            decision=decision,
            dataset_version=labeling_manifest.dataset_version,
            next_dataset_version=labeling_manifest.next_dataset_version,
            required_samples=required,
            reviewed_samples=reviewed,
            accepted_samples=accepted,
            rejected_samples=rejected,
            needs_fix_samples=needs_fix,
            reviewed_ratio=reviewed_ratio,
            rejected_ratio=rejected_ratio,
            parent_manifest_sha256=parent_sha,
            promoted=decision == "promoted",
            reasons=reasons,
        )


def load_reviewed_labels(path: Path | str) -> ReviewedLabels:
    """Load reviewed labels from JSON or YAML."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) if input_path.suffix.lower() in {".yaml", ".yml"} else json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Reviewed labels must contain a mapping: {input_path}")
    return ReviewedLabels.model_validate(data)


def _result(
    decision: PromotionDecision,
    manifest: LabelingManifest,
    required: int,
    parent_sha: str | None,
    reasons: list[str],
) -> DatasetPromotionResult:
    return DatasetPromotionResult(
        decision=decision,
        dataset_version=manifest.dataset_version,
        next_dataset_version=manifest.next_dataset_version,
        required_samples=required,
        parent_manifest_sha256=parent_sha,
        promoted=decision == "promoted",
        reasons=reasons,
    )
