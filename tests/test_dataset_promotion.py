"""Dataset promotion policy tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.active_learning import LabelingManifest, MinedSample
from yolo_agent.core.dataset_promotion import (
    DatasetPromotionPolicy,
    ReviewedLabelSample,
    ReviewedLabels,
)
from yolo_agent.core.dataset_versioning import DatasetFileRecord, DatasetVersionManifest


def _manifest() -> LabelingManifest:
    return LabelingManifest(
        target="label_studio",
        dataset_version="dataset-v1",
        next_dataset_version="dataset-v2",
        samples=[
            MinedSample(image_path=Path("unlabeled/a.jpg"), score=1.0, reasons=["low_confidence"]),
            MinedSample(image_path=Path("unlabeled/b.jpg"), score=0.8, reasons=["high_entropy"]),
        ],
    )


def _parent() -> DatasetVersionManifest:
    return DatasetVersionManifest(
        version="dataset-v1",
        source_root=Path("dataset"),
        files=[DatasetFileRecord(path="images/train/img1.jpg", sha256="abc", size_bytes=3)],
    )


def test_dataset_promotion_needs_review_when_reviews_are_missing() -> None:
    """Promotion should wait for reviewed labels instead of pretending the next dataset is ready."""
    result = DatasetPromotionPolicy().evaluate(_manifest(), reviewed_labels=None, parent_manifest=_parent())

    assert result.decision == "needs_more_review"
    assert result.promoted is False
    assert result.required_samples == 2
    assert "Missing reviewed_labels" in result.reasons[0]


def test_dataset_promotion_promotes_when_all_reviews_pass() -> None:
    """All reviewed and accepted samples should promote dataset_vNext."""
    reviewed = ReviewedLabels(
        dataset_version="dataset-v1",
        next_dataset_version="dataset-v2",
        samples=[
            ReviewedLabelSample(image_path=Path("unlabeled/a.jpg"), status="accepted"),
            ReviewedLabelSample(image_path=Path("unlabeled/b.jpg"), status="accepted"),
        ],
    )

    result = DatasetPromotionPolicy().evaluate(_manifest(), reviewed, _parent())

    assert result.decision == "promoted"
    assert result.promoted is True
    assert result.reviewed_ratio == 1.0
    assert result.accepted_samples == 2


def test_dataset_promotion_rejects_high_rejected_ratio() -> None:
    """Too many rejected reviewed samples should block promotion."""
    reviewed = ReviewedLabels(
        dataset_version="dataset-v1",
        next_dataset_version="dataset-v2",
        samples=[
            ReviewedLabelSample(image_path=Path("unlabeled/a.jpg"), status="accepted"),
            ReviewedLabelSample(image_path=Path("unlabeled/b.jpg"), status="rejected"),
        ],
    )

    result = DatasetPromotionPolicy().evaluate(_manifest(), reviewed, _parent())

    assert result.decision == "rejected"
    assert result.promoted is False
    assert result.rejected_ratio == 0.5
    assert "Rejected ratio" in result.reasons[0]


def test_dataset_promotion_rejects_manifest_version_mismatch() -> None:
    """Promotion must be tied to the same parent dataset manifest version."""
    parent = DatasetVersionManifest(version="dataset-v0", source_root=Path("dataset"))

    result = DatasetPromotionPolicy().evaluate(_manifest(), reviewed_labels=None, parent_manifest=parent)

    assert result.decision == "rejected"
    assert "does not match" in result.reasons[0]
