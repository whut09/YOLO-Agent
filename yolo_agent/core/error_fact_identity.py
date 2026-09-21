"""Prompt-18I-v2 §3: ErrorFact identity over a real DetectionErrorProfile.

Every fact derived from a real evaluation carries a deterministic identity:

    error_fact_id        stable id (profile-scoped, queryable)
    metric               which measured quantity
    value                the measured number
    slice                the slice it belongs to (global / class / scale / ...)
    evidence_path        the artifact the number came from
    calculation_version  the deterministic builder version

The records are derived ONLY from a real :class:`DetectionErrorProfile` —
there is no path that mints a fact from narrative.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.detection_error_profile import DetectionErrorProfile

#: Bump when fact extraction semantics change; identities embed it so
#: stale facts are detectable.
CALCULATION_VERSION = "error_facts.v2"


class ErrorFactIdentity(BaseModel):
    """One identified, provenance-backed error fact."""

    model_config = ConfigDict(extra="forbid")

    error_fact_id: str
    metric: str
    value: float
    slice: str
    evidence_path: str
    calculation_version: str = CALCULATION_VERSION
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> "ErrorFactIdentity":
        if not self.error_fact_id.strip():
            raise ValueError("error_fact_id must not be empty")
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if not self.slice.strip():
            raise ValueError("slice must not be empty")
        return self


def _evidence_path(profile: DetectionErrorProfile) -> str:
    return profile.gt_artifact or profile.profile_id


def _fact(
    profile: DetectionErrorProfile,
    metric: str,
    value: float | None,
    slice_name: str,
    **extra: Any,
) -> ErrorFactIdentity | None:
    """Mint one fact; a None value means the metric was not measured."""
    if value is None:
        return None
    return ErrorFactIdentity(
        error_fact_id=f"{profile.profile_id}:{slice_name}:{metric}",
        metric=metric,
        value=float(value),
        slice=slice_name,
        evidence_path=_evidence_path(profile),
        extra=extra,
    )


def error_fact_identities(profile: DetectionErrorProfile) -> list[ErrorFactIdentity]:
    """Extract every identified fact the profile actually measured.

    Facts exist only for values the profile carries; absent measurements
    (``None``) simply produce no fact, so downstream consumers can trust
    that every returned fact is real and reproducible.
    """
    facts: list[ErrorFactIdentity] = []

    g = profile.global_
    facts.extend(
        f
        for f in (
            _fact(profile, "map50", g.map50, "global"),
            _fact(profile, "map50_95", g.map50_95, "global"),
            _fact(profile, "precision", g.precision, "global"),
            _fact(profile, "recall", g.recall, "global"),
        )
        if f is not None
    )

    s = profile.scale
    facts.extend(
        f
        for f in (
            _fact(profile, "ap_small", s.ap_small, "scale:small"),
            _fact(profile, "ap_medium", s.ap_medium, "scale:medium"),
            _fact(profile, "ap_large", s.ap_large, "scale:large"),
            _fact(profile, "recall_small", s.recall_small, "scale:small"),
            _fact(profile, "recall_medium", s.recall_medium, "scale:medium"),
            _fact(profile, "recall_large", s.recall_large, "scale:large"),
        )
        if f is not None
    )

    for item in profile.per_class:
        facts.extend(
            f
            for f in (
                _fact(profile, "ap", item.ap, f"class:{item.name}", category_id=item.category_id, support=item.support),
                _fact(profile, "ap50", item.ap50, f"class:{item.name}", category_id=item.category_id, support=item.support),
                _fact(profile, "precision", item.precision, f"class:{item.name}", category_id=item.category_id, support=item.support),
                _fact(profile, "recall", item.recall, f"class:{item.name}", category_id=item.category_id, support=item.support),
            )
            if f is not None
        )

    fn = profile.false_negative
    if fn.total > 0:
        facts.append(_fact(profile, "false_negatives", float(fn.total), "false_negative"))  # type: ignore[arg-type]
        for class_name, count in sorted(fn.by_class.items()):
            facts.append(_fact(profile, "false_negatives", float(count), f"false_negative:class:{class_name}"))  # type: ignore[arg-type]
        for bucket, count in sorted(fn.by_scale.items()):
            facts.append(_fact(profile, "false_negatives", float(count), f"false_negative:scale:{bucket}"))  # type: ignore[arg-type]
        for bin_label, count in sorted(fn.by_confidence.items()):
            facts.append(_fact(profile, "false_negatives", float(count), f"false_negative:confidence:{bin_label}"))  # type: ignore[arg-type]

    fp = profile.false_positive
    if fp.total > 0:
        facts.append(_fact(profile, "false_positives", float(fp.total), "false_positive"))  # type: ignore[arg-type]
        facts.append(_fact(profile, "background_false_positives", float(fp.background_fp), "false_positive:background"))  # type: ignore[arg-type]
        facts.append(_fact(profile, "duplicate_detections", float(fp.duplicate_fp), "false_positive:duplicate"))  # type: ignore[arg-type]
        facts.append(_fact(profile, "class_confusions", float(fp.class_confusion_fp), "false_positive:class_confusion"))  # type: ignore[arg-type]
        facts.append(_fact(profile, "high_confidence_false_positives", float(fp.high_confidence_fp), "false_positive:high_confidence"))  # type: ignore[arg-type]

    loc = profile.localization
    facts.extend(
        f
        for f in (
            _fact(profile, "mean_matched_iou", loc.mean_matched_iou, "localization"),
            _fact(profile, "localization_error_count", float(loc.localization_error_count), "localization"),
            _fact(profile, "ap50_ap75_gap", loc.ap50_vs_ap75_gap, "localization"),
        )
        if f is not None
    )

    for pair, count in profile.classification.top_confusion_pairs:
        facts.append(_fact(profile, "confusion_pair_count", float(count), f"confusion:{pair}"))  # type: ignore[arg-type]

    conf = profile.confidence
    facts.extend(
        f
        for f in (
            _fact(profile, "expected_calibration_error", conf.expected_calibration_error, "confidence"),
        )
        if f is not None
    )

    return facts


def hypotheses_may_reference(
    fact_ids: list[str],
    facts: list[ErrorFactIdentity],
) -> bool:
    """Every referenced fact id must exist among the extracted facts."""
    known = {fact.error_fact_id for fact in facts}
    return bool(fact_ids) and all(fid in known for fid in fact_ids)


__all__ = [
    "CALCULATION_VERSION",
    "ErrorFactIdentity",
    "error_fact_identities",
    "hypotheses_may_reference",
]
