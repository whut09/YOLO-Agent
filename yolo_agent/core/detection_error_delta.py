"""Paired error-profile deltas for one optimization round (Prompt-18I).

A :class:`DetectionErrorDelta` compares a candidate's
:class:`DetectionErrorProfile` against its parent item by item — not just
mAP — and labels every section movement as improved, regressed, or
unchanged so the agent must acknowledge accuracy gains *and*
precision/latency-style regressions in the same view.

Deltas are computed only between profiles from the same GT artifact and
dataset manifest hash (matched evaluation); anything else is refused
rather than silently compared.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    ResourceSnapshot,
)

DELTA_SCHEMA_VERSION = "detection_error_delta.v1"

DeltaVerdict = Literal["improved", "regressed", "unchanged", "not_comparable"]
EvidenceVerdict = Literal["complete", "partial", "insufficient"]


def _delta(candidate: float | None, parent: float | None) -> float | None:
    if candidate is None or parent is None:
        return None
    return round(candidate - parent, 6)


def _verdict(value: float | None, higher_is_better: bool, epsilon: float = 1e-9) -> DeltaVerdict:
    if value is None:
        return "not_comparable"
    if abs(value) <= epsilon:
        return "unchanged"
    improved = value > 0 if higher_is_better else value < 0
    return "improved" if improved else "regressed"


class MetricDelta(BaseModel):
    """One scalar movement with its verdict."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    candidate: float | None = None
    parent: float | None = None
    delta: float | None = None
    verdict: DeltaVerdict = "not_comparable"
    higher_is_better: bool = True


class CountDelta(BaseModel):
    """One error-count movement (lower is better)."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    candidate: int = 0
    parent: int = 0
    delta: int = 0
    verdict: DeltaVerdict = "unchanged"


class SectionDelta(BaseModel):
    """A named group of metric/count movements."""

    model_config = ConfigDict(extra="forbid")

    section: str
    metrics: list[MetricDelta] = Field(default_factory=list)
    counts: list[CountDelta] = Field(default_factory=list)

    def regressions(self) -> list[str]:
        names = [item.metric for item in self.metrics if item.verdict == "regressed"]
        names += [item.metric for item in self.counts if item.verdict == "regressed"]
        return [f"{self.section}.{name}" for name in names]

    def improvements(self) -> list[str]:
        names = [item.metric for item in self.metrics if item.verdict == "improved"]
        names += [item.metric for item in self.counts if item.verdict == "improved"]
        return [f"{self.section}.{name}" for name in names]


class DetectionErrorDelta(BaseModel):
    """The item-by-item movement of a candidate against its parent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = DELTA_SCHEMA_VERSION
    candidate_profile_id: str
    parent_profile_id: str
    run_id: str
    candidate_id: str
    parent_candidate_id: str = ""
    matched_evaluation: bool = True
    sections: list[SectionDelta] = Field(default_factory=list)
    candidate_resources: ResourceSnapshot | None = None
    parent_resources: ResourceSnapshot | None = None

    def section(self, name: str) -> SectionDelta | None:
        return next((item for item in self.sections if item.section == name), None)

    def metric(self, section: str, metric: str) -> MetricDelta | None:
        found = self.section(section)
        if found is None:
            return None
        return next((item for item in found.metrics if item.metric == metric), None)

    @property
    def resources(self) -> SectionDelta | None:
        """The latency/params/FLOPs/memory movement section, when present."""
        return self.section("resources")

    def regressions(self) -> list[str]:
        names: list[str] = []
        for item in self.sections:
            names.extend(item.regressions())
        return names

    def improvements(self) -> list[str]:
        names: list[str] = []
        for item in self.sections:
            names.extend(item.improvements())
        return names

    def has_regressions(self) -> bool:
        return bool(self.regressions())


def build_detection_error_delta(
    candidate: DetectionErrorProfile,
    parent: DetectionErrorProfile,
) -> DetectionErrorDelta:
    """Compare two real profiles item by item (matched evaluation enforced)."""
    matched = (
        candidate.gt_artifact == parent.gt_artifact
        and candidate.dataset_manifest_hash == parent.dataset_manifest_hash
        and candidate.split == parent.split
    )
    sections: list[SectionDelta] = []

    # --- global ---------------------------------------------------------------
    global_section = SectionDelta(section="global")
    for metric, higher in (
        ("map50", True),
        ("map50_95", True),
        ("precision", True),
        ("recall", True),
    ):
        candidate_value = getattr(candidate.global_, metric)
        parent_value = getattr(parent.global_, metric)
        delta = _delta(candidate_value, parent_value)
        global_section.metrics.append(
            MetricDelta(
                metric=metric,
                candidate=candidate_value,
                parent=parent_value,
                delta=delta,
                verdict=_verdict(delta, higher),
                higher_is_better=higher,
            )
        )
    sections.append(global_section)

    # --- scale ----------------------------------------------------------------
    scale_section = SectionDelta(section="scale")
    for metric in ("ap_small", "ap_medium", "ap_large"):
        candidate_value = getattr(candidate.scale, metric)
        parent_value = getattr(parent.scale, metric)
        delta = _delta(candidate_value, parent_value)
        scale_section.metrics.append(
            MetricDelta(
                metric=metric,
                candidate=candidate_value,
                parent=parent_value,
                delta=delta,
                verdict=_verdict(delta, True),
            )
        )
    sections.append(scale_section)

    # --- per-class (support-weighted recall + AP table) ------------------------
    class_section = SectionDelta(section="per_class")
    parent_by_name = {item.name: item for item in parent.per_class}
    for item in candidate.per_class:
        counterpart = parent_by_name.get(item.name)
        if counterpart is None:
            continue
        for metric, higher in (("ap", True), ("precision", True), ("recall", True)):
            candidate_value = getattr(item, metric)
            parent_value = getattr(counterpart, metric)
            delta = _delta(candidate_value, parent_value)
            class_section.metrics.append(
                MetricDelta(
                    metric=f"{item.name}.{metric}",
                    candidate=candidate_value,
                    parent=parent_value,
                    delta=delta,
                    verdict=_verdict(delta, higher),
                )
            )
    sections.append(class_section)

    # --- false negatives (counts) ----------------------------------------------
    fn_section = SectionDelta(section="false_negative")
    fn_section.counts.append(_count_delta("total", candidate.false_negative.total, parent.false_negative.total))
    for key in sorted(set(candidate.false_negative.by_scale) | set(parent.false_negative.by_scale)):
        fn_section.counts.append(
            _count_delta(
                f"by_scale.{key}",
                candidate.false_negative.by_scale.get(key, 0),
                parent.false_negative.by_scale.get(key, 0),
            )
        )
    for key in sorted(set(candidate.false_negative.by_class) | set(parent.false_negative.by_class)):
        fn_section.counts.append(
            _count_delta(
                f"by_class.{key}",
                candidate.false_negative.by_class.get(key, 0),
                parent.false_negative.by_class.get(key, 0),
            )
        )
    sections.append(fn_section)

    # --- false positives (counts) ----------------------------------------------
    fp_section = SectionDelta(section="false_positive")
    fp_section.counts.append(_count_delta("total", candidate.false_positive.total, parent.false_positive.total))
    for metric in ("background_fp", "duplicate_fp", "class_confusion_fp", "high_confidence_fp"):
        fp_section.counts.append(
            _count_delta(
                metric,
                getattr(candidate.false_positive, metric),
                getattr(parent.false_positive, metric),
            )
        )
    sections.append(fp_section)

    # --- localization ----------------------------------------------------------
    localization_section = SectionDelta(section="localization")
    localization_section.counts.append(
        _count_delta(
            "localization_error_count",
            candidate.localization.localization_error_count,
            parent.localization.localization_error_count,
        )
    )
    gap_delta = _delta(candidate.localization.ap50_vs_ap75_gap, parent.localization.ap50_vs_ap75_gap)
    localization_section.metrics.append(
        MetricDelta(
            metric="ap50_vs_ap75_gap",
            candidate=candidate.localization.ap50_vs_ap75_gap,
            parent=parent.localization.ap50_vs_ap75_gap,
            delta=gap_delta,
            # A smaller gap means better localization quality across IoUs.
            verdict=_verdict(gap_delta, higher_is_better=False),
            higher_is_better=False,
        )
    )
    sections.append(localization_section)

    # --- confidence --------------------------------------------------------------
    confidence_section = SectionDelta(section="confidence")
    confidence_section.metrics.append(
        MetricDelta(
            metric="expected_calibration_error",
            candidate=candidate.confidence.expected_calibration_error,
            parent=parent.confidence.expected_calibration_error,
            delta=_delta(
                candidate.confidence.expected_calibration_error,
                parent.confidence.expected_calibration_error,
            ),
            # Lower ECE is better calibrated.
            verdict=_verdict(
                _delta(
                    candidate.confidence.expected_calibration_error,
                    parent.confidence.expected_calibration_error,
                ),
                higher_is_better=False,
            ),
            higher_is_better=False,
        )
    )
    sections.append(confidence_section)

    # --- resources (latency/params/FLOPs/memory) -------------------------------
    # Only appended when both sides carry real runtime resource snapshots;
    # a missing snapshot never produces a fabricated movement.
    if candidate.resources is not None and parent.resources is not None:
        resources_section = SectionDelta(section="resources")
        for metric, higher, cast in (
            ("latency_ms", False, float),
            ("params", False, int),
            ("flops_g", False, float),
            ("peak_memory_mb", False, float),
        ):
            candidate_value = getattr(candidate.resources, metric)
            parent_value = getattr(parent.resources, metric)
            if candidate_value is None or parent_value is None:
                continue
            if cast is int:
                resources_section.counts.append(
                    _count_delta(metric, int(candidate_value), int(parent_value))
                )
            else:
                movement = _delta(candidate_value, parent_value)
                resources_section.metrics.append(
                    MetricDelta(
                        metric=metric,
                        candidate=candidate_value,
                        parent=parent_value,
                        delta=movement,
                        verdict=_verdict(movement, higher),
                        higher_is_better=higher,
                    )
                )
        if resources_section.metrics or resources_section.counts:
            sections.append(resources_section)

    return DetectionErrorDelta(
        candidate_profile_id=candidate.profile_id,
        parent_profile_id=parent.profile_id,
        run_id=candidate.run_id,
        candidate_id=candidate.candidate_id,
        parent_candidate_id=parent.candidate_id,
        matched_evaluation=matched,
        sections=sections,
        candidate_resources=candidate.resources,
        parent_resources=parent.resources,
    )


def _count_delta(metric: str, candidate: int, parent: int) -> CountDelta:
    delta = candidate - parent
    return CountDelta(
        metric=metric,
        candidate=candidate,
        parent=parent,
        delta=delta,
        verdict="unchanged" if delta == 0 else ("improved" if delta < 0 else "regressed"),
    )
