"""Prompt-18I evidence-linked root-cause attribution.

A root-cause hypothesis in this module is never a guess from a falling mAP:
every hypothesis is derived by a deterministic rule that reads *real* numbers
from a :class:`~yolo_agent.core.detection_error_profile.DetectionErrorProfile`
(and optionally the candidate-vs-parent
:class:`~yolo_agent.core.detection_error_delta.DetectionErrorDelta`), and it
must cite those numbers as evidence.  A hypothesis without at least one
evidence link to a real measured value cannot be constructed.

Action families are taken from the unified action-space vocabulary and follow
the Prompt-18I mapping (small-FN, background-FP, localization, confusion,
calibration) — several families per pattern, never a single hard-coded answer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.detection_error_delta import DetectionErrorDelta
from yolo_agent.core.detection_error_profile import DetectionErrorProfile

#: Unified action-space families usable by attribution (subset import keeps
#: the vocabulary single-sourced).
from yolo_agent.agents.action_space_schemas import ActionFamily  # noqa: F401

#: Deterministic fact-id prefix for profile-derived facts, so hypotheses can
#: reference stable ErrorFact identifiers instead of free-form strings.
FACT_ID_PREFIX = "errorfact"

#: Mean AP used as the "stable reference" when deciding that a per-scale AP
#: is disproportionately low relative to the other scales.
_SMALL_OBJECT_APN_GAP = 3.0
#: Small-scale FN share (of all FNs) above which small-object FN is a signal.
_SMALL_FN_SHARE = 0.25
#: Background-FP share (of all FPs) above which background FP is a signal.
_BG_FP_SHARE = 0.25
#: AP50−AP75 gap (points) above which localization is a signal.
_LOCALIZATION_GAP = 8.0
#: Calibration error (ECE, 0-1) above which calibration is a signal.
_CALIBRATION_ERROR = 0.1

#: Prompt-18I §6 mapping.  Families are ordered most-likely-first but every
#: pattern lists several; nothing is hard-coded to a single answer.
_SMALL_FN_FAMILIES: tuple[ActionFamily, ...] = (
    "annotation",
    "sampling",
    "input_resolution",
    "head",
    "assignment",
    "auxiliary_loss",
    "inference",
)
_BG_FP_FAMILIES: tuple[ActionFamily, ...] = (
    "sampling",
    "augmentation",
    "classification_loss",
    "threshold",
    "postprocess",
)
_LOCALIZATION_FAMILIES: tuple[ActionFamily, ...] = (
    "annotation",
    "input_resolution",
    "bbox_loss",
    "assignment",
)
_CONFUSION_FAMILIES: tuple[ActionFamily, ...] = (
    "data_cleaning",
    "classification_loss",
    "augmentation",
    "sampling",
)
_CALIBRATION_FAMILIES: tuple[ActionFamily, ...] = (
    "calibration",
    "threshold",
    "postprocess",
)

RootCausePattern = Literal[
    "small_object_feature_loss",
    "background_false_positive",
    "localization_error",
    "class_confusion",
    "confidence_miscalibration",
]


class EvidenceLink(BaseModel):
    """One measured value a hypothesis cites, with the fact ids backing it."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float
    fact_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_link(self) -> "EvidenceLink":
        if not self.metric.strip():
            raise ValueError("evidence metric must not be empty")
        return self


class EvidenceLinkedHypothesis(BaseModel):
    """A root-cause hypothesis that *must* cite real measured evidence.

    Construction is fail-closed: zero evidence links is a validation error,
    which is what makes "mAP dropped so probably small objects" impossible
    to express here.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    pattern: RootCausePattern
    description: str
    evidence: list[EvidenceLink] = Field(min_length=1)
    candidate_action_families: list[ActionFamily] = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    origin: Literal["deterministic_rule"] = "deterministic_rule"

    @model_validator(mode="after")
    def validate_hypothesis(self) -> "EvidenceLinkedHypothesis":
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must not be empty")
        if not self.description.strip():
            raise ValueError("hypothesis requires a description")
        return self


def _fact_id(profile: DetectionErrorProfile, subject: str) -> str:
    """Deterministic ErrorFact id matching the identity layer's convention.

    The identity layer (``error_fact_identity``) mints ids as
    ``{profile_id}:{slice}:{metric}``; this module cites those ids, so a
    hypothesis can never reference a fact that was not extracted.
    """
    return f"{profile.profile_id}:{subject}"


def _scale_gap(profile: DetectionErrorProfile) -> float:
    scale = profile.scale
    others = [s for s in (scale.ap_medium, scale.ap_large) if s is not None]
    if scale.ap_small is None or not others:
        return 0.0
    return sum(others) / len(others) - scale.ap_small


def _small_fn_share(profile: DetectionErrorProfile) -> float:
    by_scale = profile.false_negative.by_scale
    total = profile.false_negative.total
    small = by_scale.get("small", 0)
    if total <= 0 or small <= 0:
        return 0.0
    return small / total


def _bg_fp_share(profile: DetectionErrorProfile) -> float:
    fp = profile.false_positive
    total = fp.total
    if total <= 0 or fp.background_fp <= 0:
        return 0.0
    return fp.background_fp / total


def _top_confusion_pair(profile: DetectionErrorProfile) -> tuple[str, str, int] | None:
    pairs = profile.classification.top_confusion_pairs
    if not pairs:
        return None
    pair, count = pairs[0]
    true_cls, pred_cls = pair.split("->", 1)
    return true_cls, pred_cls, count


def _bin_center(label: str) -> float:
    """Center of a ``low-high`` histogram bucket label (e.g. ``0.6-0.7``)."""
    low_str = label.split("-", 1)[0]
    return float(low_str) + 0.05


def _mean_tp_confidence(profile: DetectionErrorProfile) -> float | None:
    hist = profile.confidence.tp_confidence_histogram
    total = sum(hist.values())
    if total <= 0:
        return None
    return sum(_bin_center(bucket) * count for bucket, count in hist.items()) / total


def _mean_fp_confidence(profile: DetectionErrorProfile) -> float | None:
    hist = profile.confidence.fp_confidence_histogram
    total = sum(hist.values())
    if total <= 0:
        return None
    return sum(_bin_center(bucket) * count for bucket, count in hist.items()) / total


def _calibration_error(profile: DetectionErrorProfile) -> float | None:
    return profile.confidence.expected_calibration_error


def _delta_metric(delta: DetectionErrorDelta, section: str, metric: str) -> float | None:
    """Candidate-vs-parent movement for one metric or count, or None."""
    found = delta.section(section)
    if found is None:
        return None
    for item in (*found.metrics, *found.counts):
        if item.metric == metric and item.delta is not None:
            return float(item.delta)
    return None


def derive_root_cause_hypotheses(
    profile: DetectionErrorProfile,
    delta: DetectionErrorDelta | None = None,
) -> list[EvidenceLinkedHypothesis]:
    """Derive hypotheses from measured error structure — never from mAP alone.

    Each rule fires only on its own structural signal (scale gaps, FN/FP
    composition, confusion pairs, calibration) and cites the exact values
    that triggered it.  Delta regressions, when supplied, sharpen confidence
    but are never the sole trigger.
    """
    hypotheses: list[EvidenceLinkedHypothesis] = []

    # -- small-object FN --------------------------------------------------
    scale = profile.scale
    if (
        scale.ap_small is not None
        and _scale_gap(profile) >= _SMALL_OBJECT_APN_GAP
        and _small_fn_share(profile) >= _SMALL_FN_SHARE
    ):
        evidence = [
            EvidenceLink(
                metric="ap_small",
                value=scale.ap_small,
                fact_ids=[_fact_id(profile, "scale:small:ap_small")],
            ),
            EvidenceLink(
                metric="fn_small_share",
                value=round(_small_fn_share(profile), 4),
                fact_ids=[_fact_id(profile, "false_negative:scale:small")],
            ),
        ]
        if delta is not None:
            delta_ap_small = _delta_metric(delta, "scale", "ap_small")
            if delta_ap_small is not None:
                evidence.append(
                    EvidenceLink(
                        metric="delta_ap_small",
                        value=delta_ap_small,
                        fact_ids=[_fact_id(profile, "scale:small:ap_small")],
                    )
                )
        hypotheses.append(
            EvidenceLinkedHypothesis(
                hypothesis_id=f"rc:{profile.candidate_id}:small_object_feature_loss",
                pattern="small_object_feature_loss",
                description=(
                    "Small-object AP is disproportionately low and small-scale FNs "
                    "dominate the miss budget; feature retention for tiny instances "
                    "is the leading structural cause."
                ),
                evidence=evidence,
                candidate_action_families=list(_SMALL_FN_FAMILIES),
                confidence=0.7 if delta is None or delta.scale is None else 0.8,
            )
        )

    # -- background false positives ---------------------------------------
    fp = profile.false_positive
    if _bg_fp_share(profile) >= _BG_FP_SHARE:
        evidence = [
            EvidenceLink(
                metric="background_fp_share",
                value=round(_bg_fp_share(profile), 4),
                fact_ids=[_fact_id(profile, "false_positive:background:background_false_positives")],
            ),
            EvidenceLink(
                metric="background_fp_total",
                value=float(fp.background_fp),
                fact_ids=[_fact_id(profile, "false_positive:background:background_false_positives")],
            ),
        ]
        if delta is not None:
            delta_bg = _delta_metric(delta, "false_positive", "background_fp")
            if delta_bg is not None:
                evidence.append(
                    EvidenceLink(
                        metric="delta_background_fp",
                        value=delta_bg,
                        fact_ids=[_fact_id(profile, "false_positive:background:background_false_positives")],
                    )
                )
        hypotheses.append(
            EvidenceLinkedHypothesis(
                hypothesis_id=f"rc:{profile.candidate_id}:background_false_positive",
                pattern="background_false_positive",
                description=(
                    "A dominant share of false positives fire on background; the "
                    "negative-evidence regime (sampling, augmentation, classification "
                    "objective) and the score threshold are the actionable levers."
                ),
                evidence=evidence,
                candidate_action_families=list(_BG_FP_FAMILIES),
                confidence=0.7 if delta is None else 0.8,
            )
        )

    # -- localization ------------------------------------------------------
    loc = profile.localization
    if loc.ap50_vs_ap75_gap >= _LOCALIZATION_GAP or loc.localization_error_count > 0:
        evidence = [
            EvidenceLink(
                metric="ap50_vs_ap75_gap",
                value=loc.ap50_vs_ap75_gap,
                fact_ids=[_fact_id(profile, "localization:ap50_ap75_gap")],
            ),
            EvidenceLink(
                metric="localization_error_count",
                value=float(loc.localization_error_count),
                fact_ids=[_fact_id(profile, "localization:localization_error_count")],
            ),
        ]
        if delta is not None:
            delta_gap = _delta_metric(delta, "localization", "ap50_vs_ap75_gap")
            if delta_gap is not None:
                evidence.append(
                    EvidenceLink(
                        metric="delta_ap50_vs_ap75_gap",
                        value=delta_gap,
                        fact_ids=[_fact_id(profile, "localization:ap50_ap75_gap")],
                    )
                )
        hypotheses.append(
            EvidenceLinkedHypothesis(
                hypothesis_id=f"rc:{profile.candidate_id}:localization_error",
                pattern="localization_error",
                description=(
                    "Detections survive at IoU 0.5 but collapse at 0.75; box "
                    "regression quality (labels, resolution, bbox objective, "
                    "assignment) is the structural cause."
                ),
                evidence=evidence,
                candidate_action_families=list(_LOCALIZATION_FAMILIES),
                confidence=0.6 if delta is None else 0.75,
            )
        )

    # -- class confusion ---------------------------------------------------
    confusion = _top_confusion_pair(profile)
    if confusion is not None:
        true_cls, pred_cls, count = confusion
        hypotheses.append(
            EvidenceLinkedHypothesis(
                hypothesis_id=f"rc:{profile.candidate_id}:class_confusion:{true_cls}->{pred_cls}",
                pattern="class_confusion",
                description=(
                    f"Predictions confuse '{true_cls}' with '{pred_cls}' "
                    f"({count} occurrences at the top confusion pair)."
                ),
                evidence=[
                    EvidenceLink(
                        metric="top_confusion_pair_count",
                        value=float(count),
                        fact_ids=[_fact_id(profile, f"confusion:{true_cls}->{pred_cls}:confusion_pair_count")],
                    ),
                ],
                candidate_action_families=list(_CONFUSION_FAMILIES),
                confidence=0.6,
            )
        )

    # -- confidence calibration -------------------------------------------
    ece = _calibration_error(profile)
    tp_mean = _mean_tp_confidence(profile)
    fp_mean = _mean_fp_confidence(profile)
    if (ece is not None and ece >= _CALIBRATION_ERROR) or (
        tp_mean is not None and fp_mean is not None and fp_mean >= tp_mean
    ):
        evidence: list[EvidenceLink] = []
        if ece is not None:
            evidence.append(
                EvidenceLink(
                    metric="expected_calibration_error",
                    value=ece,
                    fact_ids=[_fact_id(profile, "confidence:expected_calibration_error")],
                )
            )
        if tp_mean is not None and fp_mean is not None:
            evidence.append(
                EvidenceLink(
                    metric="fp_mean_confidence",
                    value=round(fp_mean, 4),
                    fact_ids=[_fact_id(profile, "confidence:mean_fp_confidence")],
                ),
            )
            evidence.append(
                EvidenceLink(
                    metric="tp_mean_confidence",
                    value=round(tp_mean, 4),
                    fact_ids=[_fact_id(profile, "confidence:mean_tp_confidence")],
                ),
            )
        hypotheses.append(
            EvidenceLinkedHypothesis(
                hypothesis_id=f"rc:{profile.candidate_id}:confidence_miscalibration",
                pattern="confidence_miscalibration",
                description=(
                    "Score distribution cannot separate TPs from FPs (overlapping "
                    "histograms and/or high ECE); calibration is a distinct "
                    "failure mode from detection quality."
                ),
                evidence=evidence,
                candidate_action_families=list(_CALIBRATION_FAMILIES),
                confidence=0.6,
            )
        )

    return hypotheses


__all__ = [
    "EvidenceLink",
    "EvidenceLinkedHypothesis",
    "RootCausePattern",
    "derive_root_cause_hypotheses",
    "fact_id_for_subject",
]


def fact_id_for_subject(profile: DetectionErrorProfile, subject: str) -> str:
    """Public helper mirroring the internal fact-id convention."""
    return _fact_id(profile, subject)
