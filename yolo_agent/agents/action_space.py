"""Diagnosis-to-action chain over the unified action space.

Deterministic core of the unified-action-space contract: an observed
detection error never maps to a single hardcoded answer.  ``ErrorFact``
symptoms expand into multiple ``RootCauseHypothesis`` entries, each
hypothesis expands into multiple ``ActionFamily`` candidates, and each
family resolves to concrete ``ActionSpec`` records drawn from the catalog —
paper lineage preserved where the action comes from the frozen 83, local
actions allowed where they are plain engineering moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from yolo_agent.agents.action_space_schemas import (
    ActionFamilySelection,
    ActionSpec,
    DiagnosisActionChain,
    RootCauseHypothesis,
)
from yolo_agent.core.error_facts import ErrorFact


# --------------------------------------------------------------- catalog -----

class ActionCatalog:
    """A loaded set of ActionSpec records, queryable by family and tag."""

    def __init__(self, actions: list[ActionSpec]) -> None:
        ids = [action.action_id for action in actions]
        if len(ids) != len(set(ids)):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            raise ValueError(f"duplicate action_ids in catalog: {', '.join(duplicates)}")
        self.actions = list(actions)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ActionCatalog":
        """Load the catalog from a YAML file of action specs."""

        catalog_path = Path(path)
        with catalog_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"action catalog YAML must contain a mapping: {catalog_path}")
        raw_actions = data.get("actions", [])
        if not isinstance(raw_actions, list):
            raise ValueError("action catalog YAML requires an 'actions' list")
        return cls([ActionSpec.model_validate(item) for item in raw_actions])

    def by_family(self, family: str) -> list[ActionSpec]:
        return [action for action in self.actions if action.family == family]

    def by_problem_tags(
        self,
        tags: list[str],
        *,
        families: list[str] | None = None,
    ) -> list[ActionSpec]:
        """Actions matching ANY of the tags, optionally restricted to families."""

        wanted = set(tags)
        selected = [
            action
            for action in self.actions
            if wanted & set(action.problem_tags)
            and (families is None or action.family in families)
        ]
        return selected

    def paper_actions(self) -> list[ActionSpec]:
        return [action for action in self.actions if action.is_paper_action]

    def local_actions(self) -> list[ActionSpec]:
        return [action for action in self.actions if action.paper_lineage == "local"]


# -------------------------------------------------------- hypothesis rules ----

#: Deterministic hypothesis rules.  One symptom (matched by error-fact shape
#: or problem tag) expands into every plausible cause — deliberately more
#: than one family per error so no single-answer mapping can hide behind the
#: rule table.
_HYPOTHESIS_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "small_object_miss",
        "problem_tags": ["small_object_fn"],
        "hypotheses": [
            {
                "hypothesis_id": "h.annotation_missing_small",
                "description": "Small targets are under-annotated; the model never sees them as positives.",
                "confirming_evidence": ["missing_labels", "label_density_scan"],
                "candidate_action_families": ["annotation", "data_cleaning", "active_learning"],
                "confidence": 0.4,
            },
            {
                "hypothesis_id": "h.resolution_insufficient",
                "description": "Input resolution drops tiny objects below detectable scale.",
                "confirming_evidence": ["imgsz_sweep", "pixel_area_histogram"],
                "candidate_action_families": ["input_resolution", "preprocessing"],
                "confidence": 0.6,
            },
            {
                "hypothesis_id": "h.sampling_skew_small",
                "description": "Sampling under-weights small-object-rich images.",
                "confirming_evidence": ["area_distribution_by_batch"],
                "candidate_action_families": ["sampling", "data_selection"],
                "confidence": 0.45,
            },
            {
                "hypothesis_id": "h.feature_level_missing",
                "description": "Feature pyramid lacks a high-resolution level for tiny objects.",
                "confirming_evidence": ["p2_ablation", "stride_coverage"],
                "candidate_action_families": ["neck", "feature_fusion", "head"],
                "confidence": 0.55,
            },
            {
                "hypothesis_id": "h.assignment_loss_mismatch_small",
                "description": "Assignment or bbox loss metric is insensitive to tiny-box overlap.",
                "confirming_evidence": ["positive_count_per_area", "loss_term_analysis"],
                "candidate_action_families": ["assignment", "bbox_loss"],
                "confidence": 0.45,
            },
            {
                "hypothesis_id": "h.inference_slicing_gap",
                "description": "Whole-image inference loses tiny objects that slicing would keep.",
                "confirming_evidence": ["tiled_vs_whole_eval"],
                "candidate_action_families": ["inference", "postprocess"],
                "confidence": 0.4,
            },
        ],
    },
    {
        "rule_id": "area_miss",
        "problem_tags": ["small_object_fn", "medium_object_fn", "large_object_fn"],
        "hypotheses": [
            {
                "hypothesis_id": "h.medium_large_sampling_skew",
                "description": "Medium/large-object images are under-represented in training batches.",
                "confirming_evidence": ["area_distribution_by_batch"],
                "candidate_action_families": ["sampling", "data_selection"],
                "confidence": 0.45,
            },
            {
                "hypothesis_id": "h.medium_large_resolution_cost",
                "description": "Resolution reduction degrades medium/large object detail.",
                "confirming_evidence": ["imgsz_sweep"],
                "candidate_action_families": ["input_resolution", "model_scale"],
                "confidence": 0.4,
            },
            {
                "hypothesis_id": "h.medium_large_assignment_spread",
                "description": "Assignment spreads positives across levels and starves medium/large heads.",
                "confirming_evidence": ["positives_per_gt_histogram"],
                "candidate_action_families": ["assignment", "neck"],
                "confidence": 0.4,
            },
        ],
    },
    {
        "rule_id": "false_positive_background",
        "problem_tags": ["background_fp"],
        "hypotheses": [
            {
                "hypothesis_id": "h.hard_negative_pressure",
                "description": "Background false positives from under-weighted hard negatives.",
                "confirming_evidence": ["negative_loss_curve", "score_histogram_fp"],
                "candidate_action_families": ["classification_loss", "regularization", "data_selection"],
                "confidence": 0.5,
            },
            {
                "hypothesis_id": "h.nms_duplicate_collapse",
                "description": "Duplicated detections survive merge and count as background FP.",
                "confirming_evidence": ["duplicate_iou_histogram"],
                "candidate_action_families": ["postprocess", "threshold"],
                "confidence": 0.45,
            },
            {
                "hypothesis_id": "h.label_noise_fp",
                "description": "Incorrect or missing ground truth fabricates background FP counts.",
                "confirming_evidence": ["annotation_audit", "label_noise_scan"],
                "candidate_action_families": ["data_cleaning", "annotation"],
                "confidence": 0.35,
            },
        ],
    },
    {
        "rule_id": "duplicate_detection",
        "problem_tags": ["duplicate_detection"],
        "hypotheses": [
            {
                "hypothesis_id": "h.merge_strategy_weak",
                "description": "Merge/NMS strategy keeps overlapping boxes across classes or scales.",
                "confirming_evidence": ["cross_class_overlap_scan"],
                "candidate_action_families": ["postprocess", "inference"],
                "confidence": 0.55,
            },
            {
                "hypothesis_id": "h.assignment_multi_positive",
                "description": "Assignment produces multiple positives per object across levels.",
                "confirming_evidence": ["positives_per_gt_histogram"],
                "candidate_action_families": ["assignment", "head"],
                "confidence": 0.45,
            },
        ],
    },
    {
        "rule_id": "classification_confusion",
        "problem_tags": ["classification_confusion", "long_tail", "class_imbalance"],
        "hypotheses": [
            {
                "hypothesis_id": "h.class_confusion_representation",
                "description": "Inter-class representation gap for the confused pair.",
                "confirming_evidence": ["embedding_projection", "confusion_matrix"],
                "candidate_action_families": ["backbone", "auxiliary_loss", "head"],
                "confidence": 0.5,
            },
            {
                "hypothesis_id": "h.long_tail_frequency_bias",
                "description": "Head frequencies bias predictions toward dominant classes.",
                "confirming_evidence": ["per_class_ap_vs_count"],
                "candidate_action_families": ["sampling", "data_selection", "classification_loss", "regularization"],
                "confidence": 0.55,
            },
            {
                "hypothesis_id": "h.annotation_pair_noise",
                "description": "The confused pair carries label noise in one direction.",
                "confirming_evidence": ["pair_annotation_audit"],
                "candidate_action_families": ["annotation", "data_cleaning"],
                "confidence": 0.35,
            },
        ],
    },
    {
        "rule_id": "localization_error",
        "problem_tags": ["localization_error"],
        "hypotheses": [
            {
                "hypothesis_id": "h.bbox_loss_geometry",
                "description": "Bounding-box regression metric mismatched to object geometry.",
                "confirming_evidence": ["iou_distribution", "loss_gradient_scan"],
                "candidate_action_families": ["bbox_loss", "assignment"],
                "confidence": 0.55,
            },
            {
                "hypothesis_id": "h.quality_target_gap",
                "description": "Localization quality target misaligned with box quality.",
                "confirming_evidence": ["quality_target_vs_iou"],
                "candidate_action_families": ["auxiliary_loss", "assignment"],
                "confidence": 0.45,
            },
            {
                "hypothesis_id": "h.low_confidence_localization",
                "description": "Localization errors concentrate at low-confidence predictions.",
                "confirming_evidence": ["conf_by_iou_bucket"],
                "candidate_action_families": ["calibration", "threshold", "postprocess"],
                "confidence": 0.4,
            },
        ],
    },
    {
        "rule_id": "confidence_miscalibration",
        "problem_tags": ["low_confidence", "overconfidence"],
        "hypotheses": [
            {
                "hypothesis_id": "h.score_calibration_gap",
                "description": "Scores are systematically miscalibrated for the operating domain.",
                "confirming_evidence": ["reliability_diagram"],
                "candidate_action_families": ["calibration", "threshold"],
                "confidence": 0.55,
            },
            {
                "hypothesis_id": "h.quality_head_mismatch",
                "description": "Classification and localization quality disagree; scores drift.",
                "confirming_evidence": ["cls_iou_correlation"],
                "candidate_action_families": ["auxiliary_loss", "head", "assignment"],
                "confidence": 0.45,
            },
        ],
    },
    {
        "rule_id": "long_tail_imbalance",
        "problem_tags": ["long_tail", "class_imbalance"],
        "hypotheses": [
            {
                "hypothesis_id": "h.tail_sampling_starved",
                "description": "Tail classes are under-sampled in training batches.",
                "confirming_evidence": ["class_frequency_per_batch"],
                "candidate_action_families": ["sampling", "data_selection"],
                "confidence": 0.55,
            },
            {
                "hypothesis_id": "h.tail_annotation_gap",
                "description": "Tail classes are under-annotated, not under-modeled.",
                "confirming_evidence": ["annotation_density_by_class"],
                "candidate_action_families": ["annotation", "active_learning", "data_cleaning"],
                "confidence": 0.4,
            },
        ],
    },
    {
        "rule_id": "label_quality",
        "problem_tags": ["label_noise", "missing_labels"],
        "hypotheses": [
            {
                "hypothesis_id": "h.noise_poisons_training",
                "description": "Label noise teaches the model wrong positives.",
                "confirming_evidence": ["noise_rate_estimate", "loss_outlier_scan"],
                "candidate_action_families": ["data_cleaning", "annotation", "regularization"],
                "confidence": 0.55,
            },
            {
                "hypothesis_id": "h.missing_labels_suppress",
                "description": "Missing labels turn true objects into false-positive pressure.",
                "confirming_evidence": ["pseudo_label_disagreement"],
                "candidate_action_families": ["annotation", "semi_supervised", "data_cleaning"],
                "confidence": 0.5,
            },
        ],
    },
    {
        "rule_id": "image_quality",
        "problem_tags": ["low_contrast", "blur", "occlusion"],
        "hypotheses": [
            {
                "hypothesis_id": "h.domain_image_quality",
                "description": "Deployment images degrade (contrast/blur/occlusion) versus training data.",
                "confirming_evidence": ["image_quality_histogram", "train_val_gap"],
                "candidate_action_families": ["augmentation", "preprocessing", "domain_adaptation"],
                "confidence": 0.5,
            },
            {
                "hypothesis_id": "h.train_augmentation_gap",
                "description": "Training augmentation does not cover the observed degradation.",
                "confirming_evidence": ["augmentation_coverage_scan"],
                "candidate_action_families": ["augmentation"],
                "confidence": 0.45,
            },
        ],
    },
    {
        "rule_id": "domain_shift",
        "problem_tags": ["domain_shift"],
        "hypotheses": [
            {
                "hypothesis_id": "h.feature_domain_gap",
                "description": "Source/target feature distributions diverge.",
                "confirming_evidence": ["feature_statistics_shift"],
                "candidate_action_families": ["domain_adaptation", "augmentation", "preprocessing"],
                "confidence": 0.55,
            },
            {
                "hypothesis_id": "h.pseudo_label_drift",
                "description": "Target-domain pseudo labels drift as the model adapts.",
                "confirming_evidence": ["pseudo_label_confidence_trend"],
                "candidate_action_families": ["semi_supervised", "domain_adaptation"],
                "confidence": 0.4,
            },
        ],
    },
    {
        "rule_id": "fit_state",
        "problem_tags": ["overfitting", "underfitting", "training_instability"],
        "hypotheses": [
            {
                "hypothesis_id": "h.overfit_capacity",
                "description": "Model memorizes training set; regularization under-applied.",
                "confirming_evidence": ["train_val_curve"],
                "candidate_action_families": ["regularization", "augmentation", "model_scale", "training_strategy"],
                "confidence": 0.55,
            },
            {
                "hypothesis_id": "h.underfit_capacity",
                "description": "Model capacity or schedule too small for the task.",
                "confirming_evidence": ["train_loss_plateau"],
                "candidate_action_families": ["model_scale", "input_resolution", "optimizer", "training_strategy"],
                "confidence": 0.5,
            },
            {
                "hypothesis_id": "h.instability_optimization",
                "description": "Optimizer schedule or loss weighting destabilizes training.",
                "confirming_evidence": ["loss_curve_spikes", "grad_norm_history"],
                "candidate_action_families": ["optimizer", "regularization", "training_strategy", "auxiliary_loss"],
                "confidence": 0.5,
            },
        ],
    },
]


def hypotheses_for_problem_tags(
    tags: list[str],
    *,
    origin: str = "deterministic_rule",
) -> list[RootCauseHypothesis]:
    """Expand problem tags into every plausible root-cause hypothesis."""

    wanted = set(tags)
    hypotheses: list[RootCauseHypothesis] = []
    seen: set[str] = set()
    for rule in _HYPOTHESIS_RULES:
        if not wanted & set(rule["problem_tags"]):
            continue
        for raw in rule["hypotheses"]:
            if raw["hypothesis_id"] in seen:
                continue
            seen.add(raw["hypothesis_id"])
            payload = dict(raw)
            payload["problem_tags"] = list(rule["problem_tags"])
            payload["origin"] = origin
            hypotheses.append(RootCauseHypothesis.model_validate(payload))
    return hypotheses


def build_diagnosis_action_chain(
    facts: list[ErrorFact],
    catalog: ActionCatalog,
    *,
    extra_tags: list[str] | None = None,
    hypothesis_origin: str = "deterministic_rule",
) -> DiagnosisActionChain:
    """Build the full ErrorFact → hypotheses → families → specs chain.

    Every hypothesis resolves against the catalog; families without catalog
    entries still appear in the selection (the gap is actionable signal, not
    something to hide), but the chain records which families have concrete
    actions available.
    """

    tags: set[str] = set(extra_tags or [])
    fact_refs: list[str] = []
    for fact in facts:
        fact_refs.append(f"{fact.run_id}:{fact.node_id}:{fact.fact_type}:{fact.subject}")
        tag = _fact_tag(fact)
        if tag:
            tags.add(tag)
    ordered_tags = sorted(tags)

    hypotheses = hypotheses_for_problem_tags(ordered_tags, origin=hypothesis_origin)
    if not hypotheses:
        raise ValueError(
            "no hypothesis rule covers problem tags: " + ", ".join(ordered_tags)
        )

    selections: list[ActionFamilySelection] = []
    for hypothesis in hypotheses:
        families = list(hypothesis.candidate_action_families)
        specs = catalog.by_problem_tags(
            sorted(set(ordered_tags) | set(hypothesis.problem_tags)),
            families=families,
        )
        selections.append(
            ActionFamilySelection(
                hypothesis_id=hypothesis.hypothesis_id,
                families=families,
                action_specs=specs,
            )
        )
    return DiagnosisActionChain(
        error_fact_refs=fact_refs,
        problem_tags=ordered_tags,
        hypotheses=hypotheses,
        selections=selections,
    )


def _fact_tag(fact: ErrorFact) -> str:
    """Map an ErrorFact to its problem tag over the unified vocabulary."""

    fact_type = fact.fact_type
    if fact_type in {"area_metric", "scale_variation"} and fact.area == "small":
        return "small_object_fn"
    if fact_type == "area_metric" and fact.area == "medium":
        return "medium_object_fn"
    if fact_type == "area_metric" and fact.area == "large":
        return "large_object_fn"
    if fact_type in {"background_false_positive_class", "high_confidence_false_positive"}:
        return "background_fp"
    if fact_type == "duplicate_prediction":
        return "duplicate_detection"
    if fact_type in {"class_confusion_pair", "class_low_ap", "per_class_metric"}:
        return "classification_confusion"
    if fact_type in {"localization_error", "localization_heavy_class", "confidence_localization_mismatch"}:
        return "localization_error"
    if fact_type in {"false_negative_heavy_class", "assignment_conflict"}:
        return "small_object_fn" if fact.area == "small" else "localization_error"
    if fact_type in {"representation_gap", "feature_relation_gap", "capacity_gap"}:
        return "underfitting"
    return ""


__all__ = [
    "ActionCatalog",
    "build_diagnosis_action_chain",
    "hypotheses_for_problem_tags",
]
