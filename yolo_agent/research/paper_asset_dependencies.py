"""Exact paper-mechanism to runtime-asset dependency rules.

The paper pipeline has several different kinds of manifests: a supervised
dataset manifest, a domain manifest, a teacher/student manifest, and a
train-side hard-negative replay manifest.  They are intentionally kept
separate here so a broad evidence label cannot accidentally create a replay
dependency for an unrelated paper.
"""

from __future__ import annotations

from collections.abc import Iterable


HARD_NEGATIVE_REPLAY_MECHANISM = "sampling.hard_negative_replay"

# These are semantic branch IDs used by the paper-specific route registries.
# The generic canonical IDs are handled by their namespace prefixes below.
DOMAIN_BRANCH_MECHANISMS = frozenset(
    {
        "adversarial_alignment",
        "feature_alignment",
        "pseudo_label_adaptation",
        "domain_distillation",
        "source_free_adaptation",
        "cross_domain_teacher",
        "contrastive_domain_alignment",
        "active_domain_adaptation",
    }
)

DISTILLATION_BRANCH_MECHANISMS = frozenset(
    {
        "logits_distillation",
        "feature_distillation",
        "relation_distillation",
        "localization_distillation",
        "attention_distillation",
        "masked_feature_distillation",
        "quality_aware_distillation",
        "teacher_ensemble",
    }
)

# These namespaces alter the model graph.  Assignment is deliberately absent:
# assignment changes matching/loss semantics while preserving the YOLO26 graph.
GRAPH_MECHANISM_PREFIXES = (
    "neck.",
    "detection_head.",
    "feature_pyramid.",
    "attention.",
)


def normalize_mechanism_ids(mechanism_ids: Iterable[str]) -> frozenset[str]:
    """Return cleaned, stable mechanism IDs without inventing aliases."""

    return frozenset(
        str(item).strip()
        for item in mechanism_ids
        if str(item).strip()
    )


def requires_hard_negative_replay(mechanism_ids: Iterable[str]) -> bool:
    """Whether the candidate consumes a train-side replay manifest.

    Only the explicit replay component opts in.  In particular, a generic
    ``required_manifest_assets`` entry for a domain or distillation protocol
    is not evidence that the candidate uses hard-negative replay.
    """

    return HARD_NEGATIVE_REPLAY_MECHANISM in normalize_mechanism_ids(mechanism_ids)


def requires_teacher_checkpoint(mechanism_ids: Iterable[str]) -> bool:
    """Whether a frozen teacher checkpoint is part of the method protocol."""

    ids = normalize_mechanism_ids(mechanism_ids)
    return any(item.startswith("distillation.") for item in ids) or bool(
        ids & DISTILLATION_BRANCH_MECHANISMS
    ) or "cross_domain_teacher" in ids or "domain_distillation" in ids


def requires_domain_assets(mechanism_ids: Iterable[str]) -> bool:
    """Whether the method needs distinct source and target domain assets."""

    ids = normalize_mechanism_ids(mechanism_ids)
    return any(item.startswith("domain_adaptation.") for item in ids) or bool(
        ids & DOMAIN_BRANCH_MECHANISMS
    )


def requires_graph_config(mechanism_ids: Iterable[str]) -> bool:
    """Whether a candidate changes the YOLO26 model graph."""

    ids = normalize_mechanism_ids(mechanism_ids)
    return any(
        item.startswith(prefix)
        for item in ids
        for prefix in GRAPH_MECHANISM_PREFIXES
    )


def is_inference_only(mechanism_ids: Iterable[str]) -> bool:
    """Whether a mechanism is evaluation-only and cannot enter training ASHA."""

    return any(
        item.startswith("inference.")
        for item in normalize_mechanism_ids(mechanism_ids)
    )


def is_training_mechanism(mechanism_ids: Iterable[str]) -> bool:
    """Return whether the mechanism set describes a training candidate."""

    return not is_inference_only(mechanism_ids)


def asset_scope_violations(
    mechanism_ids: Iterable[str],
    *,
    teacher_assets: Iterable[str] = (),
    domain_assets: Iterable[str] = (),
    manifest_assets: Iterable[str] = (),
    graph_assets: Iterable[str] = (),
) -> list[str]:
    """Return dependency declarations that do not match exact mechanisms."""

    ids = normalize_mechanism_ids(mechanism_ids)
    teachers = tuple(str(item) for item in teacher_assets)
    domains = tuple(str(item) for item in domain_assets)
    manifests = tuple(str(item) for item in manifest_assets)
    graphs = tuple(str(item) for item in graph_assets)
    violations: list[str] = []
    if teachers and not requires_teacher_checkpoint(ids):
        violations.append("teacher_assets_without_teacher_mechanism")
    if domains and not requires_domain_assets(ids):
        violations.append("domain_assets_without_domain_mechanism")
    if graphs and not requires_graph_config(ids):
        violations.append("graph_assets_without_graph_mechanism")
    for asset in manifests:
        if asset == "hard_negative_manifest" or "hard_negative" in asset or asset == "train_replay":
            if not requires_hard_negative_replay(ids):
                violations.append(
                    f"hard_negative_manifest_without_{HARD_NEGATIVE_REPLAY_MECHANISM}"
                )
        elif asset == "teacher_student_dataset_manifest" or "teacher" in asset:
            if not requires_teacher_checkpoint(ids):
                violations.append("teacher_manifest_without_teacher_mechanism")
        elif "domain" in asset or asset in {"source_target_split", "label_availability"}:
            if not requires_domain_assets(ids):
                violations.append("domain_manifest_without_domain_mechanism")
    return list(dict.fromkeys(violations))


__all__ = [
    "DOMAIN_BRANCH_MECHANISMS",
    "DISTILLATION_BRANCH_MECHANISMS",
    "GRAPH_MECHANISM_PREFIXES",
    "HARD_NEGATIVE_REPLAY_MECHANISM",
    "is_inference_only",
    "is_training_mechanism",
    "normalize_mechanism_ids",
    "asset_scope_violations",
    "requires_domain_assets",
    "requires_graph_config",
    "requires_hard_negative_replay",
    "requires_teacher_checkpoint",
]
