"""Required evidence artifacts for paper protocol families.

These names are paper-protocol requirements. They are not ComponentContract
fields and they do not authorize training by themselves.
"""

from __future__ import annotations

from typing import Literal


ProtocolFamilyName = Literal[
    "domain_adaptation",
    "distillation",
    "model_graph",
    "inference_only",
    "standard_training",
]

COMMON_EVIDENCE = (
    "paper_protocol_contract",
    "paired_baseline_control",
    "current_node_error_facts",
    "same_protocol_hash",
)

DOMAIN_EVIDENCE = (
    "source_domain_dataset",
    "target_domain_dataset",
    "source_domain_manifest",
    "target_domain_manifest",
    "explicit_source_target_domain_ids",
    "source_target_split",
)

DISTILLATION_EVIDENCE = (
    "teacher_checkpoint",
    "teacher_checkpoint_sha256",
    "student_checkpoint",
    "teacher_student_dataset_manifest",
    "teacher_student_same_split",
    "student_only_evaluation",
)

GRAPH_EVIDENCE = (
    "graph_identity",
    "yolo26_one_to_one_head",
    "native_dfl_free_regression",
    "imgsz_640",
)

INFERENCE_EVIDENCE = (
    "inference_policy_contract",
    "inference_evaluation_protocol",
    "training_asha_excluded",
)

STANDARD_EVIDENCE = (
    "coco_official_split",
    "imgsz_640",
)

DOMAIN_DATASET_ACTIONS = (
    "provide_source_domain_dataset",
    "provide_target_domain_dataset",
    "record_explicit_domain_ids",
    "do_not_reuse_coco_train_val_as_paper_domains",
)


def evidence_artifacts_for_family(family: ProtocolFamilyName) -> list[str]:
    """Return the required evidence artifact names for one protocol family."""
    mapping: dict[ProtocolFamilyName, tuple[str, ...]] = {
        "domain_adaptation": DOMAIN_EVIDENCE,
        "distillation": DISTILLATION_EVIDENCE,
        "model_graph": GRAPH_EVIDENCE,
        "inference_only": INFERENCE_EVIDENCE,
        "standard_training": STANDARD_EVIDENCE,
    }
    return list(dict.fromkeys([*COMMON_EVIDENCE, *mapping[family]]))


def missing_dataset_actions(side: Literal["source", "target", "both"] = "both") -> list[str]:
    """Return explicit dataset recovery actions for domain-adaptation papers."""
    if side == "source":
        return [
            "provide_source_domain_dataset",
            "do_not_reuse_coco_train_val_as_paper_domains",
        ]
    if side == "target":
        return [
            "provide_target_domain_dataset",
            "do_not_reuse_coco_train_val_as_paper_domains",
        ]
    return list(DOMAIN_DATASET_ACTIONS)


def required_metrics_for_family(family: ProtocolFamilyName) -> list[str]:
    """Return the metrics a paper protocol must report."""
    if family == "inference_only":
        return ["map50_95", "latency_ms"]
    if family == "domain_adaptation":
        return ["map50_95", "target_domain_map50"]
    if family == "distillation":
        return ["map50_95", "student_map50_95"]
    return ["map50_95"]
