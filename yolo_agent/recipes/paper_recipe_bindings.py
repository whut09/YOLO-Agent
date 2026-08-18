"""One explicit recipe binding per certified paper.

Shared real implementations may list multiple paper_ids. Generic family IDs
never cover a paper family by themselves.
"""

from __future__ import annotations

from typing import Any

from yolo_agent.recipes.paper_recipe_spec import (
    PaperRecipeDisposition,
    PaperRecipeSpec,
    queue_disposition,
)
from yolo_agent.research.paper_protocol_catalog import build_paper_protocol_contract
from yolo_agent.research.paper_protocol_ids import CERTIFIED_PAPER_MECHANISMS


EXISTING_SPECIFIC_RECIPES: dict[str, dict[str, Any]] = {
    "arxiv:2103.14259": {
        "recipe_id": "yolo26_ota_assignment_shadow",
        "paper_specific_mechanism_id": "assigner.optimal_transport",
        "canonical_component_ids": ["assigner.optimal_transport"],
        "changed_variables": {"assigner.kind": "optimal_transport"},
        "runtime_plugin": "assigner.optimal_transport",
        "graph_identity": "assigner.optimal_transport",
        "shared_with": ["arxiv:2107.08430"],
    },
    "arxiv:2107.08430": {
        "recipe_id": "yolo26_ota_assignment_shadow",
        "paper_specific_mechanism_id": "assigner.optimal_transport",
        "canonical_component_ids": ["assigner.optimal_transport"],
        "changed_variables": {"assigner.kind": "optimal_transport"},
        "runtime_plugin": "assigner.optimal_transport",
        "graph_identity": "assigner.optimal_transport",
        "shared_with": ["arxiv:2103.14259"],
    },
    "arxiv:2108.07755": {
        "recipe_id": "yolo26_tood_tal_assignment_shadow",
        "paper_specific_mechanism_id": "assigner.task_aligned",
        "canonical_component_ids": ["assigner.task_aligned", "detection_head.task_aligned"],
        "changed_variables": {"assigner.kind": "task_aligned", "head.kind": "task_aligned"},
        "runtime_plugin": "assigner.task_aligned",
        "graph_identity": "assigner.task_aligned+detection_head.task_aligned",
        "shared_with": ["arxiv:2203.16250"],
    },
    "arxiv:2203.16250": {
        "recipe_id": "yolo26_tood_tal_assignment_shadow",
        "paper_specific_mechanism_id": "assigner.task_aligned",
        "canonical_component_ids": ["assigner.task_aligned"],
        "changed_variables": {"assigner.kind": "task_aligned"},
        "runtime_plugin": "assigner.task_aligned",
        "graph_identity": "assigner.task_aligned",
        "shared_with": ["arxiv:2108.07755"],
    },
    "arxiv:2208.00817": {
        "recipe_id": "yolo26_dsla_assignment_shadow",
        "paper_specific_mechanism_id": "assigner.dynamic_smooth_label",
        "canonical_component_ids": ["assigner.dynamic_smooth_label"],
        "changed_variables": {"assigner.kind": "dynamic_smooth_label"},
        "runtime_plugin": "assigner.dynamic_smooth_label",
        "graph_identity": "assigner.dynamic_smooth_label",
    },
    "arxiv:2104.14082": {
        "recipe_id": "yolo26_pseudo_iou_quality_auxiliary_loss",
        "paper_specific_mechanism_id": "loss.quality.pseudo_iou",
        "canonical_component_ids": ["loss.quality.pseudo_iou"],
        "changed_variables": {"loss.quality": "pseudo_iou"},
        "runtime_plugin": "loss.quality.pseudo_iou",
    },
    "arxiv:2301.01019": {
        "recipe_id": "yolo26_correlation_auxiliary_loss",
        "paper_specific_mechanism_id": "loss.quality.correlation",
        "canonical_component_ids": ["loss.quality.correlation"],
        "changed_variables": {"loss.quality": "correlation"},
        "runtime_plugin": "loss.quality.correlation",
    },
    "arxiv:2303.14404": {
        "recipe_id": "yolo26_bpc_calibration_auxiliary_loss",
        "paper_specific_mechanism_id": "loss.calibration.bpc",
        "canonical_component_ids": ["loss.calibration.bpc"],
        "changed_variables": {"loss.calibration": "bpc"},
        "runtime_plugin": "loss.calibration.bpc",
    },
    "arxiv:2212.07784": {
        "recipe_id": "yolo26_rtmdet_large_kernel_neck",
        "paper_specific_mechanism_id": "neck.rtmdet_large_kernel",
        "canonical_component_ids": ["neck.rtmdet_large_kernel"],
        "changed_variables": {"neck.kind": "rtmdet_large_kernel"},
        "runtime_plugin": "neck.rtmdet_large_kernel",
        "graph_identity": "neck.rtmdet_large_kernel",
    },
    "arxiv:2309.11331": {
        "recipe_id": "yolo26_gold_gather_distribute_neck",
        "paper_specific_mechanism_id": "neck.gold_gather_distribute",
        "canonical_component_ids": [
            "neck.gold_gather_distribute",
            "neck.multi_scale_fusion",
            "feature_pyramid.multi_scale",
        ],
        "changed_variables": {"neck.kind": "gold_gather_distribute"},
        "runtime_plugin": "neck.gold_gather_distribute",
        "graph_identity": "neck.gold_gather_distribute",
    },
}

NAMED_METHOD_SLUGS: dict[str, str] = {
    "cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection": "general_instance",
    "cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features": "decoupled_features",
    "cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection": "dense_relation_fewshot",
    "cvf:cvpr2021:VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object": "mega_cda",
    "cvf:cvpr2021:Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector": "rpn_prototype_alignment",
    "cvf:cvpr2022:Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response": "elastic_response_incremental",
    "cvf:cvpr2022:Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection": "scale_equivalent",
    "cvf:cvpr2022:He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation": "target_perceived_dual_branch",
    "cvf:cvpr2022:Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection": "adaptive_teacher",
    "cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection": "sigma_graph_matching",
    "cvf:cvpr2022:Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation": "cyclic_disentangled",
    "cvf:cvpr2022:Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection": "multi_source_knowledge",
    "cvf:cvpr2022:Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection": "inconsistency_alignment",
    "cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection": "localization",
    "cvf:cvpr2022:Zhou_Multi-Granularity_Alignment_Domain_Adaptation_for_Object_Detection": "multi_granularity",
    "cvf:cvpr2023:Cao_Contrastive_Mean_Teacher_for_Domain_Adaptive_Object_Detectors": "contrastive_mean_teacher",
    "cvf:cvpr2023:Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object": "asyfod",
    "cvf:cvpr2023:Liu_CIGAR_Cross-Modality_Graph_Reasoning_for_Domain_Adaptive_Object_Detection": "cigar_graph",
    "cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection": "source_free_irg",
    "cvf:cvpr2023:Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection": "object_aware_pyramid",
    "cvf:cvpr2023:Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector": "scalekd",
    "cvf:cvpr2024:Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation": "zero_shot_day_night",
    "cvf:cvpr2024:Kennerley_CAT_Exploiting_Inter-Class_Dynamics_for_Domain_Adaptive_Object_Detection": "cat_interclass",
    "cvf:cvpr2024:Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection": "active_false_negative",
    "cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection": "crosskd",
    "cvf:cvpr2024:Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large": "active_knowledge_aggregation",
    "cvf:cvpr2025:Li_SEEN-DA_SEmantic_ENtropy_guided_Domain-aware_Attention_for_Domain_Adaptive_Object": "seen_da",
    "cvf:cvpr2025:Liu_Distinguish_Then_Exploit_Source-free_Open_Set_Domain_Adaptation_via_Weight": "source_free_open_set",
    "cvf:iccv2021:Chen_Deep_Structured_Instance_Graph_for_Distilling_Object_Detectors": "structured_instance_graph",
    "cvf:iccv2021:Chen_Dual_Bipartite_Graph_Learning_A_General_Approach_for_Domain_Adaptive": "dual_bipartite_graph",
    "cvf:iccv2021:Tian_Knowledge_Mining_and_Transferring_for_Domain_Adaptive_Object_Detection": "knowledge_mining",
    "cvf:iccv2021:Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive": "gdetkd",
    "cvf:iccv2021:Yao_Multi-Source_Domain_Adaptation_for_Object_Detection": "multi_source",
    "cvf:iccv2023:Gao_CSDA_Learning_Category-Scale_Joint_Feature_for_Domain_Adaptive_Object_Detection": "csda",
    "cvf:iccv2023:Kang_Alleviating_Catastrophic_Forgetting_of_Incremental_Object_Detection_via_Within-Class_and": "incremental_within_class",
    "cvf:iccv2023:Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object": "unikd",
    "cvf:iccv2023:Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes": "spatial_self",
    "cvf:iccv2023:Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection": "cross_task_protocol",
    "cvf:iccv2023:Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection": "masked_retraining_teacher",
    "cvf:iccv2025:Cui_Debiased_Teacher_for_Day-to-Night_Domain_Adaptive_Object_Detection": "debiased_teacher",
    "cvf:iccv2025:He_Dual-Rate_Dynamic_Teacher_for_Source-Free_Domain_Adaptive_Object_Detection": "dual_rate_source_free",
    "papernotes:black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge": "black_box_retention",
    "papernotes:expert-teacher-student_collaborative_learning_for_domain_adaptive_object_detecti": "expert_teacher_student",
}


def paper_specific_mechanism_id(paper_id: str, family: str) -> str:
    """Return a non-generic mechanism ID for one paper."""
    slug = NAMED_METHOD_SLUGS.get(paper_id) or _fallback_slug(paper_id)
    return f"{family}.{slug}"


def load_certified_paper_recipe_specs(
    *,
    target_error_facts: list[dict[str, Any]] | None = None,
    has_runtime_adapter: bool = False,
) -> list[PaperRecipeSpec]:
    """Build one PaperRecipeSpec per certified paper."""
    facts = list(target_error_facts or [])
    specs: list[PaperRecipeSpec] = []
    for paper_id, mechanisms in CERTIFIED_PAPER_MECHANISMS.items():
        specs.append(
            build_paper_recipe_spec(
                paper_id,
                mechanisms=mechanisms,
                target_error_facts=facts,
                has_runtime_adapter=has_runtime_adapter,
            )
        )
    return specs


def build_paper_recipe_spec(
    paper_id: str,
    *,
    mechanisms: tuple[str, ...] | None = None,
    target_error_facts: list[dict[str, Any]] | None = None,
    has_runtime_adapter: bool = False,
    inference_only: bool = False,
) -> PaperRecipeSpec:
    """Build the explicit recipe binding for one certified paper."""
    mechanism_ids = tuple(mechanisms or CERTIFIED_PAPER_MECHANISMS[paper_id])
    existing = EXISTING_SPECIFIC_RECIPES.get(paper_id)
    family = _family(mechanism_ids)
    protocol = build_paper_protocol_contract(paper_id, mechanism_ids)
    facts = list(target_error_facts or [])
    if existing is not None:
        paper_ids = [paper_id, *list(existing.get("shared_with") or [])]
        recipe_id = str(existing["recipe_id"])
        specific = str(existing["paper_specific_mechanism_id"])
        components = list(existing["canonical_component_ids"])
        changed = dict(existing["changed_variables"])
        plugin = str(existing["runtime_plugin"])
        graph = str(existing.get("graph_identity") or "none")
    else:
        paper_ids = [paper_id]
        specific = paper_specific_mechanism_id(paper_id, family)
        recipe_id = f"paper_{_fallback_slug(paper_id)}"
        components = [specific]
        changed = _changed_variables(family, specific)
        plugin = specific
        graph = specific if family in {"model_graph", "domain_adaptation"} else "none"
    disposition = queue_disposition(
        target_error_facts=facts,
        inference_only=inference_only or family == "inference_only",
        has_runtime_adapter=has_runtime_adapter and existing is not None,
        incompatible=False,
    )
    teacher = "frozen_teacher" if family == "distillation" or protocol.teacher_requirement != "none" else "none"
    return PaperRecipeSpec(
        recipe_id=recipe_id,
        paper_ids=paper_ids,
        method_profile_ids=[f"method-profile-{_fallback_slug(paper_id)}"],
        paper_specific_mechanism_id=specific,
        canonical_component_ids=components,
        changed_variables=changed,
        runtime_plugin=plugin,
        protocol_hash=protocol.protocol_hash,
        required_evidence=list(protocol.required_evidence_artifacts),
        expected_metrics=list(protocol.required_metrics),
        stop_conditions=["pilot_no_gain", "latency_guard_regressed", "model_size_guard_regressed"],
        compatibility_requirements=["imgsz_640", "yolo26_one_to_one_head", "matched_protocol_hash"],
        target_error_facts=facts,
        inference_only=inference_only or family == "inference_only",
        teacher_identity=teacher,
        graph_identity=graph,
        disposition=disposition,
    )


def bindings_by_paper_id(
    specs: list[PaperRecipeSpec] | None = None,
) -> dict[str, PaperRecipeSpec]:
    """Index bindings by every listed paper_id."""
    index: dict[str, PaperRecipeSpec] = {}
    for spec in specs or load_certified_paper_recipe_specs():
        for paper_id in spec.paper_ids:
            if paper_id in CERTIFIED_PAPER_MECHANISMS:
                index[paper_id] = spec
    return index


def _family(mechanisms: tuple[str, ...]) -> str:
    ids = set(mechanisms)
    if any(item.startswith("inference.") for item in ids):
        return "inference"
    if "domain_adaptation.general" in ids:
        return "domain_adaptation"
    if any(item.startswith("distillation.") for item in ids):
        return "distillation"
    if any(item.startswith(("neck.", "assigner.", "detection_head.", "feature_pyramid.")) for item in ids):
        return "model_graph"
    if any(item.startswith("loss.") or item.startswith("quality_alignment.") for item in ids):
        return "loss"
    return "method"


def _changed_variables(family: str, specific: str) -> dict[str, Any]:
    if family == "domain_adaptation":
        return {"domain_adaptation.method": specific, "domain_alignment": specific.rsplit(".", 1)[-1]}
    if family == "distillation":
        return {"distillation.method": specific, "distillation.loss": specific.rsplit(".", 1)[-1]}
    if family == "loss":
        return {"loss.method": specific}
    return {"method": specific}


def _fallback_slug(paper_id: str) -> str:
    tail = paper_id.split(":")[-1]
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in tail)
    return "_".join(part for part in cleaned.split("_") if part)[:48]


def default_disposition_for_paper(paper_id: str) -> PaperRecipeDisposition:
    return build_paper_recipe_spec(paper_id).disposition
