"""Independent domain-adaptation branches.

``domain_adaptation.general`` is not an implementation. Each branch has its
own contract identity, payload, protocol, evidence, and failure disposition.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolContext,
    evaluate_paper_protocol,
)
from yolo_agent.research.paper_protocol_catalog import build_paper_protocol_contract
from yolo_agent.research.paper_protocol_ids import CERTIFIED_PAPER_MECHANISMS


CANONICAL_DOMAIN_BRANCHES = (
    "adversarial_alignment",
    "feature_alignment",
    "pseudo_label_adaptation",
    "domain_distillation",
    "source_free_adaptation",
    "cross_domain_teacher",
    "contrastive_domain_alignment",
    "active_domain_adaptation",
)

LEGACY_BRANCH_ALIASES: dict[str, str] = {
    "adversarial_feature_alignment": "adversarial_alignment",
    "class_conditional_alignment": "feature_alignment",
    "teacher_student_domain_adaptation": "cross_domain_teacher",
    "contrastive_alignment": "contrastive_domain_alignment",
    # These older routes remain explicit independent mechanisms. They are not
    # silently redirected by substring matching.
    "cross_domain_graph_alignment": "feature_alignment",
    "domain_calibration": "feature_alignment",
    "domain_specific_normalization": "feature_alignment",
}

BASE_PAYLOAD_SCHEMA: dict[str, str] = {
    "branch_id": "str",
    "weight": "float",
    "source_domain_id": "str",
    "target_domain_id": "str",
    "source_manifest": "str",
    "target_manifest": "str",
    "source_manifest_sha256": "str",
    "target_manifest_sha256": "str",
    "source_dataset_hash": "str",
    "target_dataset_hash": "str",
    "source_split": "str",
    "target_split": "str",
    "domain_pair_id": "str",
    "domain_protocol_hash": "str",
    "adaptation_mode": "str",
    "label_availability": "str",
    "evidence_artifact": "str",
    "imgsz": "int",
}

DOMAIN_BRANCH_PROFILES: dict[str, dict[str, object]] = {
    "adversarial_alignment": {
        "changed_variable": "loss.domain_adversarial_alignment.weight",
        "payload_schema": {"discriminator_layers": "int", "gradient_reversal": "bool"},
        "evidence_artifact": "domain_adversarial_alignment_evidence.json",
        "adaptation_mode": "unsupervised",
        "runtime_strategy": "adversarial_discriminator",
        "required_label_availability": "source_labeled_target_unlabeled",
        "required_evidence": ["source_target_manifest", "domain_pair", "discriminator_contract", "matched_control"],
        "requires_source_domain": True,
    },
    "feature_alignment": {
        "changed_variable": "loss.domain_feature_alignment.weight",
        "payload_schema": {"feature_levels": "list[int]", "align_variance": "bool"},
        "evidence_artifact": "domain_feature_alignment_evidence.json",
        "adaptation_mode": "unsupervised",
        "runtime_strategy": "feature_statistics_alignment",
        "required_label_availability": "source_labeled_target_unlabeled",
        "required_evidence": ["source_target_manifest", "domain_pair", "feature_levels", "matched_control"],
        "requires_source_domain": True,
    },
    "pseudo_label_adaptation": {
        "changed_variable": "loss.domain_pseudo_label_adaptation.weight",
        "payload_schema": {"pseudo_label_manifest": "str", "confidence_threshold": "float"},
        "evidence_artifact": "domain_pseudo_label_adaptation_evidence.json",
        "adaptation_mode": "semi_supervised",
        "runtime_strategy": "target_pseudo_label_consistency",
        "required_label_availability": "pseudo",
        "required_evidence": ["source_target_manifest", "domain_pair", "target_pseudo_label_manifest", "matched_control"],
        "requires_source_domain": True,
    },
    "domain_distillation": {
        "changed_variable": "loss.domain_distillation.weight",
        "payload_schema": {"teacher_checkpoint": "str", "teacher_sha256": "str", "teacher_mode": "str"},
        "evidence_artifact": "domain_distillation_evidence.json",
        "adaptation_mode": "unsupervised",
        "runtime_strategy": "domain_teacher_distillation",
        "required_label_availability": "source_labeled_target_unlabeled",
        "required_evidence": ["source_target_manifest", "domain_pair", "frozen_teacher_checkpoint", "matched_control"],
        "requires_source_domain": True,
    },
    "source_free_adaptation": {
        "changed_variable": "loss.domain_source_free_adaptation.weight",
        "payload_schema": {"source_model_checkpoint": "str", "source_model_sha256": "str", "target_only": "bool"},
        "evidence_artifact": "domain_source_free_adaptation_evidence.json",
        "adaptation_mode": "source_free",
        "runtime_strategy": "source_free_target_adaptation",
        "required_label_availability": "unlabeled",
        "required_evidence": ["target_manifest", "source_trained_checkpoint", "source_model_protocol_hash", "matched_control"],
        "requires_source_domain": False,
    },
    "cross_domain_teacher": {
        "changed_variable": "loss.domain_cross_domain_teacher.weight",
        "payload_schema": {"teacher_checkpoint": "str", "teacher_sha256": "str", "teacher_domain_id": "str"},
        "evidence_artifact": "domain_cross_domain_teacher_evidence.json",
        "adaptation_mode": "unsupervised",
        "runtime_strategy": "cross_domain_teacher",
        "required_label_availability": "source_labeled_target_unlabeled",
        "required_evidence": ["source_target_manifest", "domain_pair", "frozen_cross_domain_teacher", "matched_control"],
        "requires_source_domain": True,
    },
    "contrastive_domain_alignment": {
        "changed_variable": "loss.domain_contrastive_alignment.weight",
        "payload_schema": {"temperature": "float", "pairing_key": "str"},
        "evidence_artifact": "domain_contrastive_alignment_evidence.json",
        "adaptation_mode": "unsupervised",
        "runtime_strategy": "cross_domain_contrastive",
        "required_label_availability": "source_labeled_target_unlabeled",
        "required_evidence": ["source_target_manifest", "domain_pair", "contrastive_pair_manifest", "matched_control"],
        "requires_source_domain": True,
    },
    "active_domain_adaptation": {
        "changed_variable": "loss.domain_active_adaptation.weight",
        "payload_schema": {"query_manifest": "str", "label_budget": "int", "query_strategy": "str"},
        "evidence_artifact": "domain_active_adaptation_evidence.json",
        "adaptation_mode": "active",
        "runtime_strategy": "active_query_selection",
        "required_label_availability": "partial",
        "required_evidence": ["source_target_manifest", "domain_pair", "active_query_manifest", "matched_control"],
        "requires_source_domain": True,
    },
}


def canonical_branch_id(branch_id: str) -> str:
    """Resolve an explicitly registered legacy ID; unknown IDs fail closed."""
    resolved = LEGACY_BRANCH_ALIASES.get(branch_id, branch_id)
    if resolved not in CANONICAL_DOMAIN_BRANCHES:
        raise DomainProtocolError(f"unknown domain adaptation branch: {branch_id}")
    return resolved


DomainAdaptationBranchId = Literal[
    "adversarial_alignment",
    "feature_alignment",
    "pseudo_label_adaptation",
    "domain_distillation",
    "source_free_adaptation",
    "cross_domain_teacher",
    "contrastive_domain_alignment",
    "active_domain_adaptation",
    "adversarial_feature_alignment",
    "class_conditional_alignment",
    "teacher_student_domain_adaptation",
    "contrastive_alignment",
    "cross_domain_graph_alignment",
    "domain_calibration",
    "domain_specific_normalization",
]
DomainDisposition = Literal[
    "candidate",
    "evidence_recovery",
    "incompatible",
    "implementation_request",
]


class DomainProtocolError(ValueError):
    """Raised when a domain-adaptation protocol is unsafe or incomplete."""


class DomainAdaptationBranchSpec(BaseModel, YAMLModelMixin):
    model_config = ConfigDict(extra="forbid")

    branch_id: DomainAdaptationBranchId
    component_id: str
    adapter_class: str
    plugin_class: str
    changed_variable: str
    payload_schema: dict[str, str]
    evidence_artifact: str
    adaptation_mode: Literal["unsupervised", "semi_supervised", "source_free", "active"] = "unsupervised"
    runtime_strategy: str
    required_label_availability: str
    required_evidence: list[str] = Field(default_factory=list)
    legacy_aliases: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    requires_source_domain: bool = True
    requires_target_domain: bool = True
    coco_as_domain_allowed: Literal[False] = False
    adapter_alone_authorizes_asha: Literal[False] = False
    contaminates_coco_baseline: Literal[False] = False
    execution_fingerprint: str = ""

    @field_validator("branch_id", "component_id", "changed_variable", "evidence_artifact")
    @classmethod
    def _required(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("domain adaptation branch identity must not be empty")
        return value

    @model_validator(mode="after")
    def bind_fingerprint(self) -> "DomainAdaptationBranchSpec":
        if self.coco_as_domain_allowed:
            raise DomainProtocolError("COCO cannot stand in for a paper domain protocol")
        if self.adapter_alone_authorizes_asha:
            raise DomainProtocolError("an adapter cannot by itself authorize ASHA")
        if self.contaminates_coco_baseline:
            raise DomainProtocolError("domain adaptation must not contaminate the COCO baseline")
        digest = _fingerprint(self)
        if self.execution_fingerprint and self.execution_fingerprint != digest:
            raise DomainProtocolError(f"domain branch fingerprint mismatch: {self.branch_id}")
        self.execution_fingerprint = digest
        return self

    @property
    def canonical_branch_id(self) -> str:
        return LEGACY_BRANCH_ALIASES.get(str(self.branch_id), str(self.branch_id))


class DomainPaperAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    branch_id: DomainAdaptationBranchId
    disposition: DomainDisposition
    reason_codes: list[str] = Field(default_factory=list)
    missing_dataset_actions: list[str] = Field(default_factory=list)
    allows_asha: bool = False
    execution_fingerprint: str = ""


class DomainPaperCoverage(BaseModel, YAMLModelMixin):
    schema_version: str = "domain_adaptation_paper_coverage.v1"
    papers_total: int
    candidate: int = 0
    evidence_recovery: int = 0
    incompatible: int = 0
    implementation_request: int = 0
    assignments: list[DomainPaperAssignment]
    silent_drops: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_silent_drop(self) -> "DomainPaperCoverage":
        if self.silent_drops:
            raise ValueError(f"domain coverage silent drops: {self.silent_drops}")
        counted = (
            self.candidate
            + self.evidence_recovery
            + self.incompatible
            + self.implementation_request
        )
        if counted != self.papers_total:
            raise ValueError("every domain paper must have a terminal disposition")
        return self


NAMED_PAPER_BRANCHES: dict[str, DomainAdaptationBranchId] = {
    "arxiv:2210.11539": "adversarial_alignment",
    "arxiv:2303.13853": "feature_alignment",
    "arxiv:2503.23220": "pseudo_label_adaptation",
    "arxiv:2507.00721": "feature_alignment",
    "arxiv:2603.12409": "feature_alignment",
    "arxiv:2603.18541": "contrastive_domain_alignment",
    "arxiv:2603.18757": "feature_alignment",
    "arxiv:2603.28182": "active_domain_adaptation",
    "cvf:cvpr2021:VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object": "feature_alignment",
    "cvf:cvpr2021:Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector": "adversarial_alignment",
    "cvf:cvpr2022:Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection": "cross_domain_teacher",
    "cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection": "feature_alignment",
    "cvf:cvpr2022:Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection": "feature_alignment",
    "cvf:cvpr2022:Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection": "adversarial_alignment",
    "cvf:cvpr2022:Zhou_Multi-Granularity_Alignment_Domain_Adaptation_for_Object_Detection": "feature_alignment",
    "cvf:cvpr2023:Cao_Contrastive_Mean_Teacher_for_Domain_Adaptive_Object_Detectors": "contrastive_domain_alignment",
    "cvf:cvpr2023:Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object": "pseudo_label_adaptation",
    "cvf:cvpr2023:Liu_CIGAR_Cross-Modality_Graph_Reasoning_for_Domain_Adaptive_Object_Detection": "feature_alignment",
    "cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection": "source_free_adaptation",
    "cvf:cvpr2024:Du_Boosting_Object_Detection_with_Zero-Shot_Day-Night_Domain_Adaptation": "feature_alignment",
    "cvf:cvpr2024:Kennerley_CAT_Exploiting_Inter-Class_Dynamics_for_Domain_Adaptive_Object_Detection": "feature_alignment",
    "cvf:cvpr2024:Nakamura_Active_Domain_Adaptation_with_False_Negative_Prediction_for_Object_Detection": "active_domain_adaptation",
    "cvf:cvpr2025:Li_SEEN-DA_SEmantic_ENtropy_guided_Domain-aware_Attention_for_Domain_Adaptive_Object": "feature_alignment",
    "cvf:cvpr2025:Liu_Distinguish_Then_Exploit_Source-free_Open_Set_Domain_Adaptation_via_Weight": "source_free_adaptation",
    "cvf:iccv2021:Chen_Dual_Bipartite_Graph_Learning_A_General_Approach_for_Domain_Adaptive": "feature_alignment",
    "cvf:iccv2021:Tian_Knowledge_Mining_and_Transferring_for_Domain_Adaptive_Object_Detection": "cross_domain_teacher",
    "cvf:iccv2021:Yao_Multi-Source_Domain_Adaptation_for_Object_Detection": "feature_alignment",
    "cvf:iccv2023:Gao_CSDA_Learning_Category-Scale_Joint_Feature_for_Domain_Adaptive_Object_Detection": "feature_alignment",
    "cvf:iccv2023:Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection": "cross_domain_teacher",
    "cvf:iccv2025:Cui_Debiased_Teacher_for_Day-to-Night_Domain_Adaptive_Object_Detection": "cross_domain_teacher",
    "cvf:iccv2025:He_Dual-Rate_Dynamic_Teacher_for_Source-Free_Domain_Adaptive_Object_Detection": "source_free_adaptation",
    "ecva:eccv2022:3958": "adversarial_alignment",
    "ecva:eccv2024:11254": "cross_domain_teacher",
    "ecva:eccv2024:7083": "feature_alignment",
    "neurips:2021:c0cccc24dd23ded67404f5e511c342b0-Abstract": "contrastive_domain_alignment",
    "neurips:2024:6b6492cd06db22bac024506e9ed0925e-Abstract-Conference": "pseudo_label_adaptation",
    "neurips:2024:89d0d5c2f720921df93bbb8fef514571-Abstract-Conference": "active_domain_adaptation",
    "neurips:2024:bb71b5567ee985e0a4cee54ade19275c-Abstract-Conference": "feature_alignment",
    "papernotes:black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge": "source_free_adaptation",
    "papernotes:expert-teacher-student_collaborative_learning_for_domain_adaptive_object_detecti": "cross_domain_teacher",
}


def certified_domain_adaptation_papers() -> list[str]:
    return [
        paper_id
        for paper_id, mechanisms in CERTIFIED_PAPER_MECHANISMS.items()
        if "domain_adaptation.general" in mechanisms
    ]


def build_branch(branch_id: DomainAdaptationBranchId, paper_ids: list[str] | None = None) -> DomainAdaptationBranchSpec:
    canonical = canonical_branch_id(branch_id)
    profile = DOMAIN_BRANCH_PROFILES[canonical]
    component_id = f"domain_adaptation.{canonical}"
    return DomainAdaptationBranchSpec(
        branch_id=canonical,  # type: ignore[arg-type]
        component_id=component_id,
        adapter_class="DomainAdaptationBranchAdapter",
        plugin_class="DomainAdaptationBranchPlugin",
        changed_variable=profile["changed_variable"],
        payload_schema={**BASE_PAYLOAD_SCHEMA, **profile["payload_schema"]},
        evidence_artifact=profile["evidence_artifact"],
        adaptation_mode=profile["adaptation_mode"],
        runtime_strategy=profile["runtime_strategy"],
        required_label_availability=profile["required_label_availability"],
        required_evidence=list(profile["required_evidence"]),
        legacy_aliases=[alias for alias, target in LEGACY_BRANCH_ALIASES.items() if target == canonical],
        paper_ids=list(paper_ids or []),
        requires_source_domain=profile["requires_source_domain"],
        requires_target_domain=True,
    )


class DomainAdaptationMethodRegistry:
    def __init__(self) -> None:
        papers: dict[str, list[str]] = {item: [] for item in CANONICAL_DOMAIN_BRANCHES}
        for paper_id, branch_id in NAMED_PAPER_BRANCHES.items():
            papers[canonical_branch_id(branch_id)].append(paper_id)
        self._branches = {
            branch_id: build_branch(branch_id, ids) for branch_id, ids in papers.items()
        }

    def get(self, branch_id: DomainAdaptationBranchId) -> DomainAdaptationBranchSpec:
        return self._branches[canonical_branch_id(str(branch_id))]

    def branches(self) -> list[DomainAdaptationBranchSpec]:
        return list(self._branches.values())

    def assign(
        self,
        paper_id: str,
        context: PaperProtocolContext | None = None,
    ) -> DomainPaperAssignment:
        branch_id = NAMED_PAPER_BRANCHES.get(paper_id)
        if branch_id is None:
            return DomainPaperAssignment(
                paper_id=paper_id,
                branch_id="adversarial_feature_alignment",
                disposition="implementation_request",
                reason_codes=["domain_branch_unmapped"],
                allows_asha=False,
            )
        branch = self.get(branch_id)
        facts = context or PaperProtocolContext()
        protocol = build_paper_protocol_contract(paper_id)
        evaluation = evaluate_paper_protocol(protocol, facts)
        disposition: DomainDisposition
        if evaluation.disposition == "incompatible":
            disposition = "incompatible"
        elif evaluation.disposition == "evidence_recovery":
            disposition = "evidence_recovery"
        elif evaluation.ok:
            disposition = "candidate"
        else:
            disposition = "implementation_request"
        return DomainPaperAssignment(
            paper_id=paper_id,
            branch_id=branch_id,
            disposition=disposition,
            reason_codes=list(evaluation.reason_codes),
            missing_dataset_actions=list(evaluation.missing_dataset_actions),
            allows_asha=disposition == "candidate" and not branch.adapter_alone_authorizes_asha and evaluation.allows_asha_registration,
            execution_fingerprint=branch.execution_fingerprint,
        )

    def coverage(
        self,
        context: PaperProtocolContext | None = None,
        paper_ids: list[str] | None = None,
    ) -> DomainPaperCoverage:
        ids = list(paper_ids or certified_domain_adaptation_papers())
        assignments = [self.assign(paper_id, context) for paper_id in ids]
        found = {item.paper_id for item in assignments}
        return DomainPaperCoverage(
            papers_total=len(ids),
            candidate=sum(item.disposition == "candidate" for item in assignments),
            evidence_recovery=sum(item.disposition == "evidence_recovery" for item in assignments),
            incompatible=sum(item.disposition == "incompatible" for item in assignments),
            implementation_request=sum(
                item.disposition == "implementation_request" for item in assignments
            ),
            assignments=assignments,
            silent_drops=[paper_id for paper_id in ids if paper_id not in found],
        )


def default_domain_adaptation_registry() -> DomainAdaptationMethodRegistry:
    return DomainAdaptationMethodRegistry()


def _fingerprint(branch: DomainAdaptationBranchSpec) -> str:
    payload = branch.model_dump(mode="json", exclude={"execution_fingerprint", "paper_ids"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
