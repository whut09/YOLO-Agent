"""Paper-facing distillation method registry.

A generic teacher-student loss cannot cover every distillation paper. Each
branch has its own variables, payload, loss mode, evidence, and fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yolo_agent.components.distillation.mechanism_losses import (
    DistillationMechanismLoss,
    build_distillation_mechanism_loss,
)
from yolo_agent.components.distillation.mechanisms import (
    DISTILLATION_MECHANISMS,
    DistillationMechanism,
)
from yolo_agent.core.yaml_io import YAMLModelMixin


DistillationBranchId = Literal[
    "logits_distillation",
    "feature_distillation",
    "relation_distillation",
    "localization_distillation",
    "attention_distillation",
    "masked_feature_distillation",
    "quality_aware_distillation",
    "teacher_ensemble",
    "source_free_teacher",
    "cross_domain_teacher",
    "contrastive_distillation",
]
PaperAssignmentDisposition = Literal[
    "assigned",
    "implementation_request",
]


class DistillationTeacherMissingError(FileNotFoundError):
    """Raised when a distillation branch is missing a bound teacher checkpoint."""


class DistillationBranchSpec(BaseModel, YAMLModelMixin):
    """One independent distillation branch."""

    model_config = ConfigDict(extra="forbid")

    branch_id: DistillationBranchId
    mechanism: DistillationMechanism
    component_id: str
    changed_variables: dict[str, Any]
    loss_mode: str
    evidence_schema: list[str]
    runtime_payload_schema: dict[str, str]
    paper_specific_configuration: dict[str, Any] = Field(default_factory=dict)
    paper_ids: list[str] = Field(default_factory=list)
    teacher_protocol: dict[str, Any]
    student_protocol: dict[str, Any]
    export_teacher: Literal[False] = False
    measure_student_only: Literal[True] = True
    allow_dfl: Literal[False] = False
    allow_head_replacement: Literal[False] = False
    requires_teacher_checkpoint: Literal[True] = True
    execution_fingerprint: str = ""

    @field_validator("branch_id", "mechanism", "component_id", "loss_mode")
    @classmethod
    def _required(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("distillation branch identity must not be empty")
        return value

    @model_validator(mode="after")
    def bind_fingerprint(self) -> "DistillationBranchSpec":
        if self.export_teacher:
            raise ValueError("distillation branches must not export the teacher")
        if not self.measure_student_only:
            raise ValueError("latency and model-size must measure the student only")
        if self.allow_dfl or self.allow_head_replacement:
            raise ValueError("YOLO26 DFL and native head replacement are forbidden")
        digest = compute_branch_fingerprint(self)
        if self.execution_fingerprint and self.execution_fingerprint != digest:
            raise ValueError(f"branch fingerprint mismatch: {self.branch_id}")
        self.execution_fingerprint = digest
        return self

    def build_loss(self, **options: Any) -> DistillationMechanismLoss:
        return build_distillation_mechanism_loss(self.mechanism, **options)

    def require_teacher_checkpoint(self, path: str | None) -> str:
        if not path or not str(path).strip():
            raise DistillationTeacherMissingError(
                f"{self.branch_id} requires a local teacher checkpoint; silent skip is forbidden"
            )
        return str(path)


class DistillationPaperAssignment(BaseModel):
    """One certified distillation paper mapped to a branch or blocker."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    disposition: PaperAssignmentDisposition
    branch_id: DistillationBranchId | None = None
    reason_codes: list[str] = Field(default_factory=list)
    execution_fingerprint: str = ""


class DistillationPaperCoverage(BaseModel, YAMLModelMixin):
    schema_version: str = "distillation_paper_coverage.v1"
    papers_total: int
    assigned: int
    implementation_request: int
    assignments: list[DistillationPaperAssignment]
    silent_drops: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_silent_drop(self) -> "DistillationPaperCoverage":
        if self.silent_drops:
            raise ValueError(f"distillation coverage silent drops: {self.silent_drops}")
        if self.papers_total != self.assigned + self.implementation_request:
            raise ValueError("every distillation paper must be assigned or blocked")
        return self


BRANCH_TO_MECHANISM: dict[DistillationBranchId, DistillationMechanism] = {
    "logits_distillation": "logits",
    "feature_distillation": "feature",
    "relation_distillation": "relation",
    "localization_distillation": "localization",
    "attention_distillation": "attention",
    "masked_feature_distillation": "masked_feature",
    "quality_aware_distillation": "quality_aware",
    "teacher_ensemble": "teacher_ensemble",
    "source_free_teacher": "source_free_teacher",
    "cross_domain_teacher": "cross_domain_teacher",
    "contrastive_distillation": "contrastive",
}

NAMED_PAPER_BRANCHES: dict[str, DistillationBranchId] = {
    "cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection": "relation_distillation",
    "cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features": "feature_distillation",
    "cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection": "relation_distillation",
    "cvf:cvpr2022:Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response": "quality_aware_distillation",
    "cvf:cvpr2022:Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection": "feature_distillation",
    "cvf:cvpr2022:He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation": "cross_domain_teacher",
    "cvf:cvpr2022:Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation": "feature_distillation",
    "cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection": "localization_distillation",
    "cvf:cvpr2023:Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection": "feature_distillation",
    "cvf:cvpr2023:Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector": "feature_distillation",
    "cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection": "logits_distillation",
    "cvf:cvpr2024:Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large": "teacher_ensemble",
    "cvf:iccv2021:Chen_Deep_Structured_Instance_Graph_for_Distilling_Object_Detectors": "relation_distillation",
    "cvf:iccv2021:Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive": "contrastive_distillation",
    "cvf:iccv2023:Kang_Alleviating_Catastrophic_Forgetting_of_Incremental_Object_Detection_via_Within-Class_and": "logits_distillation",
    "cvf:iccv2023:Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object": "logits_distillation",
    "cvf:iccv2023:Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes": "attention_distillation",
    "cvf:iccv2023:Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection": "logits_distillation",
}

CERTIFIED_DISTILLATION_PAPERS = (
    "cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection",
    "cvf:cvpr2021:Guo_Distilling_Object_Detectors_via_Decoupled_Features",
    "cvf:cvpr2021:Hu_Dense_Relation_Distillation_With_Context-Aware_Aggregation_for_Few-Shot_Object_Detection",
    "cvf:cvpr2022:Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response",
    "cvf:cvpr2022:Guo_Scale-Equivalent_Distillation_for_Semi-Supervised_Object_Detection",
    "cvf:cvpr2022:He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation",
    "cvf:cvpr2022:Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation",
    "cvf:cvpr2022:Zheng_Localization_Distillation_for_Dense_Object_Detection",
    "cvf:cvpr2023:Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection",
    "cvf:cvpr2023:Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector",
    "cvf:cvpr2024:Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection",
    "cvf:cvpr2024:Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large",
    "cvf:iccv2021:Chen_Deep_Structured_Instance_Graph_for_Distilling_Object_Detectors",
    "cvf:iccv2021:Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive",
    "cvf:iccv2023:Kang_Alleviating_Catastrophic_Forgetting_of_Incremental_Object_Detection_via_Within-Class_and",
    "cvf:iccv2023:Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object",
    "cvf:iccv2023:Wu_Spatial_Self-Distillation_for_Object_Detection_with_Inaccurate_Bounding_Boxes",
    "cvf:iccv2023:Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection",
    "ecva:eccv2022:1356",
    "ecva:eccv2022:2285",
    "ecva:eccv2022:2717",
    "ecva:eccv2022:3523",
    "ecva:eccv2022:6004",
    "ecva:eccv2022:6328",
    "ecva:eccv2024:11200",
    "ecva:eccv2024:6619",
    "neurips:2021:082a8bbf2c357c09f26675f9cf5bcba3-Abstract",
    "neurips:2021:29c0c0ee223856f336d7ea8052057753-Abstract",
    "neurips:2021:892c91e0a653ba19df81a90f89d99bcd-Abstract",
    "neurips:2022:18c0102cb7f1a02c14f0929089b2e576-Abstract-Conference",
    "neurips:2022:631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference",
    "neurips:2025:6460e378f24da3a79f20ac2640732a00-Abstract-Conference",
)


def compute_branch_fingerprint(branch: DistillationBranchSpec) -> str:
    payload = {
        "branch_id": branch.branch_id,
        "mechanism": branch.mechanism,
        "changed_variables": branch.changed_variables,
        "loss_mode": branch.loss_mode,
        "evidence_schema": branch.evidence_schema,
        "runtime_payload_schema": branch.runtime_payload_schema,
        "paper_specific_configuration": branch.paper_specific_configuration,
        "teacher_protocol": branch.teacher_protocol,
        "student_protocol": branch.student_protocol,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _shared_teacher_protocol() -> dict[str, Any]:
    return {
        "frozen": True,
        "checkpoint_required": True,
        "sha256_required": True,
        "export_forbidden": True,
        "allowed_names": ["yolo26s.pt", "yolo26m.pt"],
    }


def _shared_student_protocol() -> dict[str, Any]:
    return {
        "checkpoint": "yolo26n.pt",
        "imgsz": 640,
        "native_dfl_free": True,
        "native_one_to_one_head": True,
        "export_and_measure": "student_only",
    }


def build_branch(
    branch_id: DistillationBranchId,
    paper_ids: list[str] | None = None,
) -> DistillationBranchSpec:
    mechanism = BRANCH_TO_MECHANISM[branch_id]
    spec = DISTILLATION_MECHANISMS[mechanism]
    signal_type = _signal_type(branch_id)
    return DistillationBranchSpec(
        branch_id=branch_id,
        mechanism=mechanism,
        component_id=spec.component_id,
        changed_variables={
            spec.changed_variable: 1.0,
            "distillation.branch": branch_id,
            "distillation.loss_mode": branch_id,
        },
        loss_mode=branch_id,
        evidence_schema=[
            "teacher_checkpoint",
            "teacher_checkpoint_sha256",
            "teacher_student_same_split",
            "student_only_evaluation",
            f"distillation_branch:{branch_id}",
        ],
        runtime_payload_schema={
            "branch_id": "str",
            "mechanism": "str",
            "loss_mode": branch_id,
            "teacher": "path",
            "student": "path",
            "teacher_sha256": "sha256",
            "student_sha256": "sha256",
            "teacher_architecture": "str",
            "student_architecture": "yolo26n",
            "teacher_split": "str",
            "student_split": "str",
            "imgsz": "640",
            "weight": "float",
            "teacher_signal": signal_type,
            "student_signal": signal_type,
        },
        paper_specific_configuration={
            "branch_id": branch_id,
            "loss_mode": branch_id,
            "signal_type": signal_type,
            "student": "yolo26n",
            "imgsz": 640,
        },
        paper_ids=list(paper_ids or []),
        teacher_protocol=_shared_teacher_protocol(),
        student_protocol=_shared_student_protocol(),
    )


def _signal_type(branch_id: DistillationBranchId) -> str:
    if branch_id == "teacher_ensemble":
        return "teacher_response_tensor_list"
    if branch_id == "localization_distillation":
        return "native_dfl_free_box_tensor"
    if branch_id in {
        "feature_distillation",
        "relation_distillation",
        "attention_distillation",
        "masked_feature_distillation",
        "contrastive_distillation",
    }:
        return "intermediate_feature_tensor_list"
    return "response_logits_tensor"


class DistillationMethodRegistry:
    """Registry of independent distillation branches and paper assignments."""

    def __init__(self) -> None:
        papers_by_branch: dict[DistillationBranchId, list[str]] = {
            branch_id: [] for branch_id in BRANCH_TO_MECHANISM
        }
        for paper_id, branch_id in NAMED_PAPER_BRANCHES.items():
            papers_by_branch[branch_id].append(paper_id)
        self._branches = {
            branch_id: build_branch(branch_id, papers)
            for branch_id, papers in papers_by_branch.items()
        }

    def get(self, branch_id: DistillationBranchId) -> DistillationBranchSpec:
        return self._branches[branch_id]

    def branches(self) -> list[DistillationBranchSpec]:
        return [self._branches[item] for item in BRANCH_TO_MECHANISM]

    def assign(self, paper_id: str) -> DistillationPaperAssignment:
        branch_id = NAMED_PAPER_BRANCHES.get(paper_id)
        if branch_id is None:
            return DistillationPaperAssignment(
                paper_id=paper_id,
                disposition="implementation_request",
                reason_codes=["distillation_branch_unmapped", "paper_method_identity_missing"],
            )
        branch = self.get(branch_id)
        return DistillationPaperAssignment(
            paper_id=paper_id,
            disposition="assigned",
            branch_id=branch_id,
            execution_fingerprint=branch.execution_fingerprint,
        )

    def coverage(
        self,
        paper_ids: tuple[str, ...] = CERTIFIED_DISTILLATION_PAPERS,
    ) -> DistillationPaperCoverage:
        assignments = [self.assign(paper_id) for paper_id in paper_ids]
        found = {item.paper_id for item in assignments}
        return DistillationPaperCoverage(
            papers_total=len(paper_ids),
            assigned=sum(item.disposition == "assigned" for item in assignments),
            implementation_request=sum(
                item.disposition == "implementation_request" for item in assignments
            ),
            assignments=assignments,
            silent_drops=[paper_id for paper_id in paper_ids if paper_id not in found],
        )


def default_distillation_method_registry() -> DistillationMethodRegistry:
    return DistillationMethodRegistry()
