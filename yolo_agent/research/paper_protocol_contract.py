"""Paper-level execution protocol contracts.

ComponentContract describes a reusable adapter. PaperProtocolContract binds one
paper to the dataset, split, teacher, graph, and evidence rules that must hold
before materialization or ASHA registration. A missing protocol is a hard stop.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.paper_evidence_requirements import (
    DOMAIN_DATASET_ACTIONS,
    missing_dataset_actions,
)


DatasetRole = Literal[
    "coco_standard",
    "source_target_domain",
    "custom_labeled",
    "unlabeled_target",
    "unknown",
]
SplitRole = Literal[
    "coco_train",
    "coco_val",
    "coco_test",
    "source_train",
    "source_val",
    "target_train",
    "target_val",
    "target_test",
    "unknown",
]
AnnotationRequirement = Literal[
    "fully_labeled",
    "source_labeled_target_unlabeled",
    "source_free",
    "weakly_labeled",
    "none",
]
TrainValTestProtocol = Literal[
    "coco_official",
    "source_target_domain",
    "teacher_student_same_split",
    "inference_only",
    "unknown",
]
ModelFamily = Literal[
    "yolo26",
    "yolo11",
    "generic_detector",
    "separate_detector_family",
]
HeadConstraint = Literal[
    "yolo26_one_to_one",
    "native_dfl_free",
    "generic",
    "incompatible",
]
TeacherRequirement = Literal[
    "none",
    "frozen_teacher_checkpoint",
    "mean_teacher",
    "cross_domain_teacher",
]
ChangeKind = Literal["none", "required", "optional"]
ProtocolFamily = Literal[
    "domain_adaptation",
    "distillation",
    "model_graph",
    "inference_only",
    "standard_training",
]
ProtocolDisposition = Literal[
    "queued",
    "evidence_recovery",
    "implementation_request",
    "incompatible",
    "blocked_runtime",
]
ExecutionClass = Literal[
    "pilot_candidate",
    "inference_candidate",
    "blocked",
]
COCO_SPLITS = {"coco_train", "coco_val", "coco_test"}
SCHEMA_VERSION = "paper_protocol_contract.v1"


class PaperProtocolError(ValueError):
    """Raised when a paper protocol is missing or cannot authorize execution."""


class PaperProtocolContext(BaseModel):
    """Runtime facts used to evaluate one paper protocol."""

    model_config = ConfigDict(extra="forbid")

    has_source_domain_data: bool = False
    has_target_domain_data: bool = False
    coco_train_used_as_source: bool = False
    coco_val_used_as_target: bool = False
    teacher_checkpoint_exists: bool = False
    teacher_sha256: str | None = None
    teacher_dataset_manifest: str | None = None
    student_dataset_manifest: str | None = None
    teacher_split: str | None = None
    student_split: str | None = None
    evaluate_teacher: bool = False
    declared_graph_identity: str | None = None
    asha_track: Literal["training", "inference", "none"] = "training"
    component_ids: list[str] = Field(default_factory=list)


class PaperProtocolEvaluation(BaseModel):
    """Deterministic evaluation of one paper protocol against runtime facts."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    protocol_hash: str
    ok: bool
    disposition: ProtocolDisposition
    execution_class: ExecutionClass
    reason_codes: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    missing_dataset_actions: list[str] = Field(default_factory=list)
    allows_materialization: bool = False
    allows_asha_registration: bool = False

    @model_validator(mode="after")
    def validate_boundary(self) -> "PaperProtocolEvaluation":
        if self.ok and not (self.allows_materialization and self.allows_asha_registration):
            raise ValueError("ok protocol evaluation must authorize materialization and ASHA")
        if self.execution_class == "inference_candidate" and self.allows_asha_registration:
            raise ValueError("inference_candidate cannot register with training ASHA")
        if not self.ok and not self.reason_codes:
            raise ValueError("failed protocol evaluation requires reason_codes")
        if self.disposition == "evidence_recovery" and not self.required_evidence:
            raise ValueError("evidence_recovery requires required_evidence")
        return self


class PaperProtocolContract(BaseModel, YAMLModelMixin):
    """Explicit execution protocol for one paper, independent of ComponentContract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    paper_id: str
    dataset_role: DatasetRole
    source_split: SplitRole
    target_split: SplitRole
    annotation_requirement: AnnotationRequirement
    train_val_test_protocol: TrainValTestProtocol
    imgsz: int = 640
    model_family: ModelFamily
    head_constraint: HeadConstraint
    teacher_requirement: TeacherRequirement
    graph_change: ChangeKind
    loss_change: ChangeKind
    inference_change: ChangeKind
    required_metrics: list[str] = Field(min_length=1)
    paired_baseline_requirement: bool = True
    required_evidence_artifacts: list[str] = Field(min_length=1)
    protocol_family: ProtocolFamily
    graph_identity: str | None = None
    yolo26_one_to_one_head: bool = True
    native_dfl_free_regression: bool = True
    mechanism_ids: list[str] = Field(default_factory=list)
    protocol_hash: str = ""

    @field_validator("paper_id")
    @classmethod
    def _require_paper_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("paper protocol requires paper_id")
        return value.strip()

    @field_validator("imgsz")
    @classmethod
    def _require_imgsz_640(cls, value: int) -> int:
        if value != 640:
            raise ValueError("paper protocol imgsz must be 640")
        return value

    @field_validator("required_metrics", "required_evidence_artifacts")
    @classmethod
    def _require_non_empty_items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("paper protocol lists must not be empty")
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def bind_protocol_hash(self) -> "PaperProtocolContract":
        if self.graph_change == "required" and not (self.graph_identity or "").strip():
            raise ValueError("model-graph protocol must declare graph_identity")
        if self.inference_change == "required" and self.protocol_family != "inference_only":
            raise ValueError("inference-only change must use inference_only protocol family")
        digest = compute_paper_protocol_hash(self)
        if self.protocol_hash and self.protocol_hash != digest:
            raise ValueError(
                f"protocol_hash mismatch for {self.paper_id}: "
                f"{self.protocol_hash} != {digest}"
            )
        self.protocol_hash = digest
        return self

    @property
    def is_domain_adaptation(self) -> bool:
        return self.protocol_family == "domain_adaptation" or self.dataset_role == "source_target_domain"

    @property
    def is_distillation(self) -> bool:
        return self.protocol_family == "distillation" or self.teacher_requirement != "none"

    @property
    def is_model_graph(self) -> bool:
        return self.protocol_family == "model_graph" or self.graph_change == "required"

    @property
    def is_inference_only(self) -> bool:
        return self.protocol_family == "inference_only" or self.inference_change == "required"


def compute_paper_protocol_hash(contract: PaperProtocolContract) -> str:
    """Return a stable SHA-256 over the paper-specific protocol fields."""
    payload = contract.model_dump(mode="json", exclude={"protocol_hash"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def missing_protocol_evaluation(paper_id: str) -> PaperProtocolEvaluation:
    """Fail closed when a paper has no explicit protocol contract."""
    return PaperProtocolEvaluation(
        paper_id=paper_id,
        protocol_hash="",
        ok=False,
        disposition="blocked_runtime",
        execution_class="blocked",
        reason_codes=["paper_protocol_missing"],
        required_evidence=["paper_protocol_contract"],
        allows_materialization=False,
        allows_asha_registration=False,
    )


def evaluate_paper_protocol(
    contract: PaperProtocolContract,
    context: PaperProtocolContext | None = None,
) -> PaperProtocolEvaluation:
    """Evaluate one paper protocol. Missing facts fail closed."""
    facts = context or PaperProtocolContext()
    reasons: list[str] = []
    evidence: list[str] = list(contract.required_evidence_artifacts)
    actions: list[str] = []
    disposition: ProtocolDisposition = "queued"
    execution_class: ExecutionClass = "pilot_candidate"

    if contract.is_inference_only:
        execution_class = "inference_candidate"
        reasons.append("inference_only_not_training_candidate")
        if facts.asha_track == "training":
            reasons.append("inference_only_excluded_from_training_asha")
        return PaperProtocolEvaluation(
            paper_id=contract.paper_id,
            protocol_hash=contract.protocol_hash,
            ok=False,
            disposition="blocked_runtime",
            execution_class="inference_candidate",
            reason_codes=list(dict.fromkeys(reasons)),
            required_evidence=list(dict.fromkeys(evidence)),
            allows_materialization=False,
            allows_asha_registration=False,
        )

    if contract.is_domain_adaptation:
        _evaluate_domain_adaptation(contract, facts, reasons, evidence, actions)

    if contract.is_distillation:
        _evaluate_distillation(contract, facts, reasons, evidence)

    if contract.is_model_graph:
        _evaluate_model_graph(contract, facts, reasons, evidence)

    if reasons:
        disposition = _disposition_from_reasons(reasons)
        return PaperProtocolEvaluation(
            paper_id=contract.paper_id,
            protocol_hash=contract.protocol_hash,
            ok=False,
            disposition=disposition,
            execution_class="blocked",
            reason_codes=list(dict.fromkeys(reasons)),
            required_evidence=list(dict.fromkeys(evidence)),
            missing_dataset_actions=list(dict.fromkeys(actions)),
            allows_materialization=False,
            allows_asha_registration=False,
        )

    return PaperProtocolEvaluation(
        paper_id=contract.paper_id,
        protocol_hash=contract.protocol_hash,
        ok=True,
        disposition="queued",
        execution_class="pilot_candidate",
        reason_codes=[],
        required_evidence=list(dict.fromkeys(evidence)),
        allows_materialization=True,
        allows_asha_registration=True,
    )


def evaluate_paper_ids(
    paper_ids: list[str],
    context: PaperProtocolContext | None = None,
    registry: "PaperProtocolRegistry | None" = None,
) -> list[PaperProtocolEvaluation]:
    """Evaluate every paper id; a missing contract is a hard stop."""
    catalog = default_paper_protocol_registry() if registry is None else registry
    return [catalog.evaluate(paper_id, context) for paper_id in paper_ids]


def authorize_paper_execution(
    paper_ids: list[str],
    context: PaperProtocolContext | None = None,
    registry: "PaperProtocolRegistry | None" = None,
) -> PaperProtocolEvaluation | None:
    """Return the first blocking evaluation, or None when all papers authorize."""
    if not paper_ids:
        return None
    for evaluation in evaluate_paper_ids(paper_ids, context, registry):
        if not evaluation.allows_materialization or not evaluation.allows_asha_registration:
            return evaluation
    return None


class PaperProtocolRegistry:
    """In-memory registry of explicit per-paper protocol contracts."""

    def __init__(self, contracts: list[PaperProtocolContract] | None = None) -> None:
        self._contracts: dict[str, PaperProtocolContract] = {}
        for contract in contracts or []:
            self.register(contract)

    def register(self, contract: PaperProtocolContract) -> PaperProtocolContract:
        existing = self._contracts.get(contract.paper_id)
        if existing is not None and existing.protocol_hash != contract.protocol_hash:
            raise PaperProtocolError(
                f"duplicate protocol for {contract.paper_id} with different hash"
            )
        self._contracts[contract.paper_id] = contract
        return contract

    def get(self, paper_id: str) -> PaperProtocolContract | None:
        return self._contracts.get(paper_id)

    def require(self, paper_id: str) -> PaperProtocolContract:
        contract = self.get(paper_id)
        if contract is None:
            raise PaperProtocolError(f"paper protocol missing: {paper_id}")
        return contract

    def evaluate(
        self,
        paper_id: str,
        context: PaperProtocolContext | None = None,
    ) -> PaperProtocolEvaluation:
        contract = self.get(paper_id)
        if contract is None:
            return missing_protocol_evaluation(paper_id)
        return evaluate_paper_protocol(contract, context)

    @property
    def paper_ids(self) -> list[str]:
        return sorted(self._contracts)

    def __len__(self) -> int:
        return len(self._contracts)

    def __bool__(self) -> bool:
        return True

    def hashes(self) -> dict[str, str]:
        return {paper_id: contract.protocol_hash for paper_id, contract in self._contracts.items()}


_DEFAULT_REGISTRY: PaperProtocolRegistry | None = None


def default_paper_protocol_registry() -> PaperProtocolRegistry:
    """Load the certified 83-paper protocol catalog once."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        from yolo_agent.research.paper_protocol_catalog import load_certified_paper_protocols

        _DEFAULT_REGISTRY = PaperProtocolRegistry(load_certified_paper_protocols())
    return _DEFAULT_REGISTRY


def reset_default_paper_protocol_registry() -> None:
    """Clear the cached default registry. Tests only."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = None


def _evaluate_domain_adaptation(
    contract: PaperProtocolContract,
    facts: PaperProtocolContext,
    reasons: list[str],
    evidence: list[str],
    actions: list[str],
) -> None:
    if contract.source_split in COCO_SPLITS or contract.target_split in COCO_SPLITS:
        reasons.append("coco_split_cannot_stand_in_for_paper_domain")
        disposition_reason = "incompatible_coco_as_domain"
        reasons.append(disposition_reason)
    if facts.coco_train_used_as_source or facts.coco_val_used_as_target:
        reasons.append("coco_split_cannot_stand_in_for_paper_domain")
        reasons.append("incompatible_coco_as_domain")
    needs_source = contract.annotation_requirement != "source_free"
    if needs_source and not facts.has_source_domain_data:
        reasons.append("domain_source_data_missing")
        evidence.extend(["source_domain_dataset", "source_domain_manifest"])
        actions.extend(missing_dataset_actions("source"))
    if not facts.has_target_domain_data:
        reasons.append("domain_target_data_missing")
        evidence.extend(["target_domain_dataset", "target_domain_manifest"])
        actions.extend(missing_dataset_actions("target"))
    if "domain_source_data_missing" in reasons or "domain_target_data_missing" in reasons:
        reasons.append("domain_adaptation_blocked_from_coco_map_training")
        evidence.append("explicit_source_target_domain_ids")
        actions.extend(DOMAIN_DATASET_ACTIONS)


def _evaluate_distillation(
    contract: PaperProtocolContract,
    facts: PaperProtocolContext,
    reasons: list[str],
    evidence: list[str],
) -> None:
    if not facts.teacher_checkpoint_exists:
        reasons.append("teacher_checkpoint_missing")
        evidence.append("teacher_checkpoint")
    if not facts.teacher_sha256:
        reasons.append("teacher_checkpoint_sha256_missing")
        evidence.append("teacher_checkpoint_sha256")
    if not facts.teacher_dataset_manifest or not facts.student_dataset_manifest:
        reasons.append("teacher_student_dataset_manifest_missing")
        evidence.append("teacher_student_dataset_manifest")
    elif facts.teacher_dataset_manifest != facts.student_dataset_manifest:
        reasons.append("teacher_student_dataset_manifest_mismatch")
    if facts.teacher_split and facts.student_split and facts.teacher_split != facts.student_split:
        reasons.append("teacher_student_split_mismatch")
    if facts.evaluate_teacher:
        reasons.append("teacher_must_not_be_evaluated")
    if contract.train_val_test_protocol != "teacher_student_same_split" and contract.protocol_family == "distillation":
        reasons.append("distillation_requires_teacher_student_same_split")


def _evaluate_model_graph(
    contract: PaperProtocolContract,
    facts: PaperProtocolContext,
    reasons: list[str],
    evidence: list[str],
) -> None:
    identity = (contract.graph_identity or facts.declared_graph_identity or "").strip()
    if not identity:
        reasons.append("graph_identity_missing")
        evidence.append("graph_identity")
    if not contract.yolo26_one_to_one_head or contract.head_constraint not in {
        "yolo26_one_to_one",
        "native_dfl_free",
    }:
        reasons.append("yolo26_one_to_one_head_unsatisfied")
    if not contract.native_dfl_free_regression:
        reasons.append("native_dfl_free_regression_unsatisfied")
    if contract.imgsz != 640:
        reasons.append("fixed_imgsz_640_violation")
    if contract.model_family != "yolo26":
        reasons.append("model_graph_requires_yolo26_family")


def _disposition_from_reasons(reasons: list[str]) -> ProtocolDisposition:
    incompatible = {
        "coco_split_cannot_stand_in_for_paper_domain",
        "incompatible_coco_as_domain",
        "yolo26_one_to_one_head_unsatisfied",
        "native_dfl_free_regression_unsatisfied",
        "fixed_imgsz_640_violation",
        "model_graph_requires_yolo26_family",
        "teacher_must_not_be_evaluated",
        "teacher_student_split_mismatch",
        "teacher_student_dataset_manifest_mismatch",
        "distillation_requires_teacher_student_same_split",
    }
    implementation = {
        "graph_identity_missing",
    }
    recoverable = {
        "domain_source_data_missing",
        "domain_target_data_missing",
        "domain_adaptation_blocked_from_coco_map_training",
        "teacher_checkpoint_missing",
        "teacher_checkpoint_sha256_missing",
        "teacher_student_dataset_manifest_missing",
    }
    if any(reason in incompatible for reason in reasons):
        return "incompatible"
    if any(reason in implementation for reason in reasons):
        return "implementation_request"
    if any(reason in recoverable for reason in reasons):
        return "evidence_recovery"
    return "blocked_runtime"


def protocol_context_from_mapping(data: dict[str, Any] | None = None) -> PaperProtocolContext:
    """Build a protocol context from node or payload metadata."""
    payload = data or {}
    return PaperProtocolContext(
        has_source_domain_data=_truthy(payload.get("has_source_domain_data")),
        has_target_domain_data=_truthy(payload.get("has_target_domain_data")),
        coco_train_used_as_source=_truthy(payload.get("coco_train_used_as_source")),
        coco_val_used_as_target=_truthy(payload.get("coco_val_used_as_target")),
        teacher_checkpoint_exists=_truthy(payload.get("teacher_checkpoint_exists")),
        teacher_sha256=_optional_text(payload.get("teacher_sha256") or payload.get("teacher_checkpoint_hash")),
        teacher_dataset_manifest=_optional_text(payload.get("teacher_dataset_manifest")),
        student_dataset_manifest=_optional_text(payload.get("student_dataset_manifest")),
        teacher_split=_optional_text(payload.get("teacher_split")),
        student_split=_optional_text(payload.get("student_split")),
        evaluate_teacher=_truthy(payload.get("evaluate_teacher")),
        declared_graph_identity=_optional_text(payload.get("graph_identity")),
        asha_track="inference" if _truthy(payload.get("inference_only")) else "training",
        component_ids=[
            str(item)
            for item in list(payload.get("component_ids") or payload.get("components") or [])
            if str(item).strip()
        ],
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def paper_ids_from_values(*values: Any) -> list[str]:
    """Normalize paper id collections from priors, nodes, or metadata."""
    found: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            found.extend(part.strip() for part in value.split(",") if part.strip())
            continue
        if isinstance(value, (list, tuple, set)):
            found.extend(str(item).strip() for item in value if str(item).strip())
            continue
        paper_ids = getattr(value, "paper_ids", None)
        if paper_ids:
            found.extend(paper_ids_from_values(paper_ids))
        metadata = getattr(value, "metadata", None)
        if isinstance(metadata, dict):
            found.extend(paper_ids_from_values(metadata.get("paper_ids"), metadata.get("paper_id")))
        changed = getattr(value, "changed_variables", None)
        if isinstance(changed, dict):
            found.extend(paper_ids_from_values(changed.get("paper_ids"), changed.get("paper_id")))
        command = getattr(value, "command_spec", None)
        if command is not None:
            found.extend(paper_ids_from_values(command))
        config = getattr(value, "candidate_config", None)
        if config is not None:
            found.extend(paper_ids_from_values(getattr(config, "paper_ids", None)))
    return list(dict.fromkeys(found))


def authorize_paper_ids_or_missing(
    paper_ids: list[str],
    context: PaperProtocolContext | None = None,
    registry: PaperProtocolRegistry | None = None,
) -> PaperProtocolEvaluation | None:
    """Block materialization/ASHA when any paper protocol fails or is missing."""
    from yolo_agent.research.paper_protocol_catalog import requires_explicit_protocol

    required = [paper_id for paper_id in paper_ids if requires_explicit_protocol(paper_id)]
    if not required:
        return None
    return authorize_paper_execution(required, context, registry)
