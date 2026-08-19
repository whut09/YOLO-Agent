"""Certified 83-paper protocol catalog.

Each row is derived from the frozen coverage acceptance traces. The catalog
emits a complete PaperProtocolContract per paper; it does not reuse
ComponentContract as the paper execution protocol.
"""

from __future__ import annotations

from typing import Iterable

from yolo_agent.research.paper_evidence_requirements import (
    evidence_artifacts_for_family,
    required_metrics_for_family,
)
from yolo_agent.research.paper_protocol_contract import (
    AnnotationRequirement,
    ChangeKind,
    DatasetRole,
    PaperProtocolContract,
    ProtocolFamily,
    SplitRole,
    TeacherRequirement,
    TrainValTestProtocol,
)
from yolo_agent.research.paper_protocol_ids import CERTIFIED_PAPER_MECHANISMS


GRAPH_MECHANISM_PREFIXES = (
    "neck.",
    "detection_head.",
    "feature_pyramid.",
    "assigner.",
)
SOURCE_FREE_MARKERS = ("source-free", "source_free", "sourcefree")


CATALOG_PAPER_PREFIXES = ("arxiv:", "cvf:", "ecva:", "neurips:", "papernotes:")


def certified_paper_ids() -> list[str]:
    """Return the 83 certified paper IDs in catalog order."""
    return list(CERTIFIED_PAPER_MECHANISMS)


def requires_explicit_protocol(paper_id: str) -> bool:
    """Real catalog papers must have a PaperProtocolContract before execution."""
    return paper_id in CERTIFIED_PAPER_MECHANISMS or paper_id.startswith(CATALOG_PAPER_PREFIXES)


def classify_protocol_family(paper_id: str, mechanisms: Iterable[str]) -> ProtocolFamily:
    """Assign the primary protocol family from frozen mechanism IDs."""
    ids = set(mechanisms)
    if any(item.startswith("inference.") for item in ids):
        return "inference_only"
    if "domain_adaptation.general" in ids:
        return "domain_adaptation"
    if any(item.startswith("distillation.") for item in ids):
        return "distillation"
    if any(item.startswith(prefix) for item in ids for prefix in GRAPH_MECHANISM_PREFIXES):
        return "model_graph"
    return "standard_training"


def build_paper_protocol_contract(
    paper_id: str,
    mechanisms: Iterable[str] | None = None,
) -> PaperProtocolContract:
    """Build one explicit paper protocol from certified mechanism IDs."""
    mechanism_ids = list(mechanisms or CERTIFIED_PAPER_MECHANISMS[paper_id])
    family = classify_protocol_family(paper_id, mechanism_ids)
    source_free = _is_source_free(paper_id)
    graph_identity = _graph_identity(mechanism_ids)
    teacher = _teacher_requirement(family, paper_id, mechanism_ids)
    return PaperProtocolContract(
        paper_id=paper_id,
        dataset_role=_dataset_role(family),
        source_split=_source_split(family),
        target_split=_target_split(family),
        annotation_requirement=_annotation_requirement(family, source_free),
        train_val_test_protocol=_train_protocol(family),
        imgsz=640,
        model_family="yolo26",
        head_constraint="yolo26_one_to_one",
        teacher_requirement=teacher,
        graph_change="required" if family == "model_graph" else "none",
        loss_change=_loss_change(mechanism_ids),
        inference_change="required" if family == "inference_only" else "none",
        required_metrics=required_metrics_for_family(family),
        paired_baseline_requirement=True,
        required_evidence_artifacts=evidence_artifacts_for_family(family),
        protocol_family=family,
        graph_identity=graph_identity,
        yolo26_one_to_one_head=True,
        native_dfl_free_regression=True,
        mechanism_ids=mechanism_ids,
    )


def load_certified_paper_protocols() -> list[PaperProtocolContract]:
    """Materialize the 83 certified paper protocol contracts."""
    return [
        build_paper_protocol_contract(paper_id, mechanisms)
        for paper_id, mechanisms in CERTIFIED_PAPER_MECHANISMS.items()
    ]


def inference_only_protocol(paper_id: str = "arxiv:2202.06934") -> PaperProtocolContract:
    """Build an inference-only protocol used to keep SAHI-style methods off training ASHA."""
    return PaperProtocolContract(
        paper_id=paper_id,
        dataset_role="coco_standard",
        source_split="coco_train",
        target_split="coco_val",
        annotation_requirement="fully_labeled",
        train_val_test_protocol="inference_only",
        imgsz=640,
        model_family="yolo26",
        head_constraint="yolo26_one_to_one",
        teacher_requirement="none",
        graph_change="none",
        loss_change="none",
        inference_change="required",
        required_metrics=required_metrics_for_family("inference_only"),
        paired_baseline_requirement=True,
        required_evidence_artifacts=evidence_artifacts_for_family("inference_only"),
        protocol_family="inference_only",
        graph_identity=None,
        yolo26_one_to_one_head=True,
        native_dfl_free_regression=True,
        mechanism_ids=["inference.sahi_slicing"],
    )


def _dataset_role(family: ProtocolFamily) -> DatasetRole:
    if family == "domain_adaptation":
        return "source_target_domain"
    if family == "inference_only":
        return "coco_standard"
    return "coco_standard"


def _source_split(family: ProtocolFamily) -> SplitRole:
    if family == "domain_adaptation":
        return "source_train"
    return "coco_train"


def _target_split(family: ProtocolFamily) -> SplitRole:
    if family == "domain_adaptation":
        return "target_val"
    return "coco_val"


def _annotation_requirement(
    family: ProtocolFamily,
    source_free: bool,
) -> AnnotationRequirement:
    if family != "domain_adaptation":
        return "fully_labeled"
    return "source_free" if source_free else "source_labeled_target_unlabeled"


def _train_protocol(family: ProtocolFamily) -> TrainValTestProtocol:
    if family == "domain_adaptation":
        return "source_target_domain"
    if family == "distillation":
        return "teacher_student_same_split"
    if family == "inference_only":
        return "inference_only"
    return "coco_official"


def _teacher_requirement(
    family: ProtocolFamily,
    paper_id: str,
    mechanisms: list[str],
) -> TeacherRequirement:
    if family == "distillation" or any(item.startswith("distillation.") for item in mechanisms):
        return "frozen_teacher_checkpoint"
    lowered = paper_id.lower()
    if family == "domain_adaptation" and "teacher" in lowered:
        return "cross_domain_teacher"
    return "none"


def _loss_change(mechanisms: list[str]) -> ChangeKind:
    if any(item.startswith("loss.") or item.startswith("quality_alignment.") for item in mechanisms):
        return "required"
    return "none"


def _graph_identity(mechanisms: list[str]) -> str | None:
    graph_ids = [
        item
        for item in mechanisms
        if item.startswith(GRAPH_MECHANISM_PREFIXES)
    ]
    return "+".join(graph_ids) if graph_ids else None


def _is_source_free(paper_id: str) -> bool:
    lowered = paper_id.lower().replace(" ", "_")
    return any(marker in lowered for marker in SOURCE_FREE_MARKERS)
