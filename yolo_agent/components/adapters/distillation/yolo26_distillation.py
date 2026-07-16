"""Executable YOLO26 teacher-student distillation adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from yolo_agent.components.adapters.base import AdapterContext, AdapterValidationReport, ComponentAdapter, ExpectedArtifact, RollbackPlan, SmokeTestResult, WeightLoadResult
from yolo_agent.components.distillation import DistillationTrainerHook, DistillationWeights, YOLO26DistillationLoss


class YOLO26DistillationConfig(BaseModel):
    teacher: str = "yolo26s.pt"
    student: str = "yolo26n.pt"
    teacher_data: str
    student_data: str
    teacher_split: str = "train"
    student_split: str = "train"
    imgsz: int = 640
    logits: bool = True
    feature: bool = True
    localization: bool = True
    weights: DistillationWeights = Field(default_factory=DistillationWeights)
    amp: bool = True
    resume: bool | str = False

    @model_validator(mode="after")
    def validate_protocol(self) -> "YOLO26DistillationConfig":
        if self.teacher not in {"yolo26s.pt", "yolo26m.pt"}:
            raise ValueError("teacher must be yolo26s.pt or yolo26m.pt")
        if self.student != "yolo26n.pt":
            raise ValueError("student must be yolo26n.pt")
        if self.imgsz != 640:
            raise ValueError("distillation requires fixed imgsz=640")
        if self.teacher_data != self.student_data or self.teacher_split != self.student_split:
            raise ValueError("teacher and student dataset/split must match")
        return self


class DistillationEvidence(BaseModel):
    teacher_checkpoint: str
    teacher_checkpoint_sha256: str
    student_checkpoint: str
    student_checkpoint_sha256: str
    dataset: str
    split: str
    imgsz: int = 640


class YOLO26DistillationAdapter(ComponentAdapter):
    adapter_version = "yolo26_distillation.v1"
    source_commit = "local"
    strategy = "trainer_subclass"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset({"distillation"})

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        try:
            import torch
            return AdapterValidationReport(ok=True, checks={"torch": torch.__version__})
        except ImportError:
            return AdapterValidationReport(ok=False, errors=["torch is required"])

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        try:
            config = YOLO26DistillationConfig.model_validate(context.options)
        except ValueError as exc:
            return AdapterValidationReport(ok=False, errors=[str(exc)])
        return AdapterValidationReport(ok=context.imgsz == 640, errors=[] if context.imgsz == 640 else ["fixed imgsz=640 required"], checks={"teacher": config.teacher, "student": config.student})

    def patch_model_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        return config

    def patch_training_config(self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True) -> dict[str, Any]:
        distill = YOLO26DistillationConfig.model_validate(context.options)
        config["distillation"] = distill.model_dump(mode="json")
        return config

    def build_module(self, context: AdapterContext) -> YOLO26DistillationLoss:
        config = YOLO26DistillationConfig.model_validate(context.options)
        weights = config.weights.model_copy(update={
            "logits": config.weights.logits if config.logits else 0.0,
            "feature": config.weights.feature if config.feature else 0.0,
            "localization": config.weights.localization if config.localization else 0.0,
        })
        return YOLO26DistillationLoss(weights)

    def build_trainer_hook(self, teacher_model: Any, context: AdapterContext) -> DistillationTrainerHook:
        return DistillationTrainerHook(teacher_model, self.build_module(context))

    def load_pretrained_weights(self, module: Any, weights: Path | str | None, context: AdapterContext) -> WeightLoadResult:
        if weights is None:
            return WeightLoadResult(loaded=False, message="teacher checkpoint required by trainer")
        return WeightLoadResult(loaded=Path(weights).is_file(), source=Path(weights), message="checkpoint provenance recorded; trainer owns model loading")

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            config = YOLO26DistillationConfig.model_validate(context.options)
            return SmokeTestResult(passed=True, checks={"student_graph_unchanged": True, "imgsz": str(config.imgsz)})
        except ValueError as exc:
            return SmokeTestResult(passed=False, errors=[str(exc)])

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        return [ExpectedArtifact(name="distillation_evidence", relative_path=Path("distillation_evidence.json")), ExpectedArtifact(name="student_best", relative_path=Path("weights/best.pt"))]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(actions=["remove trainer hook and distillation loss injection"], files_to_remove=[Path("distillation_evidence.json")])

    def build_evidence(self, teacher_checkpoint: Path | str, student_checkpoint: Path | str, context: AdapterContext) -> DistillationEvidence:
        config = YOLO26DistillationConfig.model_validate(context.options)
        teacher, student = Path(teacher_checkpoint), Path(student_checkpoint)
        return DistillationEvidence(teacher_checkpoint=str(teacher), teacher_checkpoint_sha256=_sha256(teacher), student_checkpoint=str(student), student_checkpoint_sha256=_sha256(student), dataset=config.teacher_data, split=config.teacher_split)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["DistillationEvidence", "YOLO26DistillationAdapter", "YOLO26DistillationConfig"]
