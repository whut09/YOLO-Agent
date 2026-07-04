"""Data-facing loop stages: init, profiling, label advice, and diagnosis."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig
from yolo_agent.agents.annotation_advisor import advise_annotations
from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopEngine
from yolo_agent.agents.error_to_action import DetectionErrorObservation
from yolo_agent.agents.loop_io import read_json, read_yaml, write_json
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.agents.run_initializer import attach_dataset_manifest_to_context
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.schemas import DeploymentConstraints
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.tools.dataset_stats import DatasetReport, profile_dataset


class DataStageRunner:
    """Run data and diagnosis stages."""

    def __init__(self, context: RunContext) -> None:
        self.context = context

    def init(self) -> StageResult:
        """Initialize run context and dataset manifest."""
        self.context.ensure_dirs()
        dataset_manifest_path = attach_dataset_manifest_to_context(self.context)
        context_path = self.context.to_yaml()
        self.context.to_json()
        return StageResult(
            stage="init",
            status="completed",
            message="Run context initialized.",
            artifacts={"run_context": context_path, "dataset_manifest": dataset_manifest_path},
        )

    def profile_data(self) -> StageResult:
        """Profile the configured YOLO dataset."""
        if not self.context.data_yaml.is_file():
            return _blocked("profile_data", f"Missing data_yaml: {self.context.data_yaml}")
        report = profile_dataset(self.context.data_yaml, self.context.artifact_path("dataset_report"))
        return StageResult(
            stage="profile_data",
            status="completed",
            message=f"Profiled images={report.image_count} labels={report.label_count}.",
            artifacts={
                "dataset_report": self.context.artifact_path("dataset_report.json"),
                "dataset_report_md": self.context.artifact_path("dataset_report.md"),
            },
        )

    def advise_labels(self) -> StageResult:
        """Generate annotation quality advice."""
        if not self.context.data_yaml.is_file():
            return _blocked("advise_labels", f"Missing data_yaml: {self.context.data_yaml}")
        report = advise_annotations(
            data_yaml=self.context.data_yaml,
            out_prefix=self.context.artifact_path("annotation_advice"),
            predictions_path=self.context.predictions_path,
        )
        return StageResult(
            stage="advise_labels",
            status="completed",
            message=f"Found label_issues={len(report.label_quality.issues)}.",
            artifacts={
                "annotation_advice": self.context.artifact_path("annotation_advice.json"),
                "annotation_advice_md": self.context.artifact_path("annotation_advice.md"),
            },
        )

    def diagnose_errors(self) -> StageResult:
        """Diagnose detection errors against task and dataset context."""
        errors_path = self.context.detection_errors_path
        if errors_path is None or not errors_path.is_file():
            return _blocked("diagnose_errors", "Missing detection_errors_path; cannot diagnose model errors.")
        dataset_report_path = self.context.artifact_path("dataset_report.json")
        if not dataset_report_path.is_file():
            return _blocked("diagnose_errors", "Missing dataset_report; run profile_data first.")

        task_spec = TaskSpec.from_yaml(self.context.task_path)
        dataset_report = DatasetReport.model_validate(read_json(dataset_report_path))
        observations = read_detection_errors(errors_path)
        deployment = DeploymentConstraints(
            target="unknown",
            max_latency_ms=task_spec.max_latency_ms,
            max_model_size_mb=task_spec.max_model_size_mb,
            preferred_export="none",
        )
        report = ErrorDrivenLoopEngine().run(
            task_spec,
            dataset_report,
            observations,
            deployment,
            fixed_imgsz=_fixed_imgsz_from_context(self.context),
        )
        path = self.context.artifact_path("loop_diagnosis.json")
        write_json(path, report.model_dump(mode="json"))
        return StageResult(
            stage="diagnose_errors",
            status="completed",
            message=f"Created loop diagnosis with {len(report.diagnostics)} diagnostics.",
            artifacts={"loop_diagnosis": path},
        )


def read_detection_errors(path: Path) -> list[DetectionErrorObservation]:
    """Load detection error observations from JSON or YAML."""
    raw = read_yaml(path) if path.suffix.lower() in {".yaml", ".yml"} else read_json(path)
    items = raw.get("errors", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Detection errors must be a list or an 'errors' list.")
    return [DetectionErrorObservation.model_validate(item) for item in items]


def _blocked(stage: LoopStage, message: str) -> StageResult:
    return StageResult(stage=stage, status="blocked", message=message)


def _fixed_imgsz_from_context(context: RunContext) -> int | None:
    """Return the fixed training imgsz from optional Ultralytics training config."""
    raw_path = context.metadata.get("training_config_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_file():
        return None
    return UltralyticsTrainingConfig.from_yaml(path).imgsz
