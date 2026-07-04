"""Evidence-producing loop stages: smoke guards and metric import."""

from __future__ import annotations

from yolo_agent.agents.loop_evidence import LoopEvidence
from yolo_agent.agents.loop_io import write_json
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.executor import BenchmarkImporter
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.run_context import RunContext
from yolo_agent.tools.smoke_runner import SmokeRunner, log_smoke_guard_evidence


class EvidenceStageRunner:
    """Run stages that create or ingest evidence."""

    def __init__(
        self,
        context: RunContext,
        evidence_store: EvidenceStore,
        evidence: LoopEvidence,
    ) -> None:
        self.context = context
        self.evidence_store = evidence_store
        self.evidence = evidence

    def smoke(self) -> StageResult:
        """Run pre-training smoke guards without real training."""
        plan_path = self.context.run_dir / "plan.yaml"
        if not plan_path.is_file():
            return _blocked("smoke", "Missing candidate plan; run generate_candidates first.")
        result = SmokeRunner(EvidenceStore(self.context.run_dir / "smoke_evidence")).run(
            plan_path=plan_path,
            data_path=self.context.data_yaml,
            run_id="smoke",
            generated_dir=self.context.artifact_path("generated_models"),
        )
        path = self.context.artifact_path("smoke_result.json")
        write_json(path, result.model_dump(mode="json"))
        log_smoke_guard_evidence(
            evidence_store=self.evidence_store,
            run_id=self.context.run_id,
            result=result,
            dataset_version=self.context.dataset_version,
            source_artifact=path,
        )
        return StageResult(
            stage="smoke",
            status="failed" if result.status == "failed" else "completed",
            message=f"Smoke status={result.status}.",
            artifacts={"smoke_result": path},
        )

    def import_metrics(self) -> StageResult:
        """Import externally produced benchmark metrics."""
        metrics_path = self.context.metrics_input_path
        if metrics_path is None or not metrics_path.is_file():
            gate_path = self.evidence.write_status()
            return StageResult(
                stage="import_metrics",
                status="blocked",
                message="Missing metrics_input_path; import external benchmark metrics later.",
                artifacts={"evidence_status": gate_path},
            )
        import_result = BenchmarkImporter(self.evidence_store).import_metrics(
            run_id=self.context.run_id,
            metrics_path=metrics_path,
            dataset_version=self.context.dataset_version,
            source="loop_ingest_metrics",
        )
        output_path = self.context.artifact_path("metrics_import.json")
        write_json(output_path, import_result.run_metrics)
        gate_path = self.evidence.write_status()
        return StageResult(
            stage="import_metrics",
            status="completed",
            message=(
                f"Imported {len(import_result.run_metrics)} run metrics "
                f"and {len(import_result.metric_records)} node metrics."
            ),
            artifacts={"metrics": output_path, "evidence_status": gate_path},
        )


def _blocked(stage: LoopStage, message: str) -> StageResult:
    return StageResult(stage=stage, status="blocked", message=message)
