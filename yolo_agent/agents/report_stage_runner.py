"""Reporting and next-round loop stages."""

from __future__ import annotations

from yolo_agent.agents.loop_evidence import LoopEvidence
from yolo_agent.agents.loop_io import read_yaml, write_yaml
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.run_context import RunContext
from yolo_agent.reports.experiment_report import generate_experiment_report


class ReportStageRunner:
    """Run report and next-round stages."""

    def __init__(
        self,
        context: RunContext,
        evidence: LoopEvidence,
    ) -> None:
        self.context = context
        self.evidence = evidence

    def report(self) -> StageResult:
        """Generate the run report."""
        gate_path = self.evidence.write_status()
        output_path = self.context.run_dir / "report.md"
        generate_experiment_report(self.context.run_dir, output_path)
        return StageResult(
            stage="report",
            status="completed",
            message=f"Wrote {output_path}.",
            artifacts={"report": output_path, "evidence_status": gate_path},
        )

    def next_round(self) -> StageResult:
        """Generate the next-round checklist."""
        loop_plan_path = self.context.artifact_path("loop_plan.yaml")
        if not loop_plan_path.is_file():
            return _blocked("next_round", "Missing loop_plan; run generate_loop_plan first.")
        raw_plan = read_yaml(loop_plan_path)
        gate_path = self.evidence.write_status()
        output_path = self.context.artifact_path("next_round.yaml")
        write_yaml(output_path, self.evidence.next_round_payload(raw_plan))
        return StageResult(
            stage="next_round",
            status="completed",
            message="Next-round checklist generated.",
            artifacts={"next_round": output_path, "evidence_status": gate_path},
        )


def _blocked(stage: LoopStage, message: str) -> StageResult:
    return StageResult(stage=stage, status="blocked", message=message)
