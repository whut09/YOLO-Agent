"""Candidate plan and ablation loop stages."""

from __future__ import annotations

import shutil

from yolo_agent.agents.ablation_planner import AblationPlanner
from yolo_agent.agents.candidate_generator import CandidateConfig, CandidatePlan
from yolo_agent.agents.loop_io import read_yaml
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluationReport
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.task_spec import TaskSpec


class CandidateStageRunner:
    """Run candidate generation and ablation stages."""

    def __init__(self, context: RunContext) -> None:
        self.context = context

    def generate_candidates(self) -> StageResult:
        """Convert accepted policy evaluations into a candidate plan."""
        evaluation_path = self.context.artifact_path("policy_evaluation.yaml")
        if not evaluation_path.is_file():
            return _blocked("generate_candidates", "Missing policy_evaluation; run evaluate_policies first.")
        task_spec = TaskSpec.from_yaml(self.context.task_path)
        evaluation = LoopPolicyEvaluationReport.model_validate(read_yaml(evaluation_path))
        candidates = [baseline_candidate(), *evaluation.accepted_candidates]
        plan = CandidatePlan(task_scene=task_spec.scene, candidates=dedupe_candidates(candidates))
        plan_path = self.context.run_dir / "plan.yaml"
        plan.to_yaml(plan_path)
        shutil.copy2(plan_path, self.context.artifact_path("candidate_plan.yaml"))
        return StageResult(
            stage="generate_candidates",
            status="completed",
            message=f"Generated {len(plan.candidates)} candidates.",
            artifacts={"candidate_plan": plan_path},
        )

    def ablate(self) -> StageResult:
        """Create a single-variable ablation plan from the candidate plan."""
        plan_path = self.context.run_dir / "plan.yaml"
        if not plan_path.is_file():
            return _blocked("ablate", "Missing candidate plan; run generate_candidates first.")
        candidate_plan = CandidatePlan.from_yaml(plan_path)
        ablation_plan = AblationPlanner().plan(candidate_plan.candidates)
        path = self.context.run_dir / "ablation_plan.yaml"
        ablation_plan.to_yaml(path)
        shutil.copy2(path, self.context.artifact_path("ablation_plan.yaml"))
        return StageResult(
            stage="ablate",
            status="completed",
            message=f"Created {len(ablation_plan.nodes)} ablation nodes.",
            artifacts={"ablation_plan": path},
        )


def baseline_candidate() -> CandidateConfig:
    """Return the default baseline candidate."""
    return CandidateConfig(
        candidate_id="yolo11n_baseline_n",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        expected_effect=["Baseline reference experiment."],
        risk="low",
    )


def dedupe_candidates(candidates: list[CandidateConfig]) -> list[CandidateConfig]:
    """Keep candidates unique by candidate_id while preserving order."""
    seen: set[str] = set()
    deduped: list[CandidateConfig] = []
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        deduped.append(candidate)
    return deduped


def _blocked(stage: LoopStage, message: str) -> StageResult:
    return StageResult(stage=stage, status="blocked", message=message)
