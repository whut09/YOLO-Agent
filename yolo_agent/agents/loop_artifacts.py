"""Artifact and stage-contract helpers for loop harness orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from yolo_agent.core.artifact_manifest import ArtifactManifest
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.loop_state import LoopStage, LoopState
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.stage_contract import LoopStageContracts, StageContractCheck


class LoopArtifacts:
    """Centralize artifact manifest recording and contract input discovery."""

    def __init__(
        self,
        context: RunContext,
        state: LoopState,
        policy: LoopStageContracts,
        evidence_store: EvidenceStore,
        evidence_status_writer: Callable[[], Path],
    ) -> None:
        self.context = context
        self.state = state
        self.policy = policy
        self.evidence_store = evidence_store
        self.evidence_status_writer = evidence_status_writer

    def record(self, stage: LoopStage, artifacts: dict[str, Path]) -> None:
        """Record artifact manifest entries for stage outputs."""
        for name, path in artifacts.items():
            artifact_path = Path(path)
            if not artifact_path.exists():
                continue
            self.evidence_store.log_artifact_manifest(
                run_id=self.context.run_id,
                name=name,
                artifact_path=artifact_path,
                producer_stage=stage,
            )

    def check_stage_contract(self, stage: LoopStage) -> StageContractCheck:
        """Validate configured requirements for a stage."""
        manifest_path = self.context.artifact_path("artifact_manifest.jsonl")
        manifest_entries = ArtifactManifest(manifest_path).read() if manifest_path.is_file() else []
        return self.policy.get(stage).check(
            self.available_contract_items(),
            manifest_entries=manifest_entries,
            current_run_dir=self.context.run_dir,
            evidence=self.evidence_store.load_run(self.context.run_id),
            evidence_artifacts=self.state.artifacts,
        )

    def stage_provides(self, stage: LoopStage) -> list[str]:
        """Return configured outputs for event logging."""
        return self.policy.get(stage).provides

    def blocked_contract_artifacts(self, stage: LoopStage) -> dict[str, Path]:
        """Write any artifacts a blocked contract should still produce."""
        contract = self.policy.get(stage)
        artifacts: dict[str, Path] = {}
        if "evidence_status" in contract.producer_artifacts:
            artifacts["evidence_status"] = self.evidence_status_writer()
        return artifacts

    def available_contract_items(self) -> set[str]:
        """Return currently available contract inputs for this run."""
        available: set[str] = set()
        if self.context.task_path.is_file():
            available.add("task_spec")
        if self.context.data_yaml.is_file():
            available.add("data_yaml")
        if self.context.component_path.exists():
            available.add("component_registry")
        if self.context.detection_errors_path is not None and self.context.detection_errors_path.is_file():
            available.add("detection_errors")
        if self.context.predictions_path is not None and self.context.predictions_path.is_file():
            available.add("predictions_input")
        if self.context.reviewed_labels_path is not None and self.context.reviewed_labels_path.is_file():
            available.add("reviewed_labels")
        if self.context.metrics_input_path is not None and self.context.metrics_input_path.is_file():
            available.add("metrics_input")
        if (self.context.run_dir / "run_context.yaml").is_file():
            available.add("run_context")
        if (self.context.run_dir / "loop_state.yaml").is_file():
            available.add("loop_state")
        if self.context.dataset_manifest_path is not None and self.context.dataset_manifest_path.is_file():
            available.add("dataset_manifest")
        if (self.context.run_dir / "metrics.json").is_file():
            available.add("metrics")
        if (self.context.run_dir / "report.md").is_file():
            available.add("report")

        available.update(self.state.artifacts.keys())
        available.update(existing_artifact_items(self.context))
        return available


def existing_artifact_items(context: RunContext) -> set[str]:
    """Return contract item names inferred from existing artifact files."""
    path_candidates = {
        "dataset_report": [context.artifact_path("dataset_report.json")],
        "label_quality_report": [context.artifact_path("annotation_advice.json")],
        "annotation_advice": [context.artifact_path("annotation_advice.json")],
        "loop_diagnosis": [context.artifact_path("loop_diagnosis.json")],
        "loop_plan": [context.artifact_path("loop_plan.yaml")],
        "policy_evaluation": [context.artifact_path("policy_evaluation.yaml")],
        "decision_ledger": [context.artifact_path("decision_ledger.jsonl")],
        "experiment_plan": [context.artifact_path("experiment_plan.yaml")],
        "candidate_plan": [context.run_dir / "plan.yaml", context.artifact_path("candidate_plan.yaml")],
        "ablation_plan": [context.run_dir / "ablation_plan.yaml", context.artifact_path("ablation_plan.yaml")],
        "smoke_result": [context.artifact_path("smoke_result.json")],
        "evidence_status": [context.artifact_path("evidence_status.json")],
        "artifact_manifest": [context.artifact_path("artifact_manifest.jsonl")],
        "execution_queue": [context.run_dir / "execution_queue.yaml"],
        "execution_results": [context.artifact_path("execution_results")],
        "active_learning_plan": [context.artifact_path("active_learning_plan.json")],
        "labeling_manifest": [context.artifact_path("labeling_manifest.json")],
        "label_handoff": [context.artifact_path("label_handoff.json")],
        "dataset_promotion": [context.artifact_path("dataset_promotion.json")],
        "dataset_manifest": [
            context.dataset_manifest_path,
            context.run_dir / "dataset_versions" / context.dataset_version / "manifest.json",
        ],
        "next_round": [context.artifact_path("next_round.yaml")],
    }
    return {
        name
        for name, paths in path_candidates.items()
        if any(path is not None and path.exists() for path in paths)
    }
