"""Active-learning loop stages for mining, labeling handoff, and dataset promotion."""

from __future__ import annotations

from yolo_agent.agents.active_learning import (
    ActiveLearningMiner,
    ActiveLearningPlan,
    LabelHandoffResult,
    LabelingManifest,
    LabelingTarget,
    MiningConfig,
    load_prediction_summaries,
)
from yolo_agent.agents.loop_evidence import LoopEvidence
from yolo_agent.agents.loop_io import read_json, write_json
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.core.dataset_promotion import DatasetPromotionPolicy, load_reviewed_labels
from yolo_agent.core.dataset_versioning import DatasetVersionManifest
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.run_context import RunContext


class ActiveLearningStageRunner:
    """Run active-learning stages as first-class loop stages."""

    def __init__(
        self,
        context: RunContext,
        evidence: LoopEvidence,
        mining_config: MiningConfig | None = None,
    ) -> None:
        self.context = context
        self.evidence = evidence
        self.mining_config = mining_config

    def mine_samples(self) -> StageResult:
        """Mine unlabeled predictions into active-learning samples."""
        predictions_path = self.context.predictions_path
        if predictions_path is None or not predictions_path.is_file():
            return _blocked("mine_samples", "Missing predictions input; cannot mine active-learning samples.")
        target = _labeling_target(self.context)
        predictions = load_prediction_summaries(predictions_path)
        plan = ActiveLearningMiner(self.mining_config).mine(
            predictions=predictions,
            dataset_version=self.context.dataset_version,
            labeling_target=target,
        )
        plan_path = self.context.artifact_path("active_learning_plan.json")
        write_json(plan_path, plan.model_dump(mode="json"))
        self.context.metadata["active_learning_next_dataset_version"] = plan.next_dataset_version
        self.context.metadata["active_learning_mined_samples"] = len(plan.mined_samples)
        self.context.to_yaml()
        self.context.to_json()
        return StageResult(
            stage="mine_samples",
            status="completed",
            message=f"Mined {len(plan.mined_samples)} samples for {plan.next_dataset_version}.",
            artifacts={"active_learning_plan": plan_path},
        )

    def label_handoff(self) -> StageResult:
        """Materialize a labeling manifest from the active-learning plan."""
        plan_path = self.context.artifact_path("active_learning_plan.json")
        if not plan_path.is_file():
            return _blocked("label_handoff", "Missing active_learning_plan; run mine_samples first.")
        plan = ActiveLearningPlan.model_validate(read_json(plan_path))
        manifest_path = self.context.artifact_path("labeling_manifest.json")
        plan.labeling_manifest.to_json(manifest_path)
        handoff = LabelHandoffResult(
            target=plan.labeling_manifest.target,
            dataset_version=plan.dataset_version,
            next_dataset_version=plan.next_dataset_version,
            sample_count=len(plan.mined_samples),
            labeling_manifest_path=manifest_path,
        )
        handoff_path = self.context.artifact_path("label_handoff.json")
        write_json(handoff_path, handoff.model_dump(mode="json"))
        self.context.metadata["active_learning_labeling_manifest"] = manifest_path.as_posix()
        self.context.to_yaml()
        self.context.to_json()
        return StageResult(
            stage="label_handoff",
            status="completed",
            message=f"Prepared {handoff.sample_count} samples for {handoff.target}.",
            artifacts={"labeling_manifest": manifest_path, "label_handoff": handoff_path},
        )

    def dataset_promote(self) -> StageResult:
        """Plan dataset promotion after labeling review without mutating data."""
        manifest_path = self.context.artifact_path("labeling_manifest.json")
        if not manifest_path.is_file():
            return _blocked("dataset_promote", "Missing labeling_manifest; run label_handoff first.")
        manifest = LabelingManifest.model_validate(read_json(manifest_path))
        parent_manifest = (
            DatasetVersionManifest.from_json(self.context.dataset_manifest_path)
            if self.context.dataset_manifest_path is not None and self.context.dataset_manifest_path.is_file()
            else None
        )
        reviewed_labels = (
            load_reviewed_labels(self.context.reviewed_labels_path)
            if self.context.reviewed_labels_path is not None and self.context.reviewed_labels_path.is_file()
            else None
        )
        promotion = DatasetPromotionPolicy().evaluate(
            labeling_manifest=manifest,
            reviewed_labels=reviewed_labels,
            parent_manifest=parent_manifest,
            parent_manifest_path=self.context.dataset_manifest_path,
        )
        path = self.context.artifact_path("dataset_promotion.json")
        write_json(path, promotion.model_dump(mode="json"))
        self.context.metadata["active_learning_next_dataset_version"] = manifest.next_dataset_version
        self.context.metadata["dataset_promotion_decision"] = promotion.decision
        self.context.metadata["dataset_promotion_status"] = promotion.decision
        self.context.metadata["dataset_promotion_path"] = path.as_posix()
        self.context.metadata["dataset_promotion_promoted"] = promotion.promoted
        self.context.to_yaml()
        self.context.to_json()
        self.evidence.record_lineage(
            metadata={
                "active_learning_next_dataset_version": manifest.next_dataset_version,
                "dataset_promotion_decision": promotion.decision,
                "dataset_promotion_promoted": promotion.promoted,
                "dataset_promotion_reasons": promotion.reasons,
            }
        )
        return StageResult(
            stage="dataset_promote",
            status="completed",
            message=f"Dataset promotion decision={promotion.decision} for {manifest.next_dataset_version}.",
            artifacts={"dataset_promotion": path},
        )


def _labeling_target(context: RunContext) -> LabelingTarget:
    value = context.metadata.get("labeling_target", "generic")
    return value if value in {"generic", "cvat", "label_studio"} else "generic"


def _blocked(stage: LoopStage, message: str) -> StageResult:
    return StageResult(stage=stage, status="blocked", message=message)
