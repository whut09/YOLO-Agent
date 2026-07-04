"""Facade for concrete loop stage runners."""

from __future__ import annotations

from collections.abc import Callable

from yolo_agent.agents.active_learning_stage_runner import ActiveLearningStageRunner
from yolo_agent.agents.candidate_stage_runner import CandidateStageRunner
from yolo_agent.agents.data_stage_runner import DataStageRunner
from yolo_agent.agents.evidence_stage_runner import EvidenceStageRunner
from yolo_agent.agents.loop_evidence import LoopEvidence
from yolo_agent.agents.loop_types import StageResult
from yolo_agent.agents.policy_stage_runner import PolicyStageRunner
from yolo_agent.agents.report_stage_runner import ReportStageRunner
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.loop_state import LoopStage
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.stage_contract import LoopStageContracts


class StageRunner:
    """Route concrete loop stages without owning the state machine."""

    def __init__(
        self,
        context: RunContext,
        policy: LoopStageContracts,
        evidence_store: EvidenceStore,
        evidence: LoopEvidence,
    ) -> None:
        self.data = DataStageRunner(context)
        self.policy = PolicyStageRunner(context, policy, evidence)
        self.candidates = CandidateStageRunner(context)
        self.evidence = EvidenceStageRunner(context, evidence_store, evidence)
        self.reports = ReportStageRunner(context, evidence)
        self.active_learning = ActiveLearningStageRunner(context, evidence)
        self._dispatch: dict[LoopStage, Callable[[], StageResult]] = {
            "init": self.data.init,
            "profile_data": self.data.profile_data,
            "advise_labels": self.data.advise_labels,
            "diagnose_errors": self.data.diagnose_errors,
            "generate_loop_plan": self.policy.generate_loop_plan,
            "evaluate_policies": self.policy.evaluate_policies,
            "generate_candidates": self.candidates.generate_candidates,
            "ablate": self.candidates.ablate,
            "smoke": self.evidence.smoke,
            "import_metrics": self.evidence.import_metrics,
            "report": self.reports.report,
            "next_round": self.reports.next_round,
            "mine_samples": self.active_learning.mine_samples,
            "label_handoff": self.active_learning.label_handoff,
            "dataset_promote": self.active_learning.dataset_promote,
        }

    def run(self, stage: LoopStage) -> StageResult:
        """Run one concrete stage."""
        return self._dispatch[stage]()
