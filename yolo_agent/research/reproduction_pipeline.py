"""Recoverable reproduction state machine for paper components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.event_log import EventLog
from yolo_agent.core.execution_queue import ExecutionQueue, ExecutionQueueStore
from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.research.reproduction_state import ReproductionContract, ReproductionState, ReproductionStatus


class ReproductionTransitionError(ValueError):
    """Raised when a reproduction transition violates its contract."""


class ReproductionPipeline:
    """State machine with append-only events and resumable state on disk."""

    ORDER: tuple[ReproductionStatus, ...] = (
        "registered", "license_checked", "adapter_required", "adapter_implemented",
        "unit_tested", "smoke_passed", "debug_passed", "pilot_running",
        "pilot_reproduced", "pilot_rejected", "full_pending_confirmation",
        "full_reproduced", "confirmed_multi_seed",
    )

    def __init__(self, run_dir: Path | str, paper_id: str, component_id: str, *, policy_path: Path | str = "configs/reproduction_policy.yaml") -> None:
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / "artifacts" / "reproduction_state.yaml"
        self.event_log = EventLog(self.run_dir / "events.jsonl")
        self.paper_id = paper_id
        self.component_id = component_id
        self.contracts = self._load_policy(Path(policy_path))

    def initialize(self, *, paper_claims: list[dict[str, Any]] | None = None, evidence: dict[str, Any] | None = None) -> ReproductionState:
        if self.state_path.exists():
            return self.load()
        state = ReproductionState(paper_id=self.paper_id, component_id=self.component_id, contracts=self.contracts, paper_claims=paper_claims or [], evidence={"registered": True, **(evidence or {})})
        state.refresh_satisfied_evidence()
        self.save(state)
        self._event("registered", "Reproduction registered", {"paper_id": self.paper_id, "component_id": self.component_id})
        return state

    def load(self) -> ReproductionState:
        return ReproductionState.from_yaml(self.state_path)

    def save(self, state: ReproductionState) -> Path:
        state.refresh_satisfied_evidence()
        return state.to_yaml(self.state_path)

    def transition(self, target: ReproductionStatus, *, evidence: dict[str, Any] | None = None, local_delta: dict[str, Any] | None = None, confirm_full: bool = False) -> ReproductionState:
        state = self.initialize()
        incoming = evidence or {}
        state.evidence.update(incoming)
        if target == "full_reproduced" and confirm_full:
            state.evidence["full_confirmed"] = True
        if local_delta:
            state.local_delta.update(local_delta)
        state.refresh_satisfied_evidence()
        if target == "full_reproduced" and not confirm_full and not state.has("full_confirmed"):
            raise ReproductionTransitionError("full reproduction requires explicit confirm_full=True")
        if target == "pilot_reproduced" and state.status not in {"pilot_running", "pilot_reproduced"}:
            raise ReproductionTransitionError("pilot cannot be marked reproduced before pilot_running")
        if target == "full_pending_confirmation" and not state.has("pilot_reproduced"):
            raise ReproductionTransitionError("full is blocked until pilot reproduction passes")
        if target not in {"pilot_rejected", "adapter_required"} and self._rank(target) < self._rank(state.status):
            raise ReproductionTransitionError(f"cannot move backwards: {state.status} -> {target}")
        contract = state.contracts.get(target, ReproductionContract())
        missing = [item for item in contract.requires if not state.has(item) and item != target]
        if missing:
            raise ReproductionTransitionError(f"{target} missing required evidence: {', '.join(missing)}")
        previous = state.status
        state.status = target
        state.evidence.update({item: True for item in contract.provides})
        state.attempts += 1
        state.last_error = None
        self.save(state)
        self._event("state_transition", f"Reproduction state {previous} -> {target}", {"from": previous, "to": target, "requires": contract.requires, "provides": contract.provides})
        return state

    def fail(self, reason: str, *, target: str | None = None) -> ReproductionState:
        state = self.initialize()
        state.last_error = reason
        self.save(state)
        self._event("state_failed", reason, {"state": state.status, "target": target})
        return state

    def enqueue(self, stage: str, *, run_id: str, queue_store: ExecutionQueueStore, command: CommandSpec, confirm_full: bool = False) -> ExecutionQueue:
        state = self.initialize()
        if stage == "pilot" and state.status != "debug_passed":
            raise ReproductionTransitionError("pilot queue requires debug_passed")
        if stage == "full":
            if state.status != "pilot_reproduced":
                raise ReproductionTransitionError("full queue requires pilot_reproduced")
            if not confirm_full:
                raise ReproductionTransitionError("full queue requires explicit confirm_full=True")
        candidate = CandidateConfig(candidate_id=f"repro_{self.component_id.replace('.', '_')}_{stage}", base_model="paper_component", scale="research", framework="generic", components=[self.component_id], action_domain="research", execution_action="run_training")
        node = ExperimentNode(node_id=f"node_{candidate.candidate_id}", candidate_config=candidate, data_version="research", command_spec=command, command=command.command, changed_variables={"reproduction_stage": stage})
        plan = ExperimentPlan(plan_id=f"reproduction_{self.component_id}_{stage}", nodes=[node], metadata={"paper_id": self.paper_id, "component_id": self.component_id, "reproduction_stage": stage})
        queue = queue_store.enqueue_from_plan(run_id, plan)
        state.queued_stage = stage
        state.queue_id = queue.items[0].queue_id
        if stage == "pilot":
            state.evidence["pilot_queued"] = True
            state.evidence["pilot_running"] = True
            state.status = "pilot_running"
        elif stage == "full":
            state.evidence["full_confirmed"] = True
            state.evidence["full_confirmation_requested"] = True
            state.evidence["full_pending_confirmation"] = True
            state.status = "full_pending_confirmation"
        self.save(state)
        self._event("queue_enqueued", f"Queued reproduction {stage}", {"queue_id": state.queue_id, "stage": stage})
        return queue

    def reconcile_queue(
        self,
        queue: ExecutionQueue,
        *,
        evidence: dict[str, Any] | None = None,
        local_delta: dict[str, Any] | None = None,
    ) -> ReproductionState:
        """Advance or preserve state from an existing queue result.

        Queue completion alone is not trusted as metric evidence.  Callers
        must provide the corresponding ``pilot_evidence`` or ``full_evidence``
        facts imported by the evidence pipeline.
        """
        state = self.initialize()
        item = next((item for item in queue.items if item.queue_id == state.queue_id), None)
        if item is None:
            raise ReproductionTransitionError("reproduction queue item is missing")
        if item.status in {"failed", "skipped", "needs_resume", "blocked_by_resource", "paused"}:
            return self.fail(item.message or f"reproduction queue ended with status {item.status}", target=state.queued_stage)
        if item.status != "completed":
            return state
        facts = evidence or {}
        if state.queued_stage == "pilot":
            return self.transition("pilot_reproduced", evidence=facts, local_delta=local_delta)
        if state.queued_stage == "full":
            return self.transition("full_reproduced", evidence=facts, local_delta=local_delta, confirm_full=state.has("full_confirmed"))
        return state

    def _event(self, event_type: str, message: str, details: dict[str, Any]) -> None:
        # Research events are intentionally represented as generic stage events
        # to remain compatible with the existing EventType literal and readers.
        mapped = {
            "state_transition": "reproduction_state_transition",
            "state_failed": "reproduction_state_failed",
            "queue_enqueued": "queue_enqueued",
            "registered": "reproduction_state_transition",
        }.get(event_type, "reproduction_state_transition")
        self.event_log.append(run_id=self.run_dir.name, event_type=mapped, message=message, details={"reproduction_event": event_type, **details})

    @classmethod
    def _load_policy(cls, path: Path) -> dict[str, ReproductionContract]:
        with path.open("r", encoding="utf-8-sig") as file:
            raw = yaml.safe_load(file) or {}
        return {str(name): ReproductionContract.model_validate(value) for name, value in raw.get("states", {}).items()}

    @classmethod
    def _rank(cls, status: ReproductionStatus) -> int:
        if status == "pilot_rejected":
            return cls.ORDER.index("pilot_running")
        return cls.ORDER.index(status)


__all__ = ["ReproductionPipeline", "ReproductionTransitionError"]
