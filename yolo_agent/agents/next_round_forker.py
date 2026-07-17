"""Materialize next-round plans into child loop runs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from yolo_agent.agents.loop_evidence import missing_evidence_from_status
from yolo_agent.agents.loop_io import read_yaml, write_yaml
from yolo_agent.agents.run_initializer import attach_dataset_manifest_to_context, inherit_dataset_manifest_to_context
from yolo_agent.core.dataset_versioning import DatasetDiff, DatasetVersionManifest, diff_manifests
from yolo_agent.core.loop_state import LoopState
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.stage_contract import LoopStageContracts


class NextRoundForker:
    """Create a child run from a parent's next_round.yaml."""

    def __init__(self, context: RunContext, policy: LoopStageContracts) -> None:
        self.context = context
        self.policy = policy

    def fork(self, new_run_id: str, orchestrator_cls: type[Any]) -> Any:
        """Materialize the next round into a fresh run that inherits loop context."""
        if not new_run_id or any(separator in new_run_id for separator in ("/", "\\")):
            raise ValueError("new_run_id must be a non-empty single path segment.")
        next_round_path = self.context.artifact_path("next_round.yaml")
        if not next_round_path.is_file():
            raise FileNotFoundError(f"Missing next_round.yaml: {next_round_path}")
        next_round = read_yaml(next_round_path)
        new_run_dir = self.context.run_root / new_run_id
        if new_run_dir.exists():
            raise FileExistsError(f"Run already exists: {new_run_dir}")
        missing_evidence = missing_evidence_from_status(self.context.artifact_path("evidence_status.json"))
        context = self._child_context(new_run_id, next_round_path, next_round, missing_evidence)
        context.ensure_dirs()
        dataset_manifest_path = self._attach_child_dataset_manifest(context, next_round)
        dataset_diff_path, dataset_diff = self._write_dataset_diff(context)
        inherited_next_round_path = context.artifact_path("parent_next_round.yaml")
        shutil.copy2(next_round_path, inherited_next_round_path)
        fork_context_path = context.artifact_path("fork_context.yaml")
        write_yaml(
            fork_context_path,
            self._fork_context_payload(
                child=context,
                next_round=next_round,
                next_round_path=next_round_path,
                inherited_next_round_path=inherited_next_round_path,
                missing_evidence=missing_evidence,
                dataset_diff_path=dataset_diff_path,
                dataset_diff=dataset_diff,
            ),
        )
        context.to_yaml()
        context.to_json()
        state = LoopState.create(
            new_run_id,
            self.policy.stage_order,
            dataset_version=context.dataset_version,
            task_spec=context.task_path,
        )
        state.mark(
            "init",
            "completed",
            "Forked from parent run.",
            {
                "run_context": context.run_dir / "run_context.yaml",
                "dataset_manifest": dataset_manifest_path,
                "fork_context": fork_context_path,
                "parent_next_round": inherited_next_round_path,
                **({"dataset_diff": dataset_diff_path} if dataset_diff_path is not None else {}),
            },
        )
        state.to_yaml(context.run_dir / "loop_state.yaml")
        orchestrator = orchestrator_cls(context, state)
        orchestrator.evidence_store.log_config(
            new_run_id,
            {
                "run_context": context.model_dump(mode="json"),
                "forked_from": self.context.run_id,
                "parent_next_round": next_round,
            },
        )
        orchestrator.artifacts.record(
            "init",
            {
                "run_context": context.run_dir / "run_context.yaml",
                "loop_state": context.run_dir / "loop_state.yaml",
                "dataset_manifest": dataset_manifest_path,
                "fork_context": fork_context_path,
                "parent_next_round": inherited_next_round_path,
                **({"dataset_diff": dataset_diff_path} if dataset_diff_path is not None else {}),
            },
        )
        orchestrator.event_log.append(
            run_id=new_run_id,
            event_type="run_initialized",
            stage="init",
            status="completed",
            message=f"Forked from parent run {self.context.run_id}.",
            artifacts={
                "run_context": context.run_dir / "run_context.yaml",
                "dataset_manifest": dataset_manifest_path,
                "fork_context": fork_context_path,
                "parent_next_round": inherited_next_round_path,
                **({"dataset_diff": dataset_diff_path} if dataset_diff_path is not None else {}),
            },
            details={
                "parent_run_id": self.context.run_id,
                "inherited_evidence_required": list(next_round.get("evidence_required", [])),
                "inherited_missing_evidence": missing_evidence,
                "recommended_stage": next_round.get("recommended_stage"),
                "parent_stop_reason": next_round.get("stop_reason"),
                "dataset_diff": dataset_diff.model_dump(mode="json") if dataset_diff is not None else None,
            },
        )
        orchestrator.evidence.record_lineage(
            parent_run_id=self.context.run_id,
            inherited_missing_evidence=missing_evidence,
            current_missing_evidence=missing_evidence,
            metadata={
                "parent_next_round_path": inherited_next_round_path.as_posix(),
                "recommended_stage": next_round.get("recommended_stage"),
                "parent_stop_reason": next_round.get("stop_reason"),
                "dataset_diff_path": dataset_diff_path.as_posix() if dataset_diff_path is not None else None,
                "dataset_diff": dataset_diff.model_dump(mode="json") if dataset_diff is not None else None,
            },
        )
        return orchestrator

    def _attach_child_dataset_manifest(
        self,
        child: RunContext,
        next_round: dict[str, Any],
    ) -> Path:
        next_dataset_version = _next_dataset_version(self.context, next_round)
        if next_dataset_version:
            child.dataset_version = next_dataset_version
            return attach_dataset_manifest_to_context(child)
        return inherit_dataset_manifest_to_context(self.context, child)

    def _write_dataset_diff(self, child: RunContext) -> tuple[Path | None, DatasetDiff | None]:
        parent_manifest_path = self.context.dataset_manifest_path
        child_manifest_path = child.dataset_manifest_path
        if parent_manifest_path is None or child_manifest_path is None:
            return None, None
        if not parent_manifest_path.is_file() or not child_manifest_path.is_file():
            return None, None
        parent_sha = self.context.dataset_manifest_sha256
        child_sha = child.dataset_manifest_sha256
        if parent_sha is not None and child_sha is not None and parent_sha == child_sha:
            return None, None
        parent_manifest = DatasetVersionManifest.from_json(parent_manifest_path)
        child_manifest = DatasetVersionManifest.from_json(child_manifest_path)
        diff = diff_manifests(parent_manifest, child_manifest)
        diff_path = child.artifact_path("dataset_diff.json")
        diff.to_json(diff_path)
        child.metadata["dataset_diff_path"] = diff_path.as_posix()
        child.metadata["dataset_diff"] = diff.model_dump(mode="json")
        return diff_path, diff

    def _child_context(
        self,
        new_run_id: str,
        next_round_path: Path,
        next_round: dict[str, Any],
        missing_evidence: list[str],
    ) -> RunContext:
        return RunContext(
            run_id=new_run_id,
            run_root=self.context.run_root,
            task_path=self.context.task_path,
            data_yaml=self.context.data_yaml,
            component_path=self.context.component_path,
            search_space_path=self.context.search_space_path,
            loop_policy_path=self.context.loop_policy_path,
            predictions_path=self.context.predictions_path,
            detection_errors_path=self.context.detection_errors_path,
            metrics_input_path=None,
            dataset_version=_next_dataset_version(self.context, next_round) or self.context.dataset_version,
            dataset_root=self.context.dataset_root,
            dataset_version_store_path=self.context.dataset_version_store_path,
            seed=self.context.seed,
            metadata={
                "parent_run_id": self.context.run_id,
                "parent_run_dir": self.context.run_dir.as_posix(),
                "parent_next_round_path": next_round_path.as_posix(),
                "inherited_next_round_path": (self.context.run_root / new_run_id / "artifacts" / "parent_next_round.yaml").as_posix(),
                "inherited_evidence_required": list(next_round.get("evidence_required", [])),
                "inherited_missing_evidence": missing_evidence,
                "inherited_changed_variables": next_round.get("changed_variables", {}),
                "inherited_guardrails": list(next_round.get("guardrails", [])),
                "inherited_parent_best_candidate": next_round.get("parent_best_candidate"),
                "inherited_unresolved_diagnoses": list(next_round.get("unresolved_diagnoses", [])),
                "inherited_improved_errors": list(next_round.get("improved_errors", [])),
                "inherited_unresolved_errors": list(next_round.get("unresolved_errors", [])),
                "inherited_regressed_errors": list(next_round.get("regressed_errors", [])),
                "inherited_error_fact_delta": next_round.get("error_fact_delta", {}),
                "inherited_improved_error_facts": list(next_round.get("improved_error_facts", [])),
                "inherited_unresolved_error_facts": list(next_round.get("unresolved_error_facts", [])),
                "inherited_regressed_error_facts": list(next_round.get("regressed_error_facts", [])),
                "inherited_next_error_actions": list(next_round.get("next_error_actions", [])),
                "inherited_effective_error_actions": list(next_round.get("effective_error_actions", [])),
                "inherited_current_round_focus": list(next_round.get("current_round_focus", [])),
                "inherited_current_round_error_actions": list(next_round.get("current_round_error_actions", [])),
                "inherited_proposal_mode": next_round.get("proposal_mode"),
                "inherited_proposal_budget_profiles_allowed": list(next_round.get("proposal_budget_profiles_allowed", [])),
                "inherited_proposal_budget_profiles_blocked": list(next_round.get("proposal_budget_profiles_blocked", [])),
                "inherited_proposal_required_bindings": list(next_round.get("proposal_required_bindings", [])),
                "inherited_newly_available_evidence": list(next_round.get("newly_available_evidence", [])),
                "recommended_stage": next_round.get("recommended_stage"),
                "parent_stop_reason": next_round.get("stop_reason"),
                "parent_evidence_delta": next_round.get("evidence_delta", {}),
                "optimization_objective_path": self.context.metadata.get("optimization_objective_path"),
                "optimization_objective_hash": self.context.metadata.get("optimization_objective_hash"),
                "baseline_protocol_hash": self.context.metadata.get("baseline_protocol_hash"),
                "research_snapshot_hash": self.context.metadata.get("research_snapshot_hash"),
                "research_snapshot_path": self.context.metadata.get("research_snapshot_path"),
                "research_snapshot_verified": self.context.metadata.get("research_snapshot_verified", False),
                "paper_intelligence": self.context.metadata.get("paper_intelligence", "unavailable"),
                "unavailable_reason": self.context.metadata.get("unavailable_reason"),
                "research_network_allowed": False,
                "maturity_summary": self.context.metadata.get("maturity_summary", {}),
                "parent_dataset_version": self.context.dataset_version,
                "parent_dataset_manifest_sha256": self.context.dataset_manifest_sha256,
                "next_dataset_version": _next_dataset_version(self.context, next_round),
            },
        )

    def _fork_context_payload(
        self,
        child: RunContext,
        next_round: dict[str, Any],
        next_round_path: Path,
        inherited_next_round_path: Path,
        missing_evidence: list[str],
        dataset_diff_path: Path | None = None,
        dataset_diff: DatasetDiff | None = None,
    ) -> dict[str, Any]:
        return {
            "parent_run_id": self.context.run_id,
            "parent_run_dir": self.context.run_dir.as_posix(),
            "parent_next_round_path": next_round_path.as_posix(),
            "inherited_next_round_path": inherited_next_round_path.as_posix(),
            "inherited_evidence_required": list(next_round.get("evidence_required", [])),
            "inherited_missing_evidence": missing_evidence,
            "inherited_changed_variables": next_round.get("changed_variables", {}),
            "inherited_guardrails": list(next_round.get("guardrails", [])),
            "inherited_parent_best_candidate": next_round.get("parent_best_candidate"),
            "inherited_unresolved_diagnoses": list(next_round.get("unresolved_diagnoses", [])),
            "inherited_improved_errors": list(next_round.get("improved_errors", [])),
            "inherited_unresolved_errors": list(next_round.get("unresolved_errors", [])),
            "inherited_regressed_errors": list(next_round.get("regressed_errors", [])),
            "inherited_error_fact_delta": next_round.get("error_fact_delta", {}),
            "inherited_improved_error_facts": list(next_round.get("improved_error_facts", [])),
            "inherited_unresolved_error_facts": list(next_round.get("unresolved_error_facts", [])),
            "inherited_regressed_error_facts": list(next_round.get("regressed_error_facts", [])),
            "inherited_next_error_actions": list(next_round.get("next_error_actions", [])),
            "inherited_effective_error_actions": list(next_round.get("effective_error_actions", [])),
            "inherited_current_round_focus": list(next_round.get("current_round_focus", [])),
            "inherited_current_round_error_actions": list(next_round.get("current_round_error_actions", [])),
            "inherited_proposal_mode": next_round.get("proposal_mode"),
            "inherited_proposal_budget_profiles_allowed": list(next_round.get("proposal_budget_profiles_allowed", [])),
            "inherited_proposal_budget_profiles_blocked": list(next_round.get("proposal_budget_profiles_blocked", [])),
            "inherited_proposal_required_bindings": list(next_round.get("proposal_required_bindings", [])),
            "inherited_newly_available_evidence": list(next_round.get("newly_available_evidence", [])),
            "recommended_stage": next_round.get("recommended_stage"),
            "parent_stop_reason": next_round.get("stop_reason"),
            "parent_evidence_delta": next_round.get("evidence_delta", {}),
            "optimization_objective_path": self.context.metadata.get("optimization_objective_path"),
            "optimization_objective_hash": self.context.metadata.get("optimization_objective_hash"),
            "baseline_protocol_hash": self.context.metadata.get("baseline_protocol_hash"),
            "research_snapshot_hash": self.context.metadata.get("research_snapshot_hash"),
            "research_snapshot_path": self.context.metadata.get("research_snapshot_path"),
            "research_snapshot_verified": self.context.metadata.get("research_snapshot_verified", False),
            "paper_intelligence": self.context.metadata.get("paper_intelligence", "unavailable"),
            "unavailable_reason": self.context.metadata.get("unavailable_reason"),
            "research_network_allowed": False,
            "maturity_summary": self.context.metadata.get("maturity_summary", {}),
            "parent_dataset_version": self.context.dataset_version,
            "parent_dataset_manifest_sha256": self.context.dataset_manifest_sha256,
            "dataset_version": child.dataset_version,
            "dataset_manifest_sha256": child.dataset_manifest_sha256,
            "dataset_diff_path": dataset_diff_path.as_posix() if dataset_diff_path is not None else None,
            "dataset_diff": dataset_diff.model_dump(mode="json") if dataset_diff is not None else None,
        }


def _next_dataset_version(context: RunContext, next_round: dict[str, Any]) -> str | None:
    value = next_round.get("next_dataset_version") or context.metadata.get("active_learning_next_dataset_version")
    return str(value) if value else None
