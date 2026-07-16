"""Canonical execution plan for one evidence-driven optimization round."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.core.experiment_graph import ExperimentNode, ExperimentPlan, MetricEvidence
from yolo_agent.core.matched_baseline import MatchedBaselineControl, PairedMetricDelta, paired_metric_delta
from yolo_agent.core.yaml_io import YAMLModelMixin


ROUND_EXECUTION_PLAN_SCHEMA_VERSION = "1.0"
RoundStage = Literal["pilot_3", "pilot_10", "full_pending_confirmation", "completed"]
RoundPlanStatus = Literal["ready", "running", "awaiting_evidence", "full_pending_confirmation", "completed", "blocked"]
AssignmentStatus = Literal["active", "pending", "completed", "eliminated", "deferred"]


class RoundStageSpec(BaseModel):
    """One bounded budget stage in the canonical round plan."""

    stage_id: Literal["pilot_3", "pilot_10", "candidate_full"]
    training_profile: str
    epochs: int = Field(ge=1)
    fraction: float = Field(gt=0.0, le=1.0)
    keep_top_k: int | None = Field(default=None, ge=1)
    keep_ratio: float | None = Field(default=None, gt=0.0, le=1.0)


class RoundAssignment(BaseModel):
    """Candidate assignment advanced only from imported evidence."""

    stage_id: Literal["pilot_3", "pilot_10", "candidate_full"]
    candidate_id: str
    source_node_id: str
    execution_node_id: str
    rank: int
    status: AssignmentStatus
    reason: str
    role: Literal["candidate", "baseline_control"] = "candidate"
    score: float | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    paired_delta: float | None = None
    matched_control_hash: str | None = None


class RoundAblationNode(BaseModel):
    """Ablation constraint embedded in the authoritative plan."""

    node_id: str
    candidate_id: str
    parent_id: str | None = None
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    valid: bool = True
    reason: str = "single_variable_ablation"


class SurvivorDecision(BaseModel):
    """Evidence-backed promotion or elimination decision."""

    from_stage: Literal["pilot_3", "pilot_10"]
    candidate_id: str
    source_node_id: str
    promoted: bool
    rank: int
    metric_name: str
    metric_value: float
    paired_delta: float
    matched_control_hash: str
    reason: str


class RoundExecutionPlan(BaseModel, YAMLModelMixin):
    """The only executable planning authority for an automatic round."""

    schema_version: str = ROUND_EXECUTION_PLAN_SCHEMA_VERSION
    run_id: str
    round_id: str
    objective_hash: str | None = None
    decision_context_hash: str | None = None
    source_decision_bundle_hash: str | None = None
    source_policy_evaluation_hash: str | None = None
    selected_recipes: list[dict[str, Any]] = Field(default_factory=list)
    critic_results: list[dict[str, Any]] = Field(default_factory=list)
    stages: list[RoundStageSpec] = Field(default_factory=list)
    assignments: list[RoundAssignment] = Field(default_factory=list)
    ablation_nodes: list[RoundAblationNode] = Field(default_factory=list)
    active_stage: RoundStage = "pilot_3"
    survivor_decisions: list[SurvivorDecision] = Field(default_factory=list)
    execution_nodes: list[ExperimentNode] = Field(default_factory=list)
    deferred_nodes: list[ExperimentNode] = Field(default_factory=list)
    eliminated_node_ids: list[str] = Field(default_factory=list)
    evidence_requirements: dict[str, list[str]] = Field(default_factory=dict)
    primary_metric: str = "map50_95"
    status: RoundPlanStatus = "ready"
    blocked_reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_authority(self) -> "RoundExecutionPlan":
        active_ids = {item.execution_node_id for item in self.assignments if item.status == "active"}
        execution_ids = {node.node_id for node in self.execution_nodes}
        if execution_ids != active_ids:
            raise ValueError("execution_nodes must exactly match active round assignments")
        if self.active_stage == "full_pending_confirmation" and self.execution_nodes:
            raise ValueError("full_pending_confirmation cannot contain executable nodes")
        return self

    def plan_hash(self) -> str:
        """Return a stable semantic hash used for queue invalidation."""
        payload = self.model_dump(mode="json", exclude={"created_at", "updated_at"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def experiment_projection(self) -> ExperimentPlan:
        """Project the active stage for legacy readers without granting authority."""
        plan = ExperimentPlan(
            plan_id=f"{self.round_id}_{self.active_stage}_projection",
            nodes=self.execution_nodes,
            metadata={
                "source": "RoundExecutionPlan",
                "projection_only": True,
                "source_round_plan_hash": self.plan_hash(),
                "active_stage": self.active_stage,
            },
        )
        plan.metadata["plan_hash"] = plan.plan_hash()
        return plan

    def ablation_projection(self) -> dict[str, Any]:
        """Return the non-authoritative ablation view."""
        return {
            "projection_only": True,
            "source_round_plan_hash": self.plan_hash(),
            "nodes": [item.model_dump(mode="json") for item in self.ablation_nodes],
        }

    def budget_projection(self) -> dict[str, Any]:
        """Return the non-authoritative budget/halving view."""
        return {
            "projection_only": True,
            "source_round_plan_hash": self.plan_hash(),
            "active_stage": self.active_stage,
            "status": self.status,
            "stages": [item.model_dump(mode="json") for item in self.stages],
            "assignments": [item.model_dump(mode="json") for item in self.assignments],
            "survivor_decisions": [item.model_dump(mode="json") for item in self.survivor_decisions],
        }

    def reconcile(self, metric_records: list[MetricEvidence]) -> bool:
        """Advance pilot stages only after every active node has trusted evidence."""
        if self.active_stage not in {"pilot_3", "pilot_10"}:
            return False
        active = [
            item
            for item in self.assignments
            if item.status == "active" and item.stage_id == self.active_stage and item.role == "candidate"
        ]
        controls = [
            item
            for item in self.assignments
            if item.status == "active" and item.stage_id == self.active_stage and item.role == "baseline_control"
        ]
        paired, diagnostics = _paired_metric_values(active, metric_records, self.primary_metric)
        if len(paired) != len(active):
            self.status = "awaiting_evidence"
            missing = [item.execution_node_id for item in active if item.execution_node_id not in paired]
            reasons = sorted(
                {
                    reason
                    for node_id in missing
                    for reason in diagnostics.get(node_id, MatchedBaselineControl(
                        candidate_id="unknown", candidate_node_id=node_id
                    )).missing_dimensions
                }
                | {
                    reason
                    for node_id in missing
                    for reason in diagnostics.get(node_id, MatchedBaselineControl(
                        candidate_id="unknown", candidate_node_id=node_id
                    )).mismatch_reasons
                }
            )
            detail = f" ({', '.join(reasons)})" if reasons else ""
            self.blocked_reason = f"needs matched baseline {self.primary_metric} for: {', '.join(missing)}{detail}"
            self.updated_at = datetime.now(timezone.utc)
            return False

        stage = next(item for item in self.stages if item.stage_id == self.active_stage)
        ranked = sorted(active, key=lambda item: paired[item.execution_node_id].effect_delta, reverse=True)
        keep_count = _keep_count(stage, len(ranked))
        kept = ranked[:keep_count]
        for rank, item in enumerate(ranked, start=1):
            promoted = item in kept
            item.status = "completed" if promoted else "eliminated"
            delta = paired[item.execution_node_id]
            item.score = delta.effect_delta
            item.metric_name = self.primary_metric
            item.metric_value = delta.candidate_value
            item.paired_delta = delta.paired_delta
            item.matched_control_hash = delta.match_key_hash
            item.reason = "survived_paired_delta_rank" if promoted else "eliminated_by_paired_delta_rank"
            if not promoted:
                self.eliminated_node_ids.append(item.source_node_id)
            self.survivor_decisions.append(
                SurvivorDecision(
                    from_stage=self.active_stage,
                    candidate_id=item.candidate_id,
                    source_node_id=item.source_node_id,
                    promoted=promoted,
                    rank=rank,
                    metric_name=self.primary_metric,
                    metric_value=delta.candidate_value,
                    paired_delta=delta.paired_delta,
                    matched_control_hash=delta.match_key_hash,
                    reason=item.reason,
                )
            )

        for control in controls:
            control.status = "completed"
            control.reason = "matched_baseline_control_completed"

        if self.active_stage == "pilot_10":
            for rank, item in enumerate(kept, start=1):
                self.assignments.append(
                    RoundAssignment(
                        stage_id="candidate_full",
                        candidate_id=item.candidate_id,
                        source_node_id=item.source_node_id,
                        execution_node_id=f"{item.source_node_id}__candidate_full",
                        rank=rank,
                        status="deferred",
                        reason="pilot_10_survivor_waiting_for_explicit_full_confirmation",
                        score=paired[item.execution_node_id].effect_delta,
                        metric_name=self.primary_metric,
                        metric_value=paired[item.execution_node_id].candidate_value,
                        paired_delta=paired[item.execution_node_id].paired_delta,
                        matched_control_hash=paired[item.execution_node_id].match_key_hash,
                    )
                )
            self.active_stage = "full_pending_confirmation"
            self.status = "full_pending_confirmation"
            self.blocked_reason = "candidate_full requires explicit confirmation"
            self.execution_nodes = []
            self.updated_at = datetime.now(timezone.utc)
            return True

        source_nodes = {node.candidate_config.candidate_id: node for node in [*self.execution_nodes, *self.deferred_nodes]}
        next_nodes: list[ExperimentNode] = []
        next_control_source = next((node for node in self.deferred_nodes if _is_baseline_control_node(node)), None)
        if next_control_source is None:
            self.status = "awaiting_evidence"
            self.blocked_reason = "missing baseline control source for pilot_10"
            self.execution_nodes = []
            return False
        next_control = _node_for_stage(next_control_source, "pilot_10", epochs=10, fraction=0.1)
        self.assignments.append(
            RoundAssignment(
                stage_id="pilot_10",
                candidate_id=next_control.candidate_config.candidate_id,
                source_node_id=next_control_source.node_id,
                execution_node_id=next_control.node_id,
                rank=0,
                status="active",
                role="baseline_control",
                reason="matched_baseline_control_for_pilot_10",
            )
        )
        for rank, item in enumerate(kept, start=1):
            source = source_nodes[item.candidate_id]
            node = _node_for_stage(source, "pilot_10", epochs=10, fraction=0.1)
            next_nodes.append(node)
            self.assignments.append(
                RoundAssignment(
                    stage_id="pilot_10",
                    candidate_id=item.candidate_id,
                    source_node_id=item.source_node_id,
                    execution_node_id=node.node_id,
                    rank=rank,
                    status="active",
                    reason="promoted_from_pilot_3_by_imported_evidence",
                )
            )
        self.active_stage = "pilot_10"
        self.execution_nodes = [next_control, *next_nodes]
        self.status = "ready"
        self.blocked_reason = ""
        self.updated_at = datetime.now(timezone.utc)
        return True


def build_round_execution_plan(
    *,
    run_id: str,
    nodes: list[ExperimentNode],
    ranks: dict[str, int] | None = None,
    objective_hash: str | None = None,
    decision_context_hash: str | None = None,
    source_decision_bundle_hash: str | None = None,
    source_policy_evaluation_hash: str | None = None,
    primary_metric: str = "map50_95",
    baseline_control_node: ExperimentNode | None = None,
) -> RoundExecutionPlan:
    """Build a canonical plan that initially activates only pilot_3 nodes."""
    rank_map = ranks or {}
    valid_nodes: list[ExperimentNode] = []
    ablations: list[RoundAblationNode] = []
    for node in nodes:
        changed = dict(node.changed_variables)
        valid = len(changed) <= 1
        ablations.append(
            RoundAblationNode(
                node_id=node.node_id,
                candidate_id=node.candidate_config.candidate_id,
                parent_id=node.parent_id,
                changed_variables=changed,
                valid=valid,
                reason="single_variable_ablation" if valid else "multiple_changed_variables_require_coupled_recipe",
            )
        )
        if valid:
            valid_nodes.append(node)
    ordered = sorted(valid_nodes, key=lambda node: rank_map.get(node.candidate_config.candidate_id, 10**6))
    execution_nodes = [_node_for_stage(node, "pilot_3", epochs=3, fraction=0.1) for node in ordered]
    assignments = [
        RoundAssignment(
            stage_id="pilot_3",
            candidate_id=node.candidate_config.candidate_id,
            source_node_id=node.node_id,
            execution_node_id=execution.node_id,
            rank=index,
            status="active",
            reason="selected_by_guarded_budget_for_pilot_3",
        )
        for index, (node, execution) in enumerate(zip(ordered, execution_nodes), start=1)
    ]
    deferred_nodes = list(ordered)
    if baseline_control_node is not None and execution_nodes:
        control_source = _mark_baseline_control(baseline_control_node)
        control_execution = _node_for_stage(control_source, "pilot_3", epochs=3, fraction=0.1)
        execution_nodes.insert(0, control_execution)
        deferred_nodes.insert(0, control_source)
        assignments.insert(
            0,
            RoundAssignment(
                stage_id="pilot_3",
                candidate_id=control_execution.candidate_config.candidate_id,
                source_node_id=control_source.node_id,
                execution_node_id=control_execution.node_id,
                rank=0,
                status="active",
                role="baseline_control",
                reason="matched_baseline_control_for_pilot_3",
            ),
        )
    return RoundExecutionPlan(
        run_id=run_id,
        round_id=f"{run_id}_round",
        objective_hash=objective_hash,
        decision_context_hash=decision_context_hash,
        source_decision_bundle_hash=source_decision_bundle_hash,
        source_policy_evaluation_hash=source_policy_evaluation_hash,
        stages=[
            RoundStageSpec(stage_id="pilot_3", training_profile="pilot", epochs=3, fraction=0.1, keep_ratio=0.5),
            RoundStageSpec(stage_id="pilot_10", training_profile="pilot", epochs=10, fraction=0.1, keep_top_k=2),
            RoundStageSpec(stage_id="candidate_full", training_profile="candidate_full", epochs=100, fraction=1.0, keep_top_k=1),
        ],
        assignments=assignments,
        ablation_nodes=ablations,
        execution_nodes=execution_nodes,
        deferred_nodes=deferred_nodes,
        primary_metric=primary_metric,
        status="ready" if execution_nodes else "blocked",
        blocked_reason="" if execution_nodes else "no valid guarded ablation nodes",
    )


def _node_for_stage(node: ExperimentNode, stage_id: str, *, epochs: int, fraction: float) -> ExperimentNode:
    spec = node.command_spec
    if spec is not None:
        argv = _replace_cli_values(spec.argv, {"epochs": epochs, "fraction": fraction})
        spec = spec.model_copy(
            update={
                "argv": argv,
                "command": argv[0] if argv else spec.command,
                "args": argv[1:] if argv else spec.args,
                "metadata": {
                    **spec.metadata,
                    "round_stage": stage_id,
                    "training_budget_profile": "pilot" if stage_id.startswith("pilot") else "candidate_full",
                    "epochs": epochs,
                    "data_fraction": fraction,
                },
            }
        )
    node_id = f"{node.node_id}__{stage_id}"
    return node.model_copy(
        update={
            "node_id": node_id,
            "command_spec": spec,
            "command": spec.display() if spec is not None else node.command,
            "changed_variables": dict(node.changed_variables),
        }
    )


def _replace_cli_values(argv: list[str], values: dict[str, int | float]) -> list[str]:
    result = list(argv)
    for key, value in values.items():
        prefix = f"{key}="
        replacement = f"{key}={value}"
        for index, arg in enumerate(result):
            if arg.startswith(prefix):
                result[index] = replacement
                break
        else:
            result.append(replacement)
    return result


def _metric_values(
    assignments: list[RoundAssignment],
    records: list[MetricEvidence],
    metric_name: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for assignment in assignments:
        matches = [
            record
            for record in records
            if record.node_id == assignment.execution_node_id
            and record.candidate_id == assignment.candidate_id
            and record.metric_name == metric_name
            and record.verified
            and isinstance(record.value, (int, float))
        ]
        if matches:
            latest = max(matches, key=lambda record: record.created_at)
            values[assignment.execution_node_id] = float(latest.value)
    return values


def _paired_metric_values(
    assignments: list[RoundAssignment],
    records: list[MetricEvidence],
    metric_name: str,
) -> tuple[dict[str, PairedMetricDelta], dict[str, MatchedBaselineControl]]:
    baseline_records = [record for record in records if record.evidence_role == "baseline_reference"]
    deltas: dict[str, PairedMetricDelta] = {}
    diagnostics: dict[str, MatchedBaselineControl] = {}
    for assignment in assignments:
        matches = [
            record
            for record in records
            if record.node_id == assignment.execution_node_id
            and record.candidate_id == assignment.candidate_id
            and record.metric_name == metric_name
            and record.verified
            and record.evidence_role == "current_observation"
            and record.inheritance_depth == 0
        ]
        if not matches:
            diagnostics[assignment.execution_node_id] = MatchedBaselineControl(
                candidate_id=assignment.candidate_id,
                candidate_node_id=assignment.execution_node_id,
                mismatch_reasons=["missing_current_candidate_metric"],
            )
            continue
        candidate = max(matches, key=lambda record: record.created_at)
        control, delta = paired_metric_delta(candidate, baseline_records)
        diagnostics[assignment.execution_node_id] = control
        if delta is not None:
            deltas[assignment.execution_node_id] = delta
    return deltas, diagnostics


def _mark_baseline_control(node: ExperimentNode) -> ExperimentNode:
    spec = node.command_spec
    if spec is not None:
        spec = spec.model_copy(
            update={
                "metadata": {
                    **spec.metadata,
                    "matched_baseline_control": True,
                }
            }
        )
    return node.model_copy(update={"command_spec": spec, "command": spec.display() if spec is not None else node.command})


def _is_baseline_control_node(node: ExperimentNode) -> bool:
    return bool(node.command_spec and node.command_spec.metadata.get("matched_baseline_control"))


def _keep_count(stage: RoundStageSpec, count: int) -> int:
    if count <= 0:
        return 0
    if stage.keep_top_k is not None:
        return min(count, stage.keep_top_k)
    if stage.keep_ratio is not None:
        return max(1, min(count, math.ceil(count * stage.keep_ratio)))
    return count
