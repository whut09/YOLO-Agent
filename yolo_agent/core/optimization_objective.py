"""Typed optimization objective shared by planning, promotion, and stopping."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentPlan, MetricEvidence
from yolo_agent.core.matched_baseline import PairedMetricDelta, paired_metric_delta
from yolo_agent.core.task_spec import MetricName
from yolo_agent.core.yaml_io import YAMLModelMixin

if TYPE_CHECKING:
    from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig


DeltaMode = Literal["absolute", "relative"]


class OptimizationObjective(BaseModel, YAMLModelMixin):
    """Executable accuracy target, guard metrics, confidence, and budget."""

    schema_version: str = "1.0"
    goal_expression: str = "+2map"
    primary_metric: MetricName = "map50_95"
    delta_mode: DeltaMode = "absolute"
    target_absolute_delta: float | None = Field(default=0.02, gt=0.0)
    target_relative_delta: float | None = Field(default=None, gt=0.0)
    baseline_run_id: str
    baseline_candidate_id: str
    baseline_protocol_hash: str
    fixed_imgsz: int = Field(default=640, ge=640, le=640)
    max_latency_regression: float = Field(default=0.05, ge=0.0)
    max_model_size_regression: float = Field(default=0.10, ge=0.0)
    confirmation_seeds: int = Field(default=3, ge=2)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    max_gpu_hours: float = Field(default=200.0, gt=0.0)
    max_pilot_rounds: int = Field(default=30, ge=1)
    no_improvement_patience: int = Field(default=5, ge=1)
    minimum_pilot_delta: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_delta(self) -> "OptimizationObjective":
        if self.delta_mode == "absolute":
            if self.target_absolute_delta is None or self.target_relative_delta is not None:
                raise ValueError("absolute objectives require only target_absolute_delta")
        elif self.target_relative_delta is None or self.target_absolute_delta is not None:
            raise ValueError("relative objectives require only target_relative_delta")
        return self

    @property
    def objective_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def required_delta(self, baseline_value: float | None = None) -> float | None:
        if self.delta_mode == "absolute":
            return self.target_absolute_delta
        if baseline_value is None or self.target_relative_delta is None:
            return None
        return baseline_value * self.target_relative_delta


class OptimizationObjectiveStatus(BaseModel):
    """Current progress and deterministic stop decision for an objective."""

    objective_hash: str
    primary_metric: str
    baseline_value: float | None = None
    baseline_profile: str | None = None
    baseline_trusted: bool = False
    best_candidate_id: str | None = None
    best_value: float | None = None
    observed_delta: float | None = None
    required_delta: float | None = None
    target_reached: bool = False
    confirmed: bool = False
    success: bool = False
    candidate_seed_count: int = 0
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    latency_regression: float | None = None
    model_size_regression: float | None = None
    guardrails_passed: bool = False
    gpu_hours_used: float = 0.0
    gpu_budget_remaining: float = 0.0
    completed_pilot_rounds: int = 0
    no_improvement_rounds: int = 0
    should_stop: bool = False
    stop_reason: str = "continue_search"
    blockers: list[str] = Field(default_factory=list)


def parse_optimization_goal(
    expression: str,
    *,
    baseline_run_id: str,
    baseline_candidate_id: str,
    baseline_protocol_hash: str,
    defaults: dict[str, Any] | None = None,
) -> OptimizationObjective:
    """Parse friendly goals such as ``+2map`` into an executable objective."""
    text = expression.strip().lower().replace(" ", "")
    match = re.fullmatch(r"\+(\d+(?:\.\d+)?)(%|pp|points?)?(map50-95|map50_95|map50|map)", text)
    if match is None:
        raise ValueError(
            "Unsupported goal expression. Use forms such as +2map, +0.02map50_95, +2ppmap50, or +2%map."
        )
    raw_value = float(match.group(1))
    unit = match.group(2) or ""
    metric_token = match.group(3)
    metric: MetricName = "map50" if metric_token == "map50" else "map50_95"
    values = dict(defaults or {})
    values.update(
        {
            "goal_expression": expression,
            "primary_metric": metric,
            "baseline_run_id": baseline_run_id,
            "baseline_candidate_id": baseline_candidate_id,
            "baseline_protocol_hash": baseline_protocol_hash,
        }
    )
    if unit == "%":
        values.update(
            delta_mode="relative",
            target_absolute_delta=None,
            target_relative_delta=raw_value / 100.0,
        )
    else:
        absolute_delta = raw_value / 100.0 if unit in {"pp", "point", "points"} or raw_value >= 1.0 else raw_value
        values.update(
            delta_mode="absolute",
            target_absolute_delta=absolute_delta,
            target_relative_delta=None,
        )
    return OptimizationObjective.model_validate(values)


def build_baseline_protocol_hash(
    *,
    model: str,
    data_yaml: Path | str,
    training_config: UltralyticsTrainingConfig,
    dataset_version: str,
    dataset_manifest_sha256: str | None,
) -> str:
    """Hash the full-baseline comparison protocol, independent of the active profile."""
    baseline = training_config.budget_profiles["baseline_full"]
    payload = {
        "model": model,
        "data_yaml": Path(data_yaml).resolve().as_posix(),
        "dataset_version": dataset_version,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "task": training_config.task,
        "imgsz": training_config.imgsz,
        "profile": "baseline_full",
        "epochs": baseline.epochs,
        "fraction": baseline.fraction,
        "batch_policy": baseline.batch,
        "val": baseline.val,
        "optimizer": training_config.optimizer,
        "amp": training_config.amp,
        "patience": training_config.patience,
        "overrides": {**training_config.overrides, **baseline.overrides},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_optimization_objective(path: Path | str | None) -> OptimizationObjective | None:
    if path is None:
        return None
    objective_path = Path(path)
    return OptimizationObjective.from_yaml(objective_path) if objective_path.is_file() else None


def evaluate_optimization_objective(
    objective: OptimizationObjective,
    *,
    run_root: Path | str,
    base_run_id: str,
) -> OptimizationObjectiveStatus:
    """Evaluate objective progress from local, current-run node evidence only."""
    root = Path(run_root)
    run_dirs = _objective_run_dirs(root, base_run_id)
    records: list[tuple[str, MetricEvidence]] = []
    node_profiles: dict[tuple[str, str], str] = {}
    node_seeds: dict[tuple[str, str], int] = {}
    for run_dir in run_dirs:
        run_id = run_dir.name
        evidence = EvidenceStore(root).load_run(run_id)
        records.extend(
            (run_id, item)
            for item in evidence.metric_records
            if item.verified
            and (
                item.evidence_role == "baseline_reference"
                or (
                    item.evidence_role == "current_observation"
                    and item.inheritance_depth == 0
                    and not item.source.startswith("inherited:")
                )
            )
        )
        profile_map, seed_map = _plan_node_metadata(run_dir)
        node_profiles.update({(run_id, node_id): value for node_id, value in profile_map.items()})
        node_seeds.update({(run_id, node_id): value for node_id, value in seed_map.items()})

    protocol_nodes = {
        (run_id, item.node_id)
        for run_id, item in records
        if item.metric_name == "baseline_protocol_hash" and item.value == objective.baseline_protocol_hash
    }
    metric_records = [
        (run_id, item)
        for run_id, item in records
        if item.metric_name == objective.primary_metric
        and isinstance(item.value, (int, float))
        and ((run_id, item.node_id) in protocol_nodes or not protocol_nodes)
    ]
    baseline_records = [
        (run_id, item)
        for run_id, item in metric_records
        if run_id == objective.baseline_run_id and item.candidate_id == objective.baseline_candidate_id
    ]
    if not baseline_records:
        baseline_records = [
            (run_id, item)
            for run_id, item in metric_records
            if run_id == objective.baseline_run_id and not item.candidate_id.startswith("next_")
        ]
    baseline_values = [float(item.value) for _, item in baseline_records]
    baseline_value = mean(baseline_values) if baseline_values else None
    baseline_profile = _best_baseline_profile(baseline_records, node_profiles)
    baseline_trusted = baseline_profile in {"baseline_full", "baseline_confirm"}

    all_metric_records = [item for _, item in records]
    candidates: dict[str, list[tuple[str, MetricEvidence, PairedMetricDelta]]] = {}
    for run_id, item in metric_records:
        if item.evidence_role != "current_observation" or item.inheritance_depth > 0:
            continue
        _, delta = paired_metric_delta(item, all_metric_records)
        if delta is not None:
            candidates.setdefault(item.candidate_id, []).append((run_id, item, delta))
    ranked = sorted(
        candidates.items(),
        key=lambda pair: mean(delta.effect_delta for _, _, delta in pair[1]),
        reverse=True,
    )
    best_candidate_id = ranked[0][0] if ranked else None
    best_pairs = ranked[0][1] if ranked else []
    best_records = [(run_id, item) for run_id, item, _ in best_pairs]
    best_values = [delta.candidate_value for _, _, delta in best_pairs]
    paired_values = [delta.paired_delta for _, _, delta in best_pairs]
    best_value = mean(best_values) if best_values else None
    matched_baseline_values = [delta.baseline_value for _, _, delta in best_pairs]
    if matched_baseline_values:
        baseline_value = mean(matched_baseline_values)
    observed_delta = mean(paired_values) if paired_values else None
    required_delta = objective.required_delta(baseline_value)
    target_reached = bool(
        observed_delta is not None and required_delta is not None and observed_delta >= required_delta
    )
    candidate_seeds = {delta.match_key.seed for _, _, delta in best_pairs}
    ci_low, ci_high = _confidence_interval(paired_values, objective.confidence_level)
    delta_ci_low = ci_low
    delta_ci_high = ci_high
    confirmed = bool(
        target_reached
        and len(candidate_seeds) >= objective.confirmation_seeds
        and delta_ci_low is not None
        and required_delta is not None
        and delta_ci_low >= required_delta
    )
    baseline_latency, candidate_latency = _paired_guard_values(
        best_records, all_metric_records, "latency_ms"
    )
    baseline_size, candidate_size = _paired_guard_values(
        best_records, all_metric_records, "model_size_mb"
    )
    latency_regression = _regression_ratio(baseline_latency, candidate_latency)
    size_regression = _regression_ratio(baseline_size, candidate_size)
    guardrails_passed = bool(
        latency_regression is not None
        and size_regression is not None
        and latency_regression <= objective.max_latency_regression
        and size_regression <= objective.max_model_size_regression
    )
    gpu_hours = _gpu_hours(records)
    completed_rounds, no_improvement_rounds = _pilot_round_progress(metric_records, base_run_id)
    blockers: list[str] = []
    if not baseline_trusted:
        blockers.append("trusted_baseline_full_missing")
    if not best_pairs:
        blockers.append("matched_baseline_control_missing")
    if len(candidate_seeds) < objective.confirmation_seeds:
        blockers.append(f"confirmation_seeds:{len(candidate_seeds)}/{objective.confirmation_seeds}")
    if target_reached and not confirmed:
        blockers.append("confidence_interval_not_confirmed")
    if latency_regression is None:
        blockers.append("latency_comparison_missing")
    elif latency_regression > objective.max_latency_regression:
        blockers.append("latency_regression_exceeds_objective")
    if size_regression is None:
        blockers.append("model_size_comparison_missing")
    elif size_regression > objective.max_model_size_regression:
        blockers.append("model_size_regression_exceeds_objective")

    stop_reason = "continue_search"
    should_stop = False
    if confirmed and baseline_trusted and guardrails_passed:
        should_stop, stop_reason = True, "objective_confirmed"
    elif target_reached:
        should_stop = True
        stop_reason = (
            "target_reached_pending_full_confirmation"
            if guardrails_passed
            else "target_reached_pending_guard_evidence"
        )
    elif gpu_hours >= objective.max_gpu_hours:
        should_stop, stop_reason = True, "gpu_budget_exhausted"
    elif completed_rounds >= objective.max_pilot_rounds:
        should_stop, stop_reason = True, "max_pilot_rounds_reached"
    elif no_improvement_rounds >= objective.no_improvement_patience:
        should_stop, stop_reason = True, "no_improvement_patience_reached"

    return OptimizationObjectiveStatus(
        objective_hash=objective.objective_hash,
        primary_metric=objective.primary_metric,
        baseline_value=baseline_value,
        baseline_profile=baseline_profile,
        baseline_trusted=baseline_trusted,
        best_candidate_id=best_candidate_id,
        best_value=best_value,
        observed_delta=observed_delta,
        required_delta=required_delta,
        target_reached=target_reached,
        confirmed=confirmed,
        success=confirmed and baseline_trusted and guardrails_passed,
        candidate_seed_count=len(candidate_seeds),
        confidence_interval_low=delta_ci_low,
        confidence_interval_high=delta_ci_high,
        latency_regression=latency_regression,
        model_size_regression=size_regression,
        guardrails_passed=guardrails_passed,
        gpu_hours_used=round(gpu_hours, 6),
        gpu_budget_remaining=round(max(0.0, objective.max_gpu_hours - gpu_hours), 6),
        completed_pilot_rounds=completed_rounds,
        no_improvement_rounds=no_improvement_rounds,
        should_stop=should_stop,
        stop_reason=stop_reason,
        blockers=blockers,
    )


def _objective_run_dirs(root: Path, base_run_id: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir()
            and (path.name == base_run_id or path.name.startswith(f"{base_run_id}-r"))
        ],
        key=lambda path: (_round_index(path.name, base_run_id), path.name),
    )


def _round_index(run_id: str, base_run_id: str) -> int:
    match = re.fullmatch(re.escape(base_run_id) + r"-r(\d+)", run_id)
    return int(match.group(1)) if match else 0


def _plan_node_metadata(run_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    path = run_dir / "artifacts" / "experiment_plan.yaml"
    if not path.is_file():
        return {}, {}
    plan = ExperimentPlan.from_yaml(path)
    profiles: dict[str, str] = {}
    seeds: dict[str, int] = {}
    for node in plan.nodes:
        seeds[node.node_id] = node.seed
        if node.command_spec is not None:
            value = node.command_spec.metadata.get("training_budget_profile")
            if value:
                profiles[node.node_id] = str(value)
    return profiles, seeds


def _best_baseline_profile(
    records: list[tuple[str, MetricEvidence]],
    profiles: dict[tuple[str, str], str],
) -> str | None:
    rank = {"debug": 0, "pilot": 1, "baseline_full": 2, "baseline_confirm": 3}
    values = [profiles.get((run_id, item.node_id)) for run_id, item in records]
    filtered = [value for value in values if value]
    return max(filtered, key=lambda value: rank.get(value, -1)) if filtered else None


def _confidence_interval(values: list[float], confidence_level: float) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    z = 1.96 if confidence_level >= 0.95 else 1.645
    margin = z * stdev(values) / math.sqrt(len(values))
    center = mean(values)
    return center - margin, center + margin


def _gpu_hours(records: list[tuple[str, MetricEvidence]]) -> float:
    duration_by_node: dict[tuple[str, str], float] = {}
    for run_id, item in records:
        if item.metric_name != "execution_duration_seconds" or not isinstance(item.value, (int, float)):
            continue
        key = (run_id, item.node_id)
        duration_by_node[key] = max(duration_by_node.get(key, 0.0), float(item.value))
    return sum(duration_by_node.values()) / 3600.0


def _mean_metric_for_nodes(
    records: list[tuple[str, MetricEvidence]],
    node_keys: set[tuple[str, str]],
    metric_name: str,
) -> float | None:
    values = [
        float(item.value)
        for run_id, item in records
        if (run_id, item.node_id) in node_keys
        and item.metric_name == metric_name
        and isinstance(item.value, (int, float))
    ]
    return mean(values) if values else None


def _paired_guard_values(
    candidate_records: list[tuple[str, MetricEvidence]],
    all_records: list[MetricEvidence],
    metric_name: str,
) -> tuple[float | None, float | None]:
    node_ids = {(run_id, item.node_id, item.candidate_id) for run_id, item in candidate_records}
    deltas: list[PairedMetricDelta] = []
    for record in all_records:
        if record.metric_name != metric_name or record.evidence_role != "current_observation":
            continue
        if not any(record.run_id == run_id and record.node_id == node_id and record.candidate_id == candidate_id for run_id, node_id, candidate_id in node_ids):
            continue
        _, delta = paired_metric_delta(record, all_records)
        if delta is not None:
            deltas.append(delta)
    if not deltas:
        return None, None
    return mean(item.baseline_value for item in deltas), mean(item.candidate_value for item in deltas)


def _regression_ratio(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline <= 0:
        return None
    return candidate / baseline - 1.0


def _pilot_round_progress(
    records: list[tuple[str, MetricEvidence]],
    base_run_id: str,
) -> tuple[int, int]:
    round_best: dict[int, float] = {}
    for run_id, item in records:
        index = _round_index(run_id, base_run_id)
        if index <= 0:
            continue
        round_best[index] = max(round_best.get(index, float("-inf")), float(item.value))
    best = float("-inf")
    trailing = 0
    for index in sorted(round_best):
        value = round_best[index]
        if value > best:
            best = value
            trailing = 0
        else:
            trailing += 1
    return len(round_best), trailing
