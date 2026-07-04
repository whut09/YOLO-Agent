"""Fast baseline promotion gate for Ultralytics training runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import Evidence, ExperimentNode, MetricValue


FastBaselineStage = Literal["sanity", "pilot", "full_baseline", "confirmation"]


class FastBaselineGateConfig(BaseModel):
    """Promotion policy for baseline training budgets."""

    enabled: bool = True
    profile_to_stage: dict[str, FastBaselineStage] = Field(
        default_factory=lambda: {
            "debug": "sanity",
            "pilot": "pilot",
            "baseline_full": "full_baseline",
            "baseline_confirm": "confirmation",
        }
    )
    minimum_confirmation_seeds: int = Field(default=3, ge=1)


class FastBaselineGateResult(BaseModel):
    """Decision from the fast baseline promotion gate."""

    ok: bool
    profile: str
    stage: FastBaselineStage | None = None
    blocked_by: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confirmed_seed_count: int = 0
    message: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FastBaselineGate:
    """Enforce sanity -> pilot -> full baseline -> seed confirmation."""

    def __init__(self, config: FastBaselineGateConfig | None = None) -> None:
        self.config = config or FastBaselineGateConfig()

    def evaluate(
        self,
        profile: str,
        evidence: Evidence | None = None,
        candidate_id: str | None = None,
    ) -> FastBaselineGateResult:
        """Return whether the requested training profile may run."""
        stage = self.config.profile_to_stage.get(profile)
        if not self.config.enabled or stage is None:
            return FastBaselineGateResult(ok=True, profile=profile, stage=stage, message="Gate not applicable.")
        if stage == "sanity":
            return FastBaselineGateResult(ok=True, profile=profile, stage=stage, message="Sanity stage is always allowed.")
        if evidence is None:
            return _blocked(profile, stage, ["missing_run_evidence"])
        if stage == "pilot":
            return _require_stage(profile, stage, evidence, "sanity", candidate_id)
        if stage == "full_baseline":
            return _require_stage(profile, stage, evidence, "pilot", candidate_id)
        if stage == "confirmation":
            full = _require_stage(profile, stage, evidence, "full_baseline", candidate_id)
            if not full.ok:
                return full
            seed_count = _confirmed_seed_count(evidence, candidate_id)
            warnings = []
            if seed_count < self.config.minimum_confirmation_seeds:
                warnings.append(
                    f"Confirmation requires {self.config.minimum_confirmation_seeds} seeds; current trusted seeds={seed_count}."
                )
            return FastBaselineGateResult(
                ok=True,
                profile=profile,
                stage=stage,
                warnings=warnings,
                confirmed_seed_count=seed_count,
                message="Confirmation stage is allowed after full baseline evidence.",
            )
        return FastBaselineGateResult(ok=True, profile=profile, stage=stage)

    def stage_metrics(self, profile: str, node: ExperimentNode, success: bool) -> dict[str, MetricValue]:
        """Return metrics that record completion of a gated stage."""
        stage = self.config.profile_to_stage.get(profile)
        if stage is None:
            return {}
        metrics: dict[str, MetricValue] = {
            "training_budget_profile": profile,
            "fast_baseline_stage": stage,
            "fast_baseline_seed": node.seed,
        }
        if stage == "sanity":
            metrics["fast_baseline_sanity_passed"] = success
        elif stage == "pilot":
            metrics["fast_baseline_pilot_passed"] = success
        elif stage == "full_baseline":
            metrics["fast_baseline_full_baseline_passed"] = success
        elif stage == "confirmation":
            metrics["fast_baseline_confirmation_seed_passed"] = success
        return metrics

    def persist_decision(
        self,
        store: EvidenceStore,
        run_id: str,
        node: ExperimentNode,
        result: FastBaselineGateResult,
    ) -> Path:
        """Persist a gate decision artifact and metrics."""
        artifact_path = store.create_run(run_id) / "artifacts" / f"{node.node_id}_fast_baseline_gate.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        store.log_artifact_manifest(
            run_id=run_id,
            name=f"{node.node_id}_fast_baseline_gate",
            artifact_path=artifact_path,
            producer_stage="fast_baseline_gate",
        )
        store.log_candidate_metrics(
            run_id=run_id,
            candidate_id=node.candidate_config.candidate_id,
            node_id=node.node_id,
            metrics={
                "fast_baseline_gate_ok": result.ok,
                "fast_baseline_gate_profile": result.profile,
                "fast_baseline_gate_stage": result.stage,
                "fast_baseline_confirmed_seed_count": result.confirmed_seed_count,
            },
            dataset_version=node.data_version,
            split="runtime",
            source="fast_baseline_gate",
            verified=True,
            validator="fast_baseline_gate",
            source_artifact=artifact_path,
        )
        return artifact_path


def _require_stage(
    profile: str,
    stage: FastBaselineStage,
    evidence: Evidence,
    required_stage: FastBaselineStage,
    candidate_id: str | None,
) -> FastBaselineGateResult:
    metric_name = _pass_metric(required_stage)
    if _has_passed_metric(evidence, metric_name, candidate_id):
        return FastBaselineGateResult(
            ok=True,
            profile=profile,
            stage=stage,
            message=f"{stage} allowed after {required_stage} evidence.",
        )
    return _blocked(profile, stage, [metric_name])


def _blocked(profile: str, stage: FastBaselineStage, blocked_by: list[str]) -> FastBaselineGateResult:
    return FastBaselineGateResult(
        ok=False,
        profile=profile,
        stage=stage,
        blocked_by=blocked_by,
        message=(
            "Fast Baseline Gate blocked this run. Required flow is "
            "1 epoch sanity -> 10 epoch pilot -> full baseline -> 3 seed confirmation."
        ),
    )


def _pass_metric(stage: FastBaselineStage) -> str:
    return {
        "sanity": "fast_baseline_sanity_passed",
        "pilot": "fast_baseline_pilot_passed",
        "full_baseline": "fast_baseline_full_baseline_passed",
        "confirmation": "fast_baseline_confirmation_seed_passed",
    }[stage]


def _has_passed_metric(evidence: Evidence, metric_name: str, candidate_id: str | None) -> bool:
    for record in evidence.metric_records:
        if record.metric_name != metric_name or record.value is not True or not record.verified:
            continue
        if candidate_id is not None and record.candidate_id != candidate_id:
            continue
        return True
    return False


def _confirmed_seed_count(evidence: Evidence, candidate_id: str | None) -> int:
    seeds: set[str] = set()
    passed_nodes = {
        record.node_id
        for record in evidence.metric_records
        if record.metric_name == "fast_baseline_confirmation_seed_passed"
        and record.value is True
        and record.verified
        and (candidate_id is None or record.candidate_id == candidate_id)
    }
    for record in evidence.metric_records:
        if record.node_id not in passed_nodes:
            continue
        if record.metric_name == "fast_baseline_seed" and record.value is not None:
            seeds.add(str(record.value))
    return len(seeds)
