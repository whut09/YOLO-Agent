"""Learn local posteriors for paper-derived recipes without promoting paper claims."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.core.policy_memory import (
    ActionFingerprint,
    PolicyMemoryRecord,
    PolicyMemoryStore,
)


class PaperRecipeOutcome(BaseModel):
    run_id: str
    recipe_id: str
    recipe_version: str = "unknown"
    paper_ids: list[str]
    component_ids: list[str]
    component_versions: dict[str, str] = Field(default_factory=dict)
    changed_variable: str
    before_value: Any = None
    after_value: Any = None
    detector_family: str = "yolo26"
    model_family: str = "yolo26"
    dataset_version: str
    dataset_signature: str
    protocol_hash: str
    snapshot_hash: str
    fidelity: str
    seed: int | str
    metric_name: str = "map50_95"
    paper_prior_effect: dict[str, Any] = Field(default_factory=dict)
    pilot_3_delta: float | None = None
    pilot_10_delta: float | None = None
    full_delta: float | None = None
    target_error_fact_delta: dict[str, float] = Field(default_factory=dict)
    latency_delta: float | None = None
    model_size_delta: float | None = None
    paired_bootstrap_ci: tuple[float, float] | None = None
    cross_seed_ci: tuple[float, float] | None = None
    seed_count: int = Field(default=1, ge=1)
    implementation_cost: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None
    candidate_id: str | None = None
    node_id: str | None = None

    @property
    def observed_delta(self) -> float | None:
        if self.fidelity in {"candidate_full", "full"}:
            return self.full_delta
        if self.fidelity == "pilot_10":
            return self.pilot_10_delta
        return self.pilot_3_delta


class PaperOutcomeLearningResult(BaseModel):
    appended: bool
    record: PolicyMemoryRecord
    paper_prior_effect: dict[str, Any]
    local_posterior_status: str
    duplicate: bool = False


class PaperOutcomeLearner:
    """Convert one paper-recipe experiment summary into idempotent policy memory."""

    def __init__(self, memory_store: PolicyMemoryStore | None = None) -> None:
        self.memory_store = memory_store or PolicyMemoryStore()

    def learn(self, outcome: PaperRecipeOutcome) -> PaperOutcomeLearningResult:
        fingerprint = ActionFingerprint(
            action=outcome.recipe_id,
            recipe_id=outcome.recipe_id,
            recipe_version=outcome.recipe_version,
            paper_ids=sorted(set(outcome.paper_ids)),
            component_ids=sorted(set(outcome.component_ids)),
            component_versions=dict(outcome.component_versions),
            changed_variable=outcome.changed_variable,
            before_value=outcome.before_value,
            after_value=outcome.after_value,
            detector_family=outcome.detector_family,
            model_family=outcome.model_family,
            dataset_signature=outcome.dataset_signature,
            protocol_hash=outcome.protocol_hash,
            snapshot_hash=outcome.snapshot_hash,
            fidelity=outcome.fidelity,
            seed=outcome.seed,
        )
        status, confidence, reason = _evidence_status(outcome)
        delta = outcome.observed_delta
        correlation = _pilot_full_correlation([
            *self._matching_records(outcome),
            _outcome_pair(outcome),
        ])
        record = PolicyMemoryRecord(
            run_id=outcome.run_id,
            dataset_version=outcome.dataset_version,
            action=outcome.recipe_id,
            action_fingerprint=fingerprint,
            target=outcome.metric_name,
            metric_name=outcome.metric_name,
            delta=delta,
            effect_delta=delta,
            trend="improved" if delta is not None and delta > 0 else ("regressed" if delta is not None and delta < 0 else "unchanged"),
            candidate_id=outcome.candidate_id,
            node_id=outcome.node_id,
            confidence=confidence,
            confidence_reason=reason,
            seed_count=outcome.seed_count,
            changed_variables={outcome.changed_variable: outcome.after_value},
            source="paper_recipe_local_outcome",
            paper_prior_effect=dict(outcome.paper_prior_effect),
            pilot_3_delta=outcome.pilot_3_delta,
            pilot_10_delta=outcome.pilot_10_delta,
            full_delta=outcome.full_delta,
            target_error_fact_delta=dict(outcome.target_error_fact_delta),
            latency_delta=outcome.latency_delta,
            model_size_delta=outcome.model_size_delta,
            paired_bootstrap_ci=outcome.paired_bootstrap_ci,
            cross_seed_ci=outcome.cross_seed_ci,
            pilot_full_correlation=correlation,
            implementation_cost=dict(outcome.implementation_cost),
            failure_reason=outcome.failure_reason,
            evidence_status=status,
        )
        appended = self.memory_store.append([record])
        return PaperOutcomeLearningResult(
            appended=bool(appended),
            record=record,
            paper_prior_effect=dict(outcome.paper_prior_effect),
            local_posterior_status=status,
            duplicate=not bool(appended),
        )

    def _matching_records(self, outcome: PaperRecipeOutcome) -> list[tuple[float, float]]:
        pairs = []
        for record in self.memory_store.query(action=outcome.recipe_id, dataset_version=outcome.dataset_version):
            fingerprint = record.action_fingerprint
            if fingerprint is None:
                continue
            if fingerprint.dataset_signature != outcome.dataset_signature:
                continue
            if fingerprint.protocol_hash != outcome.protocol_hash or fingerprint.snapshot_hash != outcome.snapshot_hash:
                continue
            if record.pilot_10_delta is not None and record.full_delta is not None:
                pairs.append((record.pilot_10_delta, record.full_delta))
        return pairs


def _outcome_pair(outcome: PaperRecipeOutcome) -> tuple[float, float] | None:
    if outcome.pilot_10_delta is None or outcome.full_delta is None:
        return None
    return outcome.pilot_10_delta, outcome.full_delta


def _pilot_full_correlation(pairs: list[tuple[float, float] | None]) -> float | None:
    values = [item for item in pairs if item is not None]
    if len(values) < 2:
        return None
    xs, ys = zip(*values)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in values)
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return round(numerator / denominator, 6) if denominator else None


def _evidence_status(outcome: PaperRecipeOutcome) -> tuple[str, str, str]:
    if outcome.failure_reason:
        return "failed", "low", f"local failure: {outcome.failure_reason}"
    paired_confirmed = (
        outcome.seed_count >= 3
        and outcome.paired_bootstrap_ci is not None
        and outcome.cross_seed_ci is not None
        and outcome.cross_seed_ci[0] > 0
    )
    if paired_confirmed:
        return "confirmed", "high", "multi-seed local outcome with paired and cross-seed confidence intervals"
    return "possible", "low", "single-seed or incomplete confidence intervals; paper claim remains prior only"


__all__ = ["PaperOutcomeLearner", "PaperOutcomeLearningResult", "PaperRecipeOutcome"]
