"""PolicyMemory update for a verified paper-driven pilot."""

from __future__ import annotations

import json
from pathlib import Path

from yolo_agent.agents.paper_outcome_learner import (
    PaperOutcomeLearner,
    PaperOutcomeLearningResult,
    PaperRecipeOutcome,
)
from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchContext,
)
from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperProtocolIdentity,
)
from yolo_agent.core.paired_experiment import PairedExperimentResult
from yolo_agent.core.policy_memory import PolicyMemoryStore


def record_sampling_pilot_outcome(
    *,
    memory_root: Path | str,
    run_id: str,
    research: PaperAcceptanceResearchContext,
    protocol: PaperProtocolIdentity,
    pilot_3: PairedExperimentResult,
    pilot_10: PairedExperimentResult | None,
    output_path: Path | str,
    failure_reason: str | None = None,
) -> PaperOutcomeLearningResult:
    """Append one local posterior; paper claims remain prior-only metadata."""
    primary_3 = pilot_3.metric_deltas["ap_small"]
    primary_10 = pilot_10.metric_deltas["ap_small"] if pilot_10 is not None else None
    selected = pilot_10 or pilot_3
    bootstrap = selected.paired_bootstrap_ci
    outcome = PaperRecipeOutcome(
        run_id=run_id,
        recipe_id="sampling.small_object",
        recipe_version="1",
        paper_ids=research.paper_ids,
        component_ids=[research.component_id],
        component_versions={research.component_id: research.adapter_hash},
        changed_variable="data.sampling_policy",
        before_value="uniform",
        after_value="small_object_weighted",
        detector_family="yolo26",
        model_family="yolo26n",
        dataset_version=protocol.dataset_manifest_hash,
        dataset_signature=protocol.dataset_manifest_hash,
        protocol_hash=protocol.protocol_hash,
        snapshot_hash=research.snapshot_hash,
        fidelity="pilot_10" if pilot_10 is not None else "pilot_3",
        seed=protocol.seed,
        metric_name="ap_small",
        paper_prior_effect={
            "evidence_level": "paper_prior",
            "reported_delta": None,
            "local_evidence": False,
        },
        pilot_3_delta=primary_3.paired_delta,
        pilot_10_delta=primary_10.paired_delta if primary_10 is not None else None,
        target_error_fact_delta={
            f"{item.fact_type}/{item.subject}": item.effect_delta
            for item in selected.target_error_fact_deltas
        },
        latency_delta=(
            selected.latency_delta.paired_delta
            if selected.latency_delta is not None
            else None
        ),
        model_size_delta=(
            selected.model_size_delta.paired_delta
            if selected.model_size_delta is not None
            else None
        ),
        paired_bootstrap_ci=(
            (bootstrap.confidence_interval_low, bootstrap.confidence_interval_high)
            if bootstrap is not None
            else None
        ),
        seed_count=1,
        implementation_cost={"adapter_maturity": research.maturity},
        failure_reason=failure_reason,
        candidate_id=selected.candidate_id,
        node_id=selected.candidate_node_id,
    )
    result = PaperOutcomeLearner(PolicyMemoryStore(memory_root)).learn(outcome)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result


__all__ = ["record_sampling_pilot_outcome"]
