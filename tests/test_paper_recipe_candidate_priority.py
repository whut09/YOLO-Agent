from __future__ import annotations

from yolo_agent.agents.paper_recipe_materialization.candidate_priority import (
    rank_materialized_candidate,
)
from yolo_agent.agents.paper_recipe_materialization.schemas import (
    PaperCandidatePlanningContext,
)
from yolo_agent.core.policy_memory import ActionFingerprint, PolicyMemoryRecord
from tests.paper_materialization_fixtures import candidate_input, error_fact


def test_candidate_priority_rewards_reusable_paper_coverage() -> None:
    broad = candidate_input()
    broad.planning_context = PaperCandidatePlanningContext(
        mechanism_cluster_id="sampling_class_balancing",
        covered_paper_ids=[f"paper-{index}" for index in range(8)],
        canonical_mechanism_confidence=0.9,
        runtime_hook_available=True,
        implementation_cost="low",
        expected_gpu_cost="low",
        expected_latency_cost="low",
        expected_model_size_cost="low",
    )
    narrow = candidate_input(prior_id="narrow-prior", candidate_id="narrow")
    narrow.planning_context = PaperCandidatePlanningContext(
        covered_paper_ids=["paper-only"],
        canonical_mechanism_confidence=0.9,
        runtime_hook_available=True,
        implementation_cost="low",
        expected_gpu_cost="low",
        expected_latency_cost="low",
        expected_model_size_cost="low",
    )

    broad_priority = rank_materialized_candidate(
        broad,
        current_error_facts=[error_fact()],
        local_evidence=[],
        runtime_execution_ready=True,
    )
    narrow_priority = rank_materialized_candidate(
        narrow,
        current_error_facts=[error_fact()],
        local_evidence=[],
        runtime_execution_ready=True,
    )

    assert broad_priority.covered_paper_count == 9
    assert broad_priority.score > narrow_priority.score
    assert broad_priority.breakdown["paper_coverage"] > narrow_priority.breakdown[
        "paper_coverage"
    ]


def test_local_negative_posterior_reduces_paper_candidate_priority() -> None:
    item = candidate_input()
    item.planning_context = PaperCandidatePlanningContext(
        canonical_mechanism_confidence=1.0,
        runtime_hook_available=True,
    )
    negative = PolicyMemoryRecord(
        run_id="negative-run",
        dataset_version="unversioned",
        action="small object sampling",
        action_fingerprint=ActionFingerprint(
            action="small object sampling",
            component_ids=["dummy.component"],
            dataset_signature="unversioned",
        ),
        target="ap_small",
        effect_delta=-0.02,
        confidence="high",
        seed_count=3,
        evidence_status="confirmed",
    )

    priority = rank_materialized_candidate(
        item,
        current_error_facts=[error_fact()],
        local_evidence=[negative],
        runtime_execution_ready=True,
    )

    assert priority.breakdown["local_posterior"] == -40.0
    assert "local_negative_posterior_has_priority_over_paper_prior" in priority.reasons


def test_candidate_fingerprint_deduplicates_paper_sources_for_same_mechanism() -> None:
    first = candidate_input(prior_id="first", candidate_id="first")
    second = candidate_input(prior_id="second", candidate_id="second")
    second.prior = second.prior.model_copy(update={"paper_ids": ["another-paper"]})

    first_priority = rank_materialized_candidate(
        first,
        current_error_facts=[error_fact()],
        local_evidence=[],
        runtime_execution_ready=True,
    )
    second_priority = rank_materialized_candidate(
        second,
        current_error_facts=[error_fact()],
        local_evidence=[],
        runtime_execution_ready=True,
    )

    assert first_priority.candidate_fingerprint == second_priority.candidate_fingerprint
