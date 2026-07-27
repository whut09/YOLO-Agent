"""Candidate-source selection tests for the unified decision stage."""

from __future__ import annotations

from yolo_agent.agents.policy_stage_runner import _select_candidate_policies
from yolo_agent.agents.strategy_policy import CandidatePolicy


def _policy(policy_id: str) -> CandidatePolicy:
    return CandidatePolicy(
        policy_id=policy_id,
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
    )


def test_empty_llm_training_selection_uses_guarded_deterministic_recipe() -> None:
    """An explanatory LLM response must not silently terminate a sound pilot plan."""
    fallback = _policy("next_training_optimizer_adamw")

    selected, mode, warning = _select_candidate_policies(
        llm_status="used",
        accepted_llm_policies=[],
        fallback_policies=[fallback],
        missing_diagnostic_evidence=[],
    )

    assert selected == [fallback]
    assert mode == "deterministic_fallback"
    assert warning == "llm_returned_no_training_candidate_using_deterministic_recipes"


def test_real_missing_evidence_still_blocks_fallback_training() -> None:
    """Fallback recipes must not bypass the evidence-first contract."""
    selected, mode, warning = _select_candidate_policies(
        llm_status="used",
        accepted_llm_policies=[],
        fallback_policies=[_policy("next_training_optimizer_adamw")],
        missing_diagnostic_evidence=["coco_error_facts"],
    )

    assert selected == []
    assert mode == "llm"
    assert warning is None
