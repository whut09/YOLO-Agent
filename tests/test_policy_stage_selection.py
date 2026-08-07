"""Candidate-source selection tests for the unified decision stage."""

from __future__ import annotations

from yolo_agent.agents.policy_stage_runner import (
    _apply_inherited_pilot_contract,
    _select_candidate_policies,
)
from yolo_agent.agents.strategy_policy import CandidatePolicy
from yolo_agent.core.run_context import RunContext


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


def _pilot_context() -> RunContext:
    return RunContext(
        run_id="paper-policy",
        task_path="task.yaml",
        data_yaml="data.yaml",
        metadata={
            "inherited_proposal_mode": "pilot_only",
            "inherited_current_round_focus": [
                {
                    "fact_type": "class_low_ap",
                    "subject": "person",
                    "metric_name": "per_class_ap",
                    "action_candidates": ["class_balanced_sampling"],
                }
            ],
            "inherited_current_round_error_actions": ["class_balanced_sampling"],
            "inherited_tried_action_ids": [],
        },
    )


def test_materialized_recipe_survives_pilot_action_binding() -> None:
    """Recipe ids must not be confused with the generic diagnosis vocabulary."""
    policy = CandidatePolicy(
        policy_id="paper_recipe_yolo26_small_object_sampling_v1_0_0",
        action_id="yolo26_small_object_sampling",
        components=["sampling.small_object"],
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        train_overrides={"imgsz": 640, "profile": "pilot"},
        target_error_facts=[
            {"fact_type": "area_metric", "area": "small", "metric_name": "ap_small"}
        ],
        expected_improvement={"summary": "Improve AP_small exposure."},
    )

    selected, guardrails = _apply_inherited_pilot_contract(_pilot_context(), [policy])

    assert [item.policy_id for item in selected] == [policy.policy_id]
    assert selected[0].train_overrides["imgsz"] == 640
    assert "yolo26_small_object_sampling" in selected[0].train_overrides["target_actions"]
    assert "target_error_facts_required" in guardrails


def test_materialized_recipe_without_error_facts_is_rejected() -> None:
    """Materialization never bypasses the evidence-first training contract."""
    policy = CandidatePolicy(
        policy_id="paper_recipe_yolo26_small_object_sampling_v1_0_0",
        action_id="yolo26_small_object_sampling",
        components=["sampling.small_object"],
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
    )

    selected, _ = _apply_inherited_pilot_contract(_pilot_context(), [policy])

    assert selected == []
