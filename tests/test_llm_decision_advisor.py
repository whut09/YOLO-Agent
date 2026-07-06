"""LLM decision advisor tests."""

from __future__ import annotations

import json

from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopReport, NextRoundPlan
from yolo_agent.agents.error_to_action import ErrorActionPlan
from yolo_agent.agents.llm_decision_advisor import LLMDecisionAdvisor
from yolo_agent.agents.optimization_recipe import OptimizationRecipePlan, RecipeComponents
from yolo_agent.agents.augmentation_policy import AugmentationPolicyAction, AugmentationPolicyResult
from yolo_agent.agents.sampling_policy import SamplingPolicyPlan
from yolo_agent.components.postprocess import PostProcessRecommendation
from yolo_agent.core.llm_config import LLMDecisionConfig
from yolo_agent.core.task_spec import MetricPriority, TaskSpec


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="detect",
        scene="generic",
        class_names=["object"],
        primary_metric=MetricPriority(name="map50_95"),
    )


def _diagnosis_report() -> ErrorDrivenLoopReport:
    return ErrorDrivenLoopReport(
        task_scene="generic",
        diagnostics=[],
        action_policy=ErrorActionPlan(observations=[], recommendations=[]),
        optimization_recipes=OptimizationRecipePlan(task_scene="generic", component_candidates=RecipeComponents()),
        sampling_policy=SamplingPolicyPlan(actions=[]),
        augmentation_policy=AugmentationPolicyResult(actions=AugmentationPolicyAction()),
        postprocess_policy=PostProcessRecommendation(scenario="generic", recommended_postprocess=[]),
        next_round=NextRoundPlan(),
    )


def _config() -> LLMDecisionConfig:
    return LLMDecisionConfig(
        enabled=True,
        provider="openai",
        model="gpt-5.5",
        model_alias="codex-gpt5.5",
        api_key_env="YOLO_AGENT_TEST_OPENAI_KEY",
        decision_role="proposal_generator_only",
        use_by_default=True,
        require_api_key=True,
    )


def test_llm_advisor_skips_when_default_model_has_no_api_key(monkeypatch) -> None:
    """Default LLM use should not break the harness when credentials are absent."""
    monkeypatch.delenv("YOLO_AGENT_TEST_OPENAI_KEY", raising=False)

    result = LLMDecisionAdvisor(config=_config()).propose(
        task_spec=_task(),
        diagnosis_report=_diagnosis_report(),
    )

    assert result.status == "skipped"
    assert result.proposals == []
    assert result.warnings == ["missing_api_key_env:YOLO_AGENT_TEST_OPENAI_KEY"]


def test_llm_advisor_parses_candidate_policies_from_transport(monkeypatch) -> None:
    """LLM JSON output should become guarded CandidatePolicy proposals."""
    monkeypatch.setenv("YOLO_AGENT_TEST_OPENAI_KEY", "test-key")

    def fake_transport(config, messages):
        assert config.model == "gpt-5.5"
        assert messages[0]["role"] == "system"
        return json.dumps(
            {
                "candidate_policies": [
                    {
                        "policy_id": "small_object_sampling",
                        "action_domain": "data",
                        "action_id": "small_object_oversampling",
                        "execution_action": "run_training",
                        "base_model": "yolo26n.pt",
                        "scale": "n",
                        "framework": "ultralytics",
                        "train_overrides": {"data_action": "small_object_oversampling"},
                        "target_error_facts": [{"metric_name": "ap_small"}],
                        "expected_improvement": {
                            "metric_name": "ap_small",
                            "minimum_expected_delta": "pilot_positive_delta",
                        },
                        "expected_effect": ["Target AP_small without changing imgsz."],
                        "risk": "low",
                    }
                ]
            }
        )

    result = LLMDecisionAdvisor(config=_config(), transport=fake_transport).propose(
        task_spec=_task(),
        diagnosis_report=_diagnosis_report(),
    )

    assert result.status == "used"
    assert result.proposals[0].policy_id == "llm_small_object_sampling"
    assert result.proposals[0].source == "llm"
    assert result.proposals[0].action_domain == "data"
