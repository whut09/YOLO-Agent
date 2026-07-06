"""LLM decision advisor tests."""

from __future__ import annotations

import json

from yolo_agent.agents.error_driven_loop import ErrorDrivenLoopReport, NextRoundPlan
from yolo_agent.agents.error_to_action import ErrorActionPlan
from yolo_agent.agents.llm_decision_advisor import LLMDecisionAdvisor, LLMProposalBundle
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
    assert result.prompt_sha256
    assert result.input_summary["task"]["task_type"] == "detect"
    assert result.temperature == 0.1


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


def test_llm_advisor_prompt_enforces_evidence_only_mode(monkeypatch) -> None:
    """Prompt payload should tell the LLM to emit evidence actions only when diagnostic facts are missing."""
    monkeypatch.setenv("YOLO_AGENT_TEST_OPENAI_KEY", "test-key")
    captured_payload = {}

    def fake_transport(config, messages):
        nonlocal captured_payload
        captured_payload = json.loads(messages[1]["content"])
        return json.dumps(
            {
                "candidate_policies": [
                    {
                        "policy_id": "import_ap_small",
                        "action_domain": "evidence",
                        "action_id": "import_coco_eval",
                        "execution_action": "import_metrics",
                        "base_model": "yolo26n.pt",
                        "scale": "n",
                        "framework": "ultralytics",
                        "train_overrides": {"evidence_action": "import_metrics"},
                        "evidence_required": ["coco_eval"],
                        "expected_effect": ["Collect AP_small before training proposals."],
                        "risk": "low",
                    }
                ]
            }
        )

    result = LLMDecisionAdvisor(config=_config(), transport=fake_transport).propose(
        task_spec=_task(),
        diagnosis_report=_diagnosis_report(),
        inherited_context={
            "missing_diagnostic_evidence": ["ap_small", "per_class_ap"],
            "llm_evidence_only_mode": True,
        },
    )

    assert result.status == "used"
    assert captured_payload["diagnostic_evidence_gate"]["llm_evidence_only_mode"] is True
    assert captured_payload["diagnostic_evidence_gate"]["missing_diagnostic_evidence"] == ["ap_small", "per_class_ap"]
    assert any("output evidence actions only" in rule for rule in captured_payload["hard_rules"])
    assert result.input_summary["inherited_context"]["missing_diagnostic_evidence"] == ["ap_small", "per_class_ap"]


def test_llm_advisor_parses_full_proposal_bundle(monkeypatch) -> None:
    """LLM output should preserve doctor drafts, evidence requests, and rejected actions."""
    monkeypatch.setenv("YOLO_AGENT_TEST_OPENAI_KEY", "test-key")

    def fake_transport(config, messages):
        return json.dumps(
            {
                "schema_version": "llm_proposal_bundle.v1",
                "doctor_report_draft": {
                    "primary_problem": "AP_small low",
                    "likely_causes": ["small objects below effective stride"],
                    "evidence": ["AP_small=0.21"],
                    "rejected_actions": [
                        {
                            "action": "increase_imgsz",
                            "reason": "imgsz is fixed for fair baseline comparison",
                            "blocked_by": ["fixed_imgsz"],
                        }
                    ],
                    "selected_actions": ["small_object_oversampling"],
                    "why": ["Targets the observed AP_small failure mode."],
                    "expected_improvement": {"ap_small": "+0.5 to +1.5"},
                    "stop_condition": ["Pilot does not improve AP_small."],
                    "missing_evidence": ["per_class_ap_small"],
                    "confidence": "low",
                },
                "evidence_requests": [
                    {
                        "evidence_id": "per_class_ap_small",
                        "reason": "Need class-specific small-object failures before full candidates.",
                        "evidence_type": "error_fact",
                        "target": "coco_val2017",
                        "required_before": "full",
                        "priority": "high",
                    }
                ],
                "rejected_actions": [
                    {
                        "action": "candidate_full",
                        "reason": "Full candidates require pilot evidence first.",
                        "blocked_by": ["pilot_only_proposal_mode"],
                    }
                ],
                "candidate_policies": [
                    {
                        "policy_id": "mine_small_errors",
                        "action_domain": "evidence",
                        "action_id": "mine_coco_small_errors",
                        "execution_action": "mine_errors",
                        "base_model": "yolo26n.pt",
                        "scale": "n",
                        "framework": "ultralytics",
                        "evidence_required": ["coco_eval"],
                        "target_error_facts": [{"metric_name": "ap_small"}],
                        "expected_effect": ["Collect AP_small facts before training changes."],
                        "risk": "low",
                    }
                ],
            }
        )

    result = LLMDecisionAdvisor(config=_config(), transport=fake_transport).propose(
        task_spec=_task(),
        diagnosis_report=_diagnosis_report(),
    )

    assert result.status == "used"
    assert isinstance(result.proposal_bundle, LLMProposalBundle)
    assert result.doctor_report_draft is not None
    assert result.doctor_report_draft.primary_problem == "AP_small low"
    assert result.evidence_requests[0].evidence_id == "per_class_ap_small"
    assert result.rejected_actions[0].action == "candidate_full"
    assert result.proposals[0].policy_id == "llm_mine_small_errors"
    assert result.proposals[0].execution_action == "mine_errors"


def test_llm_advisor_rejects_invalid_auxiliary_schema_without_losing_valid_policy(monkeypatch) -> None:
    """Invalid doctor/evidence fields should be warned about instead of silently trusted."""
    monkeypatch.setenv("YOLO_AGENT_TEST_OPENAI_KEY", "test-key")

    def fake_transport(config, messages):
        return json.dumps(
            {
                "unexpected": "ignored with warning",
                "doctor_report_draft": {
                    "primary_problem": "AP_small low",
                    "confidence": "certain",
                },
                "evidence_requests": [
                    {
                        "evidence_id": "runtime_profile",
                        "reason": "Need runtime facts.",
                        "extra_field": "not allowed",
                    }
                ],
                "candidate_policies": [
                    {
                        "policy_id": "profile_runtime",
                        "action_domain": "evidence",
                        "action_id": "profile_runtime",
                        "execution_action": "benchmark_latency",
                        "base_model": "yolo26n.pt",
                        "scale": "n",
                        "framework": "ultralytics",
                        "expected_effect": ["Collect latency evidence."],
                        "risk": "low",
                    }
                ],
            }
        )

    result = LLMDecisionAdvisor(config=_config(), transport=fake_transport).propose(
        task_spec=_task(),
        diagnosis_report=_diagnosis_report(),
    )

    assert result.status == "used"
    assert result.proposals[0].policy_id == "llm_profile_runtime"
    assert result.doctor_report_draft is None
    assert result.evidence_requests == []
    assert any("llm_unknown_top_level_keys:unexpected" in warning for warning in result.warnings)
    assert any("llm_doctor_report_draft_invalid" in warning for warning in result.warnings)
    assert any("llm_evidence_request_0_invalid" in warning for warning in result.warnings)
