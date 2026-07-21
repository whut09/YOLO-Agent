from __future__ import annotations

import json

from yolo_agent.agents.llm_decision_advisor import LLMDecisionAdvisor
from tests.test_llm_decision_advisor import _config, _diagnosis_report, _task


def _context(*, maturity: str = "smoke_passed", missing: list[str] | None = None) -> dict:
    return {
        "run_id": "run-1",
        "paper_candidates": [{
            "prior_id": "prior-1",
            "recipe_id": "recipe-1",
            "paper_ids": ["paper-1"],
            "component_ids": ["component-1"],
        }],
        "deterministic_recipe_candidates": [{"policy_id": "recipe-1"}],
        "recipe_critic_results": [{"recipe_id": "recipe-1", "accepted": True}],
        "component_maturity": {"component-1": maturity},
        "baseline_evidence": [{"record_id": "baseline-evidence"}],
        "current_evidence": [{"record_id": "current-evidence"}],
        "error_facts": [{"fact_id": "fact-1"}],
        "missing_evidence": missing or [],
        "fixed_constraints": {"imgsz": 640},
    }


def _policy(**updates) -> dict:
    data = {
        "policy_id": "paper_recipe",
        "action_domain": "data",
        "action_id": "small_object_sampling",
        "execution_action": "run_training",
        "base_model": "yolo26n.pt",
        "scale": "n",
        "framework": "ultralytics",
        "components": ["component-1"],
        "train_overrides": {"imgsz": 640},
        "target_error_facts": [{"fact_id": "fact-1"}],
        "expected_improvement": {"map50_95": 0.01},
        "expected_effect": ["Improve the supplied small-object error fact."],
        "risk": "low",
    }
    data.update(updates)
    return data


def _run(payload: dict, context: dict):
    calls = []

    def transport(config, messages):
        calls.append(messages)
        return json.dumps(payload)

    result = LLMDecisionAdvisor(config=_config(), transport=transport).propose(
        task_spec=_task(),
        diagnosis_report=_diagnosis_report(),
        inherited_context={"decision_context": context},
    )
    return result, calls


def test_one_call_unifies_paper_rule_and_policy_decision(monkeypatch) -> None:
    monkeypatch.setenv("YOLO_AGENT_TEST_OPENAI_KEY", "test-key")
    result, calls = _run({
        "schema_version": "llm_doctor_decision.v2",
        "diagnosis": "Small-object recall is the primary problem.",
        "likely_causes": ["Sampling under-represents small boxes."],
        "evidence": [{"source_id": "fact-1", "statement": "AP_small is low."}],
        "selected_paper_priors": ["prior-1", "invented-prior"],
        "selected_recipes": ["recipe-1"],
        "rejected_paper_priors": [],
        "rejection_reasons": {},
        "evidence_requests": [],
        "implementation_requests": [],
        "expected_improvement": {"map50_95": "possible positive pilot delta"},
        "confidence": 0.6,
        "cost": {"training": "low"},
        "stop_condition": ["Stop when paired AP_small does not improve."],
        "candidate_policies": [_policy()],
    }, _context())

    assert len(calls) == 1
    assert result.status == "used"
    assert result.proposal_bundle.diagnosis.startswith("Small-object")
    assert result.proposal_bundle.selected_paper_priors == ["prior-1"]
    assert result.proposal_bundle.selected_recipes == ["recipe-1"]
    assert len(result.proposals) == 1
    assert "unknown_paper_prior_id:invented-prior" in result.warnings


def test_metadata_only_and_missing_evidence_cannot_run_training(monkeypatch) -> None:
    monkeypatch.setenv("YOLO_AGENT_TEST_OPENAI_KEY", "test-key")
    result, _ = _run({
        "diagnosis": "Evidence is incomplete.",
        "selected_paper_priors": ["prior-1"],
        "selected_recipes": ["recipe-1"],
        "implementation_requests": [],
        "candidate_policies": [_policy()],
    }, _context(maturity="metadata_only", missing=["paired_error_delta"]))

    assert result.proposals == []
    assert result.proposal_bundle.implementation_requests == [{
        "component_id": "component-1",
        "reason": "adapter implementation required",
    }]
    assert any("metadata_only_component:component-1" in item for item in result.warnings)
    assert any("missing_key_evidence_blocks_run_training" in item for item in result.warnings)


def test_fixed_imgsz_full_and_invented_evidence_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv("YOLO_AGENT_TEST_OPENAI_KEY", "test-key")
    result, _ = _run({
        "diagnosis": "Localization error.",
        "evidence": [{"source_id": "invented-benchmark", "statement": "+5 mAP"}],
        "selected_recipes": ["recipe-1"],
        "candidate_policies": [_policy(
            action_id="candidate_full",
            train_overrides={"imgsz": 1280},
        )],
    }, _context())

    assert result.proposals == []
    assert result.proposal_bundle.evidence == []
    assert any("candidate_full_forbidden" in item for item in result.warnings)
    assert any("fixed_imgsz_640_violation" in item for item in result.warnings)
    assert "invented_evidence_source:invented-benchmark" in result.warnings


def test_recipe_critic_rejection_overrides_llm_selection(monkeypatch) -> None:
    monkeypatch.setenv("YOLO_AGENT_TEST_OPENAI_KEY", "test-key")
    context = _context()
    context["recipe_critic_results"] = [{"recipe_id": "recipe-1", "accepted": False}]
    result, _ = _run({
        "diagnosis": "Small-object error.",
        "selected_recipes": ["recipe-1"],
        "candidate_policies": [],
    }, context)
    assert result.proposal_bundle.selected_recipes == []
    assert "recipe_critic_rejected:recipe-1" in result.warnings
