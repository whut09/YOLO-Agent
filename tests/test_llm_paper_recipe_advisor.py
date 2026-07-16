import json
from pathlib import Path

from yolo_agent.agents.llm_decision_advisor import LLMDecisionAdvisor
from yolo_agent.agents.paper_recipe_planner import PaperRecipePlan
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.decision_ledger import DecisionLedger
from yolo_agent.core.llm_config import LLMDecisionConfig
from yolo_agent.research.schemas import PaperRecord


def _config() -> LLMDecisionConfig:
    return LLMDecisionConfig(enabled=True, provider="openai", model="gpt-5.5", api_key="test-key", use_by_default=True, require_api_key=True)


def _paper() -> PaperRecord:
    return PaperRecord(paper_id="paper:small", title="Small Object Recipe", year=2025)


def _contract(maturity="smoke_passed") -> ComponentContract:
    return ComponentContract(
        component_id="sampling.small_object", display_name="Small sampler", category="sampling",
        implementation_path="local", adapter_class="SmallObjectSamplingAdapter", maturity=maturity,
        fixed_imgsz_compatible=True,
    )


def _call(advisor, tmp_path: Path, *, contracts=None, facts=None, evidence=None):
    return advisor.propose_paper_recipe(
        top_error_facts=facts if facts is not None else [{"metric_name": "ap_small", "severity": "high"}],
        paper_records=[_paper()],
        component_contracts=contracts or [_contract()],
        compatibility_results={"sampling.small_object": {"compatible": True}},
        policy_memory=[{"action": "small_object_sampling", "delta": 0.01}],
        prior_pilot_deltas=[{"component_id": "sampling.small_object", "ap_small": 0.01}],
        fixed_constraints={"imgsz": 640},
        budget={"profile": "pilot", "gpu_hours": 2},
        available_executable_adapters=["SmallObjectSamplingAdapter"],
        local_evidence=evidence if evidence is not None else [{"metric_name": "ap_small", "verified": True}],
        fallback_plan=PaperRecipePlan(evidence_actions=["rule_fallback"]),
        decision_ledger=DecisionLedger(tmp_path / "decision_ledger.jsonl"),
        run_id="run-1",
    )


def test_doctor_style_recipe_accepts_only_grounded_executable_component(tmp_path: Path) -> None:
    def transport(config, messages):
        assert "allowed_component_ids" in messages[1]["content"]
        return json.dumps({
            "primary_problem": "AP_small is low",
            "likely_causes": ["small-object exposure is weak"],
            "evidence": [
                {"statement": "local AP_small is low", "source_id": "node:baseline", "evidence_level": "local_evidence"},
                {"statement": "paper reports a sampling prior", "source_id": "paper:small", "evidence_level": "paper_prior"},
            ],
            "selected_recipe": "small_object_sampling",
            "execution_action": "run_training",
            "training_profile": "pilot",
            "selected_components": [{"component_id": "sampling.small_object", "role": "training sampler", "maturity": "smoke_passed", "rationale": "targets AP_small", "paper_prior_ids": ["paper:small"]}],
            "rejected_components": [], "rejected_reasons": {},
            "expected_improvement": {"ap_small": "+0.2 to +0.8"}, "confidence": 0.55,
            "cost": {"gpu_hours": 2.0, "risk": "low"},
            "stop_condition": ["stop when AP_small does not improve"],
            "evidence_requests": [], "implementation_requests": [],
            "fixed_constraints": {"imgsz": 640},
        })

    result = _call(LLMDecisionAdvisor(config=_config(), transport=transport), tmp_path)
    assert result.status == "used" and result.critic.accepted
    assert result.proposal.training_profile == "pilot"
    records = DecisionLedger(tmp_path / "decision_ledger.jsonl").read()
    assert records[0].decision_type == "llm_paper_recipe"
    assert records[0].prompt_sha256 == result.prompt_sha256
    assert records[0].proposal["critic"]["accepted"] is True


def test_metadata_component_can_only_request_implementation(tmp_path: Path) -> None:
    def transport(config, messages):
        return json.dumps({
            "primary_problem": "AP_small is low", "likely_causes": [], "evidence": [],
            "selected_recipe": None, "execution_action": "implementation_only", "training_profile": None,
            "selected_components": [], "rejected_components": [], "rejected_reasons": {},
            "expected_improvement": {}, "confidence": 0.2, "cost": {}, "stop_condition": [],
            "evidence_requests": [],
            "implementation_requests": [{"component_id": "sampling.small_object", "current_maturity": "metadata_only", "required_adapter": "SmallObjectSamplingAdapter", "reason": "adapter missing", "acceptance_tests": ["shape", "smoke"]}],
            "fixed_constraints": {"imgsz": 640},
        })

    result = _call(LLMDecisionAdvisor(config=_config(), transport=transport), tmp_path, contracts=[_contract("metadata_only")])
    assert result.critic.accepted
    assert result.proposal.implementation_requests[0].component_id == "sampling.small_object"


def test_missing_evidence_and_invented_component_force_rule_fallback(tmp_path: Path) -> None:
    def transport(config, messages):
        return json.dumps({
            "primary_problem": "unknown", "likely_causes": [], "evidence": [],
            "selected_recipe": "invented", "execution_action": "run_training", "training_profile": "pilot",
            "selected_components": [{"component_id": "invented.component", "role": "head", "maturity": "smoke_passed", "rationale": "guess", "paper_prior_ids": []}],
            "rejected_components": [], "rejected_reasons": {}, "expected_improvement": {"map": 1.0},
            "confidence": 0.9, "cost": {}, "stop_condition": [], "evidence_requests": [],
            "implementation_requests": [], "fixed_constraints": {"imgsz": 640},
        })

    result = _call(LLMDecisionAdvisor(config=_config(), transport=transport), tmp_path, facts=[], evidence=[])
    assert result.status == "failed" and not result.critic.accepted
    assert "unknown_component:invented.component" in result.critic.rejection_reasons
    assert "critical_evidence_missing_blocks_run_training" in result.critic.rejection_reasons
    assert result.fallback_plan.evidence_actions == ["rule_fallback"]


def test_llm_failure_preserves_rule_planner_result(tmp_path: Path) -> None:
    def transport(config, messages):
        raise TimeoutError("offline")

    result = _call(LLMDecisionAdvisor(config=_config(), transport=transport), tmp_path)
    assert result.status == "failed"
    assert result.proposal is None
    assert result.fallback_plan.evidence_actions == ["rule_fallback"]


def test_paper_claim_must_be_marked_as_prior(tmp_path: Path) -> None:
    def transport(config, messages):
        return json.dumps({
            "primary_problem": "AP_small low", "likely_causes": [],
            "evidence": [{"statement": "reported improvement", "source_id": "paper:small", "evidence_level": "local_evidence"}],
            "selected_recipe": None, "execution_action": "none", "training_profile": None,
            "selected_components": [], "rejected_components": [], "rejected_reasons": {},
            "expected_improvement": {}, "confidence": 0.1, "cost": {}, "stop_condition": [],
            "evidence_requests": [], "implementation_requests": [], "fixed_constraints": {"imgsz": 640},
        })

    result = _call(LLMDecisionAdvisor(config=_config(), transport=transport), tmp_path)
    assert "paper_claim_not_marked_prior:paper:small" in result.critic.rejection_reasons
