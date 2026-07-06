"""LLM proposal critic tests."""

from __future__ import annotations

from yolo_agent.agents.llm_proposal_critic import LLMProposalCritic
from yolo_agent.agents.strategy_policy import CandidatePolicy


def _policy(**updates: object) -> CandidatePolicy:
    data: dict[str, object] = {
        "policy_id": "llm_safe_sampling",
        "source": "llm",
        "action_domain": "data",
        "action_id": "small_object_oversampling",
        "execution_action": "run_training",
        "base_model": "yolo26n.pt",
        "scale": "n",
        "framework": "ultralytics",
        "train_overrides": {"data_action": "small_object_oversampling"},
        "target_error_facts": [{"metric_name": "ap_small", "area": "small"}],
        "expected_improvement": {"metric_name": "ap_small", "minimum_expected_delta": "pilot_positive_delta"},
        "expected_effect": ["Targets AP_small without changing imgsz."],
        "risk": "low",
    }
    data.update(updates)
    return CandidatePolicy.model_validate(data)


def test_llm_proposal_critic_accepts_single_variable_bound_policy() -> None:
    """A single-variable proposal bound to error facts should pass the critic."""
    report = LLMProposalCritic().critique([_policy()], fixed_imgsz=640)

    assert report.accepted == 1
    assert report.rejected == 0
    assert report.rejection_reasons == []
    assert report.critiques[0].changed_variables == {"data_action": "small_object_oversampling"}


def test_llm_proposal_critic_rejects_missing_bindings_and_fixed_imgsz_violation() -> None:
    """The critic should reject common ungrounded LLM proposals early."""
    bad = _policy(
        policy_id="llm_bad_imgsz",
        action_domain="model",
        action_id=None,
        train_overrides={"imgsz": 960},
        target_error_facts=[],
        expected_improvement={},
    )

    report = LLMProposalCritic().critique([bad], fixed_imgsz=640)

    assert report.accepted == 0
    assert report.rejected == 1
    assert set(report.critiques[0].rejection_reasons) == {
        "missing_target_error_facts",
        "missing_expected_improvement",
        "violates_fixed_imgsz",
    }


def test_llm_proposal_critic_rejects_multi_variable_policy() -> None:
    """LLM proposals should not bundle multiple primary variables."""
    proposal = _policy(
        policy_id="llm_multi",
        components=["assigner.stal"],
        train_overrides={"data_action": "small_object_oversampling"},
    )

    report = LLMProposalCritic().critique([proposal], fixed_imgsz=640)

    assert report.rejected == 1
    assert "multi_variable_proposal" in report.critiques[0].rejection_reasons


def test_llm_proposal_critic_rejects_yolo26_incompatible_actions() -> None:
    """YOLO26-specific unsafe assumptions should be rejected before evaluation."""
    proposals = [
        _policy(policy_id="llm_nwd", components=["loss.bbox.nwd"]),
        _policy(policy_id="llm_p2", components=["head.p2_small_object"]),
        _policy(policy_id="llm_nms", train_overrides={"postprocess_action": "soft_nms"}),
    ]

    report = LLMProposalCritic().critique(proposals, fixed_imgsz=640)
    reasons = {reason for critique in report.critiques for reason in critique.rejection_reasons}

    assert report.rejected == 3
    assert "yolo26_loss_patch_requires_verified_recipe" in reasons
    assert "yolo26_p2_head_requires_verified_recipe" in reasons
    assert "yolo26_nms_incompatible_action" in reasons


def test_llm_proposal_critic_rejects_training_before_required_evidence() -> None:
    """Training proposals with unsatisfied evidence requirements should be blocked."""
    proposal = _policy(
        policy_id="llm_train_without_evidence",
        evidence_required=["per_class_ap_small"],
    )

    report = LLMProposalCritic().critique([proposal], fixed_imgsz=640)

    assert report.rejected == 1
    assert "pushes_training_before_required_evidence" in report.critiques[0].rejection_reasons


def test_llm_proposal_critic_allows_only_evidence_actions_when_diagnostic_evidence_missing() -> None:
    """Missing diagnostic facts should force LLM proposals into evidence-only mode."""
    training = _policy(policy_id="llm_train_small_object")
    evidence_action = _policy(
        policy_id="llm_import_ap_small",
        action_domain="evidence",
        action_id="import_coco_eval",
        execution_action="import_metrics",
        train_overrides={"evidence_action": "import_metrics", "missing_evidence": ["ap_small"]},
        target_error_facts=[],
        expected_improvement={},
    )

    report = LLMProposalCritic().critique(
        [training, evidence_action],
        fixed_imgsz=640,
        missing_diagnostic_evidence=["ap_small", "per_class_ap"],
    )
    by_id = {critique.policy_id: critique for critique in report.critiques}

    assert report.accepted == 1
    assert report.rejected == 1
    assert by_id["llm_import_ap_small"].accepted is True
    assert by_id["llm_train_small_object"].accepted is False
    assert "diagnostic_evidence_missing_requires_evidence_action" in by_id["llm_train_small_object"].rejection_reasons
    assert "diagnostic_evidence_missing_blocks_run_training" in by_id["llm_train_small_object"].rejection_reasons
