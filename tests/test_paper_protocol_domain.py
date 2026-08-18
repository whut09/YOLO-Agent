"""Domain-adaptation paper protocol tests. No GPU training."""

from __future__ import annotations

from yolo_agent.research.paper_evidence_requirements import missing_dataset_actions
from yolo_agent.research.paper_protocol_catalog import build_paper_protocol_contract
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolContext,
    evaluate_paper_protocol,
)


def test_domain_adaptation_without_domains_is_evidence_recovery() -> None:
    contract = build_paper_protocol_contract(
        "cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection"
    )
    evaluation = evaluate_paper_protocol(contract, PaperProtocolContext())
    assert evaluation.ok is False
    assert evaluation.disposition == "evidence_recovery"
    assert "domain_adaptation_blocked_from_coco_map_training" in evaluation.reason_codes
    assert "domain_source_data_missing" in evaluation.reason_codes
    assert "domain_target_data_missing" in evaluation.reason_codes
    assert "provide_source_domain_dataset" in evaluation.missing_dataset_actions
    assert "provide_target_domain_dataset" in evaluation.missing_dataset_actions
    assert evaluation.allows_asha_registration is False


def test_domain_adaptation_rejects_coco_as_paper_domain() -> None:
    contract = build_paper_protocol_contract("arxiv:2210.11539")
    evaluation = evaluate_paper_protocol(
        contract,
        PaperProtocolContext(
            has_source_domain_data=True,
            has_target_domain_data=True,
            coco_train_used_as_source=True,
            coco_val_used_as_target=True,
        ),
    )
    assert evaluation.disposition == "incompatible"
    assert "coco_split_cannot_stand_in_for_paper_domain" in evaluation.reason_codes
    assert evaluation.allows_materialization is False


def test_source_free_domain_adaptation_does_not_require_source_data() -> None:
    contract = build_paper_protocol_contract(
        "cvf:cvpr2023:VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection"
    )
    assert contract.annotation_requirement == "source_free"
    evaluation = evaluate_paper_protocol(
        contract,
        PaperProtocolContext(has_target_domain_data=True),
    )
    assert "domain_source_data_missing" not in evaluation.reason_codes
    assert evaluation.ok is True


def test_missing_dataset_actions_are_explicit() -> None:
    assert "do_not_reuse_coco_train_val_as_paper_domains" in missing_dataset_actions()
