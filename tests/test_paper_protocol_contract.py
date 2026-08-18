"""PaperProtocolContract tests. No GPU training."""

from __future__ import annotations

import pytest

from yolo_agent.research.paper_protocol_catalog import (
    CERTIFIED_PAPER_MECHANISMS,
    build_paper_protocol_contract,
    certified_paper_ids,
    classify_protocol_family,
    load_certified_paper_protocols,
)
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolContract,
    PaperProtocolRegistry,
    authorize_paper_execution,
    compute_paper_protocol_hash,
    missing_protocol_evaluation,
    paper_ids_from_values,
)


REQUIRED_FIELDS = (
    "dataset_role",
    "source_split",
    "target_split",
    "annotation_requirement",
    "train_val_test_protocol",
    "imgsz",
    "model_family",
    "head_constraint",
    "teacher_requirement",
    "graph_change",
    "loss_change",
    "inference_change",
    "required_metrics",
    "paired_baseline_requirement",
    "required_evidence_artifacts",
    "protocol_hash",
)


def test_contract_is_independent_of_component_contract() -> None:
    contract = build_paper_protocol_contract("arxiv:2212.07784")
    assert "component_id" not in PaperProtocolContract.model_fields
    assert contract.schema_version == "paper_protocol_contract.v1"
    for field in REQUIRED_FIELDS:
        value = getattr(contract, field)
        assert value not in (None, "")
        if isinstance(value, list):
            assert value


def test_protocol_hash_is_stable_and_paper_specific() -> None:
    first = build_paper_protocol_contract("arxiv:2212.07784")
    second = build_paper_protocol_contract("arxiv:2212.07784")
    other = build_paper_protocol_contract("arxiv:2104.14082")
    assert first.protocol_hash == second.protocol_hash
    assert first.protocol_hash == compute_paper_protocol_hash(first)
    assert first.protocol_hash != other.protocol_hash
    assert len(first.protocol_hash) == 64


def test_missing_protocol_blocks_materialization_and_asha() -> None:
    evaluation = missing_protocol_evaluation("paper:missing")
    assert evaluation.ok is False
    assert evaluation.disposition == "blocked_runtime"
    assert evaluation.allows_materialization is False
    assert evaluation.allows_asha_registration is False
    assert "paper_protocol_missing" in evaluation.reason_codes


def test_authorize_unknown_paper_is_blocked() -> None:
    evaluation = authorize_paper_execution(
        ["paper:not-in-catalog"],
        registry=PaperProtocolRegistry(),
    )
    assert evaluation is not None
    assert evaluation.reason_codes == ["paper_protocol_missing"]


def test_certified_catalog_has_83_unique_protocol_hashes() -> None:
    paper_ids = certified_paper_ids()
    contracts = load_certified_paper_protocols()
    assert len(paper_ids) == 83
    assert len(contracts) == 83
    assert len(set(paper_ids)) == 83
    hashes = [item.protocol_hash for item in contracts]
    assert len(set(hashes)) == 83
    assert set(paper_ids) == set(CERTIFIED_PAPER_MECHANISMS)


@pytest.mark.parametrize("paper_id", certified_paper_ids())
def test_each_certified_paper_declares_full_protocol(paper_id: str) -> None:
    contract = build_paper_protocol_contract(paper_id)
    for field in REQUIRED_FIELDS:
        value = getattr(contract, field)
        assert value not in (None, "")
        if isinstance(value, list):
            assert value
    assert contract.imgsz == 640
    assert contract.model_family == "yolo26"
    assert contract.yolo26_one_to_one_head is True
    assert contract.native_dfl_free_regression is True
    assert contract.paired_baseline_requirement is True
    assert contract.protocol_hash
    family = classify_protocol_family(paper_id, contract.mechanism_ids)
    assert contract.protocol_family == family
    if family == "model_graph":
        assert contract.graph_identity
    if family == "domain_adaptation":
        assert contract.dataset_role == "source_target_domain"
        assert contract.source_split not in {"coco_train", "coco_val", "coco_test"}
        assert contract.target_split not in {"coco_train", "coco_val", "coco_test"}
    if family == "distillation":
        assert contract.teacher_requirement != "none"
        assert contract.train_val_test_protocol == "teacher_student_same_split"


def test_paper_ids_from_values_reads_prior_like_objects() -> None:
    class Prior:
        paper_ids = ["arxiv:2103.14259", "arxiv:2104.14082"]

    assert paper_ids_from_values(Prior(), "arxiv:2212.07784") == [
        "arxiv:2103.14259",
        "arxiv:2104.14082",
        "arxiv:2212.07784",
    ]
