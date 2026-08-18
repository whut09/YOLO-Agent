"""Domain-adaptation branch and 40-paper coverage tests. No GPU training."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
    DomainAdaptationBranchAdapter,
    DomainAdaptationBranchConfig,
    coco_only_context,
    explicit_source_target_context,
)
from pydantic import ValidationError

from yolo_agent.components.adapters.domain_adaptation.branches import (
    NAMED_PAPER_BRANCHES,
    certified_domain_adaptation_papers,
    default_domain_adaptation_registry,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.research.paper_protocol_contract import PaperProtocolContext


def _contract(component_id: str) -> ComponentContract:
    return ComponentContract(
        component_id=component_id,
        display_name=component_id,
        category="domain_adaptation",
        implementation_path="yolo_agent.components.adapters.domain_adaptation.branch_runtime",
        adapter_class="DomainAdaptationBranchAdapter",
        maturity="adapter_implemented",
    )


def test_forty_domain_papers_are_bound_without_silent_drop() -> None:
    papers = certified_domain_adaptation_papers()
    assert len(papers) == 40
    assert set(papers) == set(NAMED_PAPER_BRANCHES)
    coverage = default_domain_adaptation_registry().coverage(coco_only_context())
    assert coverage.papers_total == 40
    assert coverage.silent_drops == []
    assert coverage.candidate == 0
    assert coverage.evidence_recovery + coverage.incompatible == 40
    for item in coverage.assignments:
        assert item.allows_asha is False
        assert item.disposition in {"evidence_recovery", "incompatible", "implementation_request"}


def test_explicit_source_target_can_become_candidates() -> None:
    coverage = default_domain_adaptation_registry().coverage(explicit_source_target_context())
    assert coverage.candidate >= 1
    assert coverage.silent_drops == []
    assert all(item.allows_asha is True for item in coverage.assignments if item.disposition == "candidate")


def test_missing_target_domain_stays_evidence_recovery() -> None:
    assignment = default_domain_adaptation_registry().assign(
        "cvf:cvpr2022:Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection",
        PaperProtocolContext(has_source_domain_data=True, has_target_domain_data=False),
    )
    assert assignment.disposition == "evidence_recovery"
    assert assignment.allows_asha is False
    assert "provide_target_domain_dataset" in assignment.missing_dataset_actions


def test_coco_masquerade_is_incompatible() -> None:
    assignment = default_domain_adaptation_registry().assign(
        "arxiv:2210.11539",
        PaperProtocolContext(
            has_source_domain_data=True,
            has_target_domain_data=True,
            coco_train_used_as_source=True,
            coco_val_used_as_target=True,
        ),
    )
    assert assignment.disposition == "incompatible"
    assert assignment.allows_asha is False


def test_adapter_existence_does_not_authorize_asha() -> None:
    branch = default_domain_adaptation_registry().get("adversarial_feature_alignment")
    assert branch.adapter_alone_authorizes_asha is False
    assert branch.contaminates_coco_baseline is False
    coco = default_domain_adaptation_registry().coverage(coco_only_context())
    assert coco.candidate == 0


def test_coco_manifest_cannot_be_used_as_both_domains() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        DomainAdaptationBranchConfig(
            branch_id="adversarial_feature_alignment",
            source_manifest="coco-train",
            target_manifest="coco-train",
        )
    with pytest.raises(ValidationError, match="masquerade"):
        DomainAdaptationBranchConfig(
            branch_id="adversarial_feature_alignment",
            source_manifest="source",
            target_manifest="target",
            coco_train_used_as_source=True,
        )


@pytest.mark.parametrize(
    "branch_id",
    list(dict.fromkeys(NAMED_PAPER_BRANCHES.values())),
)
def test_each_branch_has_independent_cpu_smoke(branch_id: str) -> None:
    branch = default_domain_adaptation_registry().get(branch_id)
    adapter = DomainAdaptationBranchAdapter(branch_id)
    context = AdapterContext(
        contract=_contract(branch.component_id),
        detector_family="yolo26",
        options={"branch_id": branch_id, "source_manifest": "src", "target_manifest": "tgt"},
    )
    smoke = adapter.smoke_test(context)
    assert smoke.passed is True
    assert smoke.checks["explicit_source_target_batch"] is True


def test_coverage_fixture_exists() -> None:
    path = Path("tests/fixtures/domain_adaptation_paper_coverage.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["papers_total"] == 40
    assert payload["silent_drops"] == []
