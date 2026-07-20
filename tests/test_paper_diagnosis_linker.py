from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.diagnosis_graph import DiagnosisGraph
from yolo_agent.agents.paper_diagnosis_linker import PaperDiagnosisLinker, PaperDiagnosisRules
from yolo_agent.core.error_facts import ErrorFact, ErrorFactType
from yolo_agent.research.harness_hint_parser import PaperDiagnosticHint
from yolo_agent.research.schemas import ComponentTaxonomy, PaperComponentClaim
from yolo_agent.research.snapshot import ResearchSnapshot
from yolo_agent.resources import ResourcePaths


def _snapshot(*, available: bool = True, frozen: bool = True) -> ResearchSnapshot:
    return ResearchSnapshot.model_construct(
        snapshot_hash="snapshot-hash",
        paper_intelligence="available" if available else "unavailable",
        unavailable_reason=None if available else "empty_catalog",
        frozen=frozen,
    )


def _fact(
    fact_type: ErrorFactType,
    subject: str,
    *,
    metric_name: str | None = None,
    area: str | None = None,
    actions: list[str] | None = None,
) -> ErrorFact:
    return ErrorFact(
        run_id="run",
        candidate_id="candidate",
        node_id="node",
        fact_type=fact_type,
        subject=subject,
        metric_name=metric_name,
        area=area,
        severity="high",
        action_candidates=actions or [],
    )


def _hint(metric: str, error_fact: str, *, paper_id: str = "paper-1") -> PaperDiagnosticHint:
    return PaperDiagnosticHint(
        paper_id=paper_id,
        symptom=error_fact,
        likely_cause="paper prior",
        evidence_needed=["per_class_ap_ar"],
        target_metrics=[metric],
        target_error_facts=[error_fact],
        confidence=0.8,
        source_location="notes/paper.md#diagnosis",
    )


def _link(
    fact: ErrorFact,
    hint: PaperDiagnosticHint,
    *,
    claims: list[PaperComponentClaim] | None = None,
    snapshot: ResearchSnapshot | None = None,
):
    return PaperDiagnosisLinker().link(
        error_facts=[fact],
        dataset_report=None,
        coco_post_eval={},
        per_class_ap_ar={},
        confusion_summary={},
        diagnostic_hints=[hint],
        component_claims=claims or [],
        taxonomy=ComponentTaxonomy(categories={}),
        research_snapshot=snapshot or _snapshot(),
    )


@pytest.mark.parametrize(
    ("fact", "hint", "expected_candidates", "expected_rejected"),
    [
        (
            _fact("area_metric", "small objects", metric_name="ap_small", area="small"),
            _hint("ap_small", "small_object"),
            {"small_object_sampling", "multi_scale_feature", "p2_head", "slicing", "deformable_attention"},
            set(),
        ),
        (
            _fact("localization_heavy_class", "high confidence localization error"),
            _hint("map50_95", "localization_error"),
            {"iou_aware_classification", "task_alignment", "bbox_regression"},
            {"dfl_localization_loss"},
        ),
        (
            _fact("class_confusion_pair", "duplicate predictions", actions=["assignment_review"]),
            _hint("precision", "duplicate_prediction"),
            {"assignment", "nms_free_duplicate_suppression"},
            {"nms_duplicate_suppression"},
        ),
        (
            _fact("subset_performance", "slow convergence", actions=["optimization_schedule_review"]),
            _hint("map50_95", "slow_convergence"),
            {"matching_strategy", "optimization_schedule"},
            {"denoising"},
        ),
        (
            _fact("class_low_ap", "long tail new class", actions=["hard_negative_mining"]),
            _hint("recall", "class_imbalance"),
            {"data_augmentation", "active_learning", "hard_negative_mining"},
            {"open_vocabulary"},
        ),
    ],
)
def test_links_each_supported_symptom_to_paper_families(
    fact: ErrorFact,
    hint: PaperDiagnosticHint,
    expected_candidates: set[str],
    expected_rejected: set[str],
) -> None:
    result = _link(fact, hint)

    candidates = {item.family_id for item in result.candidate_component_families}
    rejected = {item.family_id for item in result.rejected_paper_families}
    assert expected_candidates <= candidates
    assert expected_rejected <= rejected
    assert result.likely_causes
    assert result.confidence > 0
    assert result.paper_prior_summary.evidence_level == "paper_claim"
    assert result.paper_prior_summary.used_as_local_evidence is False


@pytest.mark.parametrize(
    "forbidden_family",
    [
        "small_object_sampling",
        "iou_aware_classification",
        "assignment",
        "optimization_schedule",
        "active_learning",
    ],
)
def test_unrelated_error_fact_does_not_link_symptom_family(forbidden_family: str) -> None:
    result = _link(
        _fact("per_class_metric", "ordinary class metric", metric_name="ap_medium"),
        _hint("ap_medium", "ordinary_metric"),
    )

    families = {
        item.family_id
        for item in [*result.candidate_component_families, *result.rejected_paper_families]
    }
    assert forbidden_family not in families
    assert not result.likely_causes


def test_missing_error_facts_returns_only_evidence_requests() -> None:
    result = PaperDiagnosisLinker().link(
        error_facts=[],
        dataset_report=None,
        coco_post_eval=None,
        per_class_ap_ar=None,
        confusion_summary=None,
        diagnostic_hints=[_hint("ap_small", "small_object")],
        component_claims=[],
        taxonomy=ComponentTaxonomy(categories={}),
        research_snapshot=_snapshot(),
    )

    assert result.confidence == 0
    assert result.evidence_requests
    assert not result.diagnosis_linked_papers
    assert not result.likely_causes
    assert not result.candidate_component_families
    assert not result.rejected_paper_families


@pytest.mark.parametrize("snapshot", [_snapshot(available=False), _snapshot(frozen=False)])
def test_unavailable_or_unfrozen_snapshot_blocks_paper_priors(snapshot: ResearchSnapshot) -> None:
    result = _link(
        _fact("area_metric", "small objects", metric_name="ap_small", area="small"),
        _hint("ap_small", "small_object"),
        snapshot=snapshot,
    )

    assert [item.evidence for item in result.evidence_requests] == ["frozen_research_snapshot"]
    assert not result.likely_causes
    assert not result.candidate_component_families


def test_links_paper_provenance_without_promoting_claim_to_local_evidence() -> None:
    claim = PaperComponentClaim(
        paper_id="paper-1",
        component_id="head.p2_small_object",
        component_category="detection_head",
        claimed_effect="Improves small-object detection.",
        evidence_level="paper_claim",
        target_metrics=["ap_small"],
        target_error_types=["small_object"],
    )
    result = _link(
        _fact("area_metric", "small objects", metric_name="ap_small", area="small"),
        _hint("ap_small", "small_object"),
        claims=[claim],
    )

    linked = result.diagnosis_linked_papers[0]
    assert linked.paper_id == "paper-1"
    assert linked.source_locations == ["notes/paper.md#diagnosis"]
    assert linked.evidence_level == "paper_claim"
    assert all(cause.local_evidence_role == "diagnostic_trigger_only" for cause in result.likely_causes)
    serialized = result.model_dump(mode="json")
    assert "CommandSpec" not in str(serialized)
    assert "run_training" not in str(serialized)


def test_existing_diagnosis_graph_exposes_paper_prior_bridge() -> None:
    facts = [_fact("area_metric", "small objects", metric_name="ap_small", area="small")]
    local, paper = DiagnosisGraph(rules=[]).diagnose_with_paper_priors(
        facts,
        paper_linker=PaperDiagnosisLinker(),
        dataset_report=None,
        coco_post_eval={"ap_small": 0.1},
        per_class_ap_ar={"person": {"ap": 0.1}},
        confusion_summary={},
        diagnostic_hints=[_hint("ap_small", "small_object")],
        component_claims=[],
        taxonomy=ComponentTaxonomy(categories={}),
        research_snapshot=_snapshot(),
    )

    assert local.findings == []
    assert {item.family_id for item in paper.candidate_component_families} >= {"small_object_sampling", "p2_head"}


def test_rules_load_from_bundled_config_offline() -> None:
    rules = PaperDiagnosisRules.from_yaml()

    assert Path(ResourcePaths.PAPER_DIAGNOSIS_RULES).is_file()
    assert {item.rule_id for item in rules.rules} == {
        "small_object_ap_low",
        "high_confidence_localization_poor",
        "duplicate_predictions",
        "slow_convergence",
        "new_class_or_long_tail",
    }
