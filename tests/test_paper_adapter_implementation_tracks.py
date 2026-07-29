from __future__ import annotations

from yolo_agent.agents.paper_adapter_implementation_planner import (
    AdapterImplementationEstimate,
    PaperAdapterImplementationPlanner,
    RuntimeHookAvailability,
)
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.schemas import PaperRecord


def _fact() -> ErrorFact:
    return ErrorFact(
        run_id="run",
        candidate_id="baseline",
        node_id="eval",
        dataset_version="coco",
        split="val2017",
        fact_type="area_metric",
        subject="small objects",
        area="small",
        metric_name="ap_small",
        severity="high",
    )


def _paper(
    paper_id: str,
    component_id: str,
    *,
    title: str = "Detection component",
    detector_family: str | None = None,
    applicability: str = "direct_adapter_candidate",
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        year=2025,
        detector_family=detector_family,
        official_code_url="https://example.test/code",
        code_license="Apache-2.0",
        component_ids=[component_id],
        applicability=applicability,
        source="awesome_object_detection",
        ingestion_version="test.v1",
        evidence_level="paper_prior",
    )


def test_direct_adapter_candidate_generates_request_not_executable_code() -> None:
    planner = PaperAdapterImplementationPlanner(ComponentAliasResolver.from_yaml())
    plan = planner.plan(
        papers=[_paper("dynamic", "dynamic_head")],
        error_facts=[_fact()],
        implementation_estimates=[AdapterImplementationEstimate(
            component_id="detection_head.dynamic",
            implementation_cost="medium",
            required_runtime_hook="detection_head",
        )],
        runtime_hooks=[RuntimeHookAvailability(
            hook_id="detection_head", available=True, verified=True,
        )],
    )

    assert plan.ready_to_materialize == []
    assert len(plan.implementation_queue) == 1
    item = plan.implementation_queue[0]
    assert item.implementation_status in {"metadata_only", "adapter_required"}
    assert item.implementation_request is not None
    assert item.implementation_request.generated_code_allowed is False
    assert "direct_adapter_candidate_is_prior_not_execution_status" in item.reasons


def test_detr_open_vocabulary_and_open_world_use_separate_track() -> None:
    planner = PaperAdapterImplementationPlanner(ComponentAliasResolver.from_yaml())
    papers = [
        _paper("detr", "learnable_proposals", title="A Better DETR", detector_family="detr"),
        _paper("ovd", "open_vocabulary_detection", title="Open Vocabulary Detection"),
        _paper("owd", "dynamic_head", title="Open World Object Detection"),
    ]

    plan = planner.plan(papers=papers, error_facts=[_fact()])

    assert {paper for item in plan.separate_detector_family for paper in item.paper_ids} == {
        "detr", "ovd", "owd",
    }
    assert plan.implementation_queue == []


def test_incompatible_unresolved_and_missing_diagnosis_are_not_actionable() -> None:
    planner = PaperAdapterImplementationPlanner(ComponentAliasResolver.from_yaml())
    incompatible = planner.plan(
        papers=[_paper("proposal", "learnable_proposals")],
        error_facts=[_fact()],
    )
    unresolved = planner.plan(
        papers=[_paper("unknown", "not_a_real_component")],
        error_facts=[_fact()],
    )
    no_diagnosis = planner.plan(
        papers=[_paper("dynamic", "dynamic_head")],
        error_facts=[],
    )

    assert incompatible.incompatible[0].component_id == "detection_head.learnable_proposals"
    assert unresolved.insufficient_information[0].component_id == "not_a_real_component"
    assert no_diagnosis.insufficient_information[0].component_id == "detection_head.dynamic"
