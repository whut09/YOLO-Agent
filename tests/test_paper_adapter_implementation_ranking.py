from __future__ import annotations

from yolo_agent.agents.paper_adapter_implementation_planner import (
    AdapterImplementationEstimate,
    PaperAdapterImplementationPlanner,
    RuntimeHookAvailability,
)
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.schemas import PaperRecord


def _small_fn_fact() -> ErrorFact:
    return ErrorFact(
        run_id="run",
        candidate_id="baseline",
        node_id="post-eval",
        dataset_version="coco",
        split="val2017",
        fact_type="false_negative_heavy_class",
        subject="small objects",
        area="small",
        metric_name="ap_small",
        severity="high",
        source="coco_post_eval",
    )


def _paper(paper_id: str, component_id: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper for {component_id}",
        year=2025,
        official_code_url=f"https://example.test/{paper_id}",
        code_license="Apache-2.0",
        component_ids=[component_id],
        applicability="direct_adapter_candidate",
        source="awesome_object_detection",
        ingestion_version="test.v1",
        evidence_level="paper_prior",
    )


def test_ap_small_fn_diagnosis_prioritizes_four_runtime_components() -> None:
    planner = PaperAdapterImplementationPlanner(ComponentAliasResolver.from_yaml())
    papers = [
        _paper("sampling-paper", "small_object_sampling"),
        _paper("p2-paper", "p2_head"),
        _paper("distill-paper", "distillation"),
        _paper("sahi-paper", "sahi"),
    ]
    estimates = [
        AdapterImplementationEstimate(
            component_id="sampling.small_object",
            implementation_cost="low",
            expected_latency_cost="low",
            expected_model_size_cost="low",
            required_runtime_hook="train_dataloader_sampler",
        ),
        AdapterImplementationEstimate(
            component_id="head.p2_small_object",
            implementation_cost="low",
            expected_latency_cost="low",
            expected_model_size_cost="low",
            required_runtime_hook="feature_pyramid_p2",
        ),
        AdapterImplementationEstimate(
            component_id="distillation.yolo26_teacher_student",
            implementation_cost="low",
            expected_latency_cost="low",
            expected_model_size_cost="low",
            required_runtime_hook="trainer_loss",
        ),
        AdapterImplementationEstimate(
            component_id="inference.sahi_slicing",
            implementation_cost="low",
            expected_latency_cost="low",
            expected_model_size_cost="low",
            required_runtime_hook="inference_policy",
        ),
    ]
    hooks = [
        RuntimeHookAvailability(hook_id=item.required_runtime_hook or "", available=True, verified=True, version="v1")
        for item in estimates
    ]

    plan = planner.plan(
        papers=papers,
        error_facts=[_small_fn_fact()],
        implementation_estimates=estimates,
        runtime_hooks=hooks,
    )

    assert plan.ready_to_materialize == []
    assert [item.component_id for item in plan.shadow_evaluation_queue] == [
        "sampling.small_object",
        "head.p2_small_object",
        "distillation.yolo26_teacher_student",
        "inference.sahi_slicing",
    ]
    assert all(item.diagnosis_targets for item in plan.shadow_evaluation_queue)
    assert plan.implementation_queue == []
    assert plan.auto_code_generation is False
