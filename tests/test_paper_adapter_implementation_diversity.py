from __future__ import annotations

from yolo_agent.agents.paper_adapter_implementation_planner import (
    AdapterImplementationEstimate,
    ImplementationHistoryRecord,
    PaperAdapterImplementationPlanner,
    RuntimeHookAvailability,
)
from yolo_agent.agents.paper_adapter_planning.diagnosis import (
    build_adapter_diagnosis_context,
    match_component_to_diagnosis,
)
from yolo_agent.agents.paper_adapter_planning.local_evidence import (
    assess_local_component_evidence,
)
from yolo_agent.agents.paper_adapter_planning.policy import PaperAdapterPlanningPolicy
from yolo_agent.agents.paper_adapter_planning.runtime_hooks import assess_runtime_hook
from yolo_agent.agents.paper_adapter_planning.scoring import score_adapter_implementation
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.policy_memory import ActionFingerprint, PolicyMemoryRecord
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


def _paper(paper_id: str, component_id: str, *, year: int = 2025) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        year=year,
        official_code_url="https://example.test/code",
        code_license="Apache-2.0",
        component_ids=[component_id],
        applicability="direct_adapter_candidate",
        source="awesome_object_detection",
        ingestion_version="test.v1",
        evidence_level="paper_prior",
    )


def test_confirmed_local_negative_evidence_outweighs_freshness() -> None:
    resolver = ComponentAliasResolver.from_yaml()
    mapping = resolver.resolve("dynamic_head").mappings[0]
    paper = _paper("fresh", "dynamic_head", year=2026)
    policy = PaperAdapterPlanningPolicy()
    record = PolicyMemoryRecord(
        run_id="negative-run",
        dataset_version="coco-signature",
        action="dynamic head adapter",
        action_fingerprint=ActionFingerprint(
            action="dynamic head adapter",
            component_ids=[mapping.canonical_component_id],
            dataset_signature="coco-signature",
        ),
        target="map50_95",
        metric_name="map50_95",
        effect_delta=-0.02,
        confidence="high",
        seed_count=3,
        evidence_status="confirmed",
    )
    local = assess_local_component_evidence(
        mapping.canonical_component_id,
        [record],
        dataset_signature="coco-signature",
        positive_max_weight=policy.local_positive_max_weight,
        negative_max_penalty=policy.local_negative_max_penalty,
    )
    diagnosis = build_adapter_diagnosis_context([_fact()])
    estimate = AdapterImplementationEstimate(
        component_id=mapping.canonical_component_id,
        required_runtime_hook="detection_head",
    )
    score = score_adapter_implementation(
        paper=paper,
        mapping=mapping,
        estimate=estimate,
        diagnosis_context=diagnosis,
        diagnosis_match=match_component_to_diagnosis(mapping, diagnosis),
        runtime_hook=assess_runtime_hook(
            estimate,
            [RuntimeHookAvailability(hook_id="detection_head", available=True, verified=True)],
            verified_weight=policy.runtime_hook_weight,
        ),
        local_evidence=local,
        policy=policy,
    )

    assert score.breakdown["paper_freshness"] == policy.freshness_max_weight
    assert score.breakdown["local_evidence"] == policy.local_negative_max_penalty
    assert score.breakdown["local_evidence"] + score.breakdown["paper_freshness"] < 0
    assert "local_negative_evidence_outweighs_freshness_prior" in score.reasons


def test_duplicate_papers_collapse_to_one_implementation_request() -> None:
    planner = PaperAdapterImplementationPlanner(ComponentAliasResolver.from_yaml())
    estimate = AdapterImplementationEstimate(
        component_id="detection_head.dynamic",
        required_runtime_hook="detection_head",
    )
    plan = planner.plan(
        papers=[_paper("paper-a", "dynamic_head"), _paper("paper-b", "dynamic_head")],
        error_facts=[_fact()],
        implementation_estimates=[estimate],
        runtime_hooks=[RuntimeHookAvailability(
            hook_id="detection_head", available=True, verified=True,
        )],
    )

    assert len(plan.implementation_queue) == 1
    assert plan.implementation_queue[0].paper_ids == ["paper-a", "paper-b"]
    assert plan.implementation_queue[0].implementation_request is not None
    assert plan.implementation_queue[0].implementation_request.paper_ids == ["paper-a", "paper-b"]


def test_duplicate_fingerprint_and_family_cooldown_defer_work() -> None:
    planner = PaperAdapterImplementationPlanner(ComponentAliasResolver.from_yaml())
    estimate = AdapterImplementationEstimate(
        component_id="detection_head.dynamic",
        required_runtime_hook="detection_head",
    )
    hooks = [RuntimeHookAvailability(
        hook_id="detection_head", available=True, verified=True,
    )]
    first = planner.plan(
        papers=[_paper("dynamic", "dynamic_head")],
        error_facts=[_fact()],
        implementation_estimates=[estimate],
        runtime_hooks=hooks,
        current_round=1,
    )
    item = first.implementation_queue[0]
    duplicate = planner.plan(
        papers=[_paper("dynamic-new-paper", "dynamic_head")],
        error_facts=[_fact()],
        implementation_estimates=[estimate],
        runtime_hooks=hooks,
        current_round=2,
        history=[ImplementationHistoryRecord(
            fingerprint=item.fingerprint,
            component_family=item.component_family,
            round_index=1,
        )],
    )
    p2 = planner.plan(
        papers=[_paper("p2", "p2_head")],
        error_facts=[_fact()],
        implementation_estimates=[AdapterImplementationEstimate(
            component_id="head.p2_small_object",
            required_runtime_hook="feature_pyramid_p2",
        )],
        runtime_hooks=[RuntimeHookAvailability(
            hook_id="feature_pyramid_p2", available=True, verified=True,
        )],
        current_round=2,
        history=[ImplementationHistoryRecord(
            fingerprint="different-fingerprint",
            component_family="detection_head",
            round_index=1,
        )],
    )

    assert "duplicate_implementation_fingerprint" in duplicate.deferred[0].reasons
    assert any(reason.startswith("implementation_family_cooldown") for reason in p2.deferred[0].reasons)
    assert duplicate.deferred[0].implementation_request is None
