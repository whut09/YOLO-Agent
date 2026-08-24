"""CPU-only end-to-end acceptance for the 83-paper readiness/ASHA boundary.

Production artifacts are inspected as evidence of coverage.  The ASHA path
uses explicitly labelled mock-ready nodes; those nodes prove routing only and
must never be counted as production readiness or trained coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.asha_scheduler import ASHAObservation, ASHAScheduler
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.execution_fingerprint import execution_fingerprint
from yolo_agent.core.paired_experiment import build_paired_experiment_result
from yolo_agent.core.round_execution_plan import build_round_execution_plan
from yolo_agent.recipes.registry import RecipeRegistry
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.executable_coverage import (
    ExecutablePaperCoverageAuditor,
    method_coverage_file_hash,
)
from yolo_agent.research.method_profiles import PaperMethodCoverageReport
from yolo_agent.research.paper_execution_inventory import (
    PaperExecutionInventoryBuilder,
)
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_protocol_catalog import (
    inference_only_protocol,
)
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolContext,
    default_paper_protocol_registry,
)
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.resources import ResourcePaths
from yolo_agent.certification.paper_readiness import PaperReadinessReport

from tests.test_all_83_paper_execution_acceptance import (
    _mock_baseline,
    _mock_node,
    _mock_ready_records,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_COVERAGE = (
    ROOT / "research" / "production" / "paper_method_coverage.yaml"
)
INVENTORY_PATH = ROOT / "runs" / "coverage-audit" / "paper_execution_inventory.yaml"
REQUIREMENTS_PATH = (
    ROOT / "runs" / "coverage-audit" / "paper_execution_requirements.yaml"
)
READINESS_PATH = ROOT / "runs" / "paper-readiness" / "paper_readiness_report.yaml"


@pytest.fixture(scope="module")
def production_inventory():  # type: ignore[no-untyped-def]
    coverage = PaperMethodCoverageReport.from_yaml(PRODUCTION_COVERAGE)
    aliases = ComponentAliasResolver.from_yaml()
    executable = ExecutablePaperCoverageAuditor(
        contracts=aliases.contracts,
    ).build(
        coverage,
        source_method_coverage_hash=method_coverage_file_hash(PRODUCTION_COVERAGE),
        source_taxonomy_hash="all-83-readiness-asha-acceptance",
    )
    recipes = RecipeRegistry.from_paths(
        [
            ResourcePaths.RECIPE_BUNDLES,
            *sorted(ResourcePaths.RECIPES_DIR.glob("*.yaml")),
        ],
        strict=False,
    )
    return PaperExecutionInventoryBuilder().build(
        coverage,
        executable,
        PaperRegistry("research").list(),
        recipes.list(),
        expected_compatible_count=83,
    )


@pytest.fixture(scope="module")
def production_requirements() -> PaperExecutionRequirementsMatrix:
    return PaperExecutionRequirementsMatrix.from_yaml(REQUIREMENTS_PATH)


@pytest.fixture(scope="module")
def production_readiness() -> PaperReadinessReport:
    return PaperReadinessReport.from_yaml(READINESS_PATH)


def test_production_inventory_requirements_and_readiness_have_same_83_papers(
    production_inventory,
    production_requirements: PaperExecutionRequirementsMatrix,
    production_readiness: PaperReadinessReport,
) -> None:  # type: ignore[no-untyped-def]
    inventory_ids = {item.paper_id for item in production_inventory.records}
    requirement_ids = {item.paper_id for item in production_requirements.requirements}
    readiness_ids = {item.paper_id for item in production_readiness.records}
    assert production_inventory.compatible_paper_count == 83
    assert production_requirements.compatible_paper_count == 83
    assert production_readiness.paper_count == 83
    assert inventory_ids == requirement_ids == readiness_ids
    assert len(inventory_ids) == 83

    allowed_states = {
        "inventory_seen",
        "contract_ready",
        "cpu_ready",
        "runtime_ready",
        "asha_eligible",
        "pre_registered",
        "blocked",
        "incompatible",
    }
    generic = {
        "distillation.yolo26_teacher_student",
        "domain_adaptation.general",
        "quality_alignment.general",
    }
    inventory_by_id = {item.paper_id: item for item in production_inventory.records}
    requirement_by_id = {
        item.paper_id: item for item in production_requirements.requirements
    }
    for readiness in production_readiness.records:
        inventory = inventory_by_id[readiness.paper_id]
        requirement = requirement_by_id[readiness.paper_id]
        assert readiness.readiness_state in allowed_states
        assert inventory.paper_mechanism_resolutions
        assert inventory.paper_specific_mechanism_ids or any(
            not resolution.resolved
            and resolution.unresolved_reason
            for resolution in inventory.paper_mechanism_resolutions
        )
        assert requirement.paper_specific_mechanism not in generic
        assert readiness.cpu_checks_passed is not None
        assert readiness.runtime_checks_passed is not None
        assert readiness.exact_blocker or readiness.asha_eligibility
        if readiness.asha_eligibility:
            assert readiness.readiness_state == "asha_eligible"
            assert readiness.cpu_checks_passed
            assert readiness.runtime_checks_passed
            assert readiness.matched_control_readiness.passed
            assert requirement.training_candidate_allowed


def test_production_readiness_is_cpu_only_and_has_no_training_claim(
    production_readiness: PaperReadinessReport,
) -> None:
    assert production_readiness.cpu_only is True
    assert production_readiness.training_started is False
    assert production_readiness.accuracy_claim == "none"
    assert production_readiness.gpu_probe == "not_run"
    assert not any(item.asha_eligibility for item in production_readiness.records)
    assert all(
        item.final_disposition != "runtime_ready" or item.asha_eligibility
        for item in production_readiness.records
    )


def test_cpu_ready_production_records_reach_runtime_gate_fields(
    production_readiness: PaperReadinessReport,
) -> None:
    cpu_ready = [
        item
        for item in production_readiness.records
        if item.cpu_checks_passed
    ]
    assert cpu_ready
    # A CPU-ready row is not silently dropped: every row has the complete
    # runtime decision tuple, even when a later protocol/control gate blocks it.
    assert all(
        item.runtime_checks_passed is not None
        and item.dataset_evidence_result is not None
        and item.teacher_evidence_result is not None
        and item.graph_evidence_result is not None
        and item.readiness_state
        in {
            "cpu_ready",
            "runtime_ready",
            "asha_eligible",
            "blocked",
            "pre_registered",
            "incompatible",
        }
        for item in cpu_ready
    )


def test_mock_ready_fingerprints_reach_plan_and_asha_without_training(
    tmp_path: Path,
    production_inventory,
) -> None:  # type: ignore[no-untyped-def]
    groups = _mock_ready_records(production_inventory)
    assert groups
    nodes = [_mock_node(tmp_path, group, index) for index, group in enumerate(groups)]
    baseline = _mock_baseline(tmp_path)
    plan = build_round_execution_plan(
        run_id="all-83-readiness-mock",
        nodes=nodes[:2],
        deferred_candidate_nodes=nodes[2:],
        baseline_control_node=baseline,
        ranks={node.candidate_config.candidate_id: index for index, node in enumerate(nodes)},
        primary_metric="map50_95",
    )
    planned_nodes = {
        item.source_node_id
        for item in plan.assignments
        if item.role == "candidate"
    } | {
        item.node_id
        for item in plan.deferred_nodes
        if item.candidate_config.action_domain == "paper"
    }
    assert planned_nodes == {node.node_id for node in nodes}

    scheduler = ASHAScheduler.create("all-83-readiness-mock")
    for node, group in zip(nodes, groups):
        scheduler.register_trial(
            trial_id=f"mock:{node.candidate_config.candidate_id}",
            candidate_id=node.candidate_config.candidate_id,
            source_run_id="all-83-readiness-mock",
            source_node=node,
            baseline_control_node=baseline,
            paper_ids=[item.paper_id for item in group],
            method_profile_ids=[item.profile_id for item in group],
            mechanism_ids=[
                mechanism
                for item in group
                for mechanism in item.paper_specific_mechanism_ids
            ],
        )
    eligible_fingerprints = {execution_fingerprint(node) for node in nodes}
    assert len(scheduler.study.trials) == len(eligible_fingerprints)
    assert {
        trial.execution_fingerprint for trial in scheduler.study.trials
    } == eligible_fingerprints
    assert all(
        trial.readiness_state == "asha_eligible"
        and trial.baseline_control_node is not None
        for trial in scheduler.study.trials
    )


def test_blockers_cannot_enter_training_asha(
    production_requirements: PaperExecutionRequirementsMatrix,
) -> None:
    registry = default_paper_protocol_registry()
    domain_requirement = next(
        item
        for item in production_requirements.requirements
        if item.required_domain_assets
    )
    domain = registry.evaluate(
        domain_requirement.paper_id,
        PaperProtocolContext(),
    )
    assert domain.allows_asha_registration is False
    assert domain.disposition in {"evidence_recovery", "blocked_runtime"}
    assert "domain_adaptation_blocked_from_coco_map_training" in domain.reason_codes

    distillation_requirement = next(
        item
        for item in production_requirements.requirements
        if item.required_teacher_assets
    )
    distillation = registry.evaluate(
        distillation_requirement.paper_id,
        PaperProtocolContext(),
    )
    assert distillation.allows_asha_registration is False
    assert "teacher_checkpoint_missing" in distillation.reason_codes

    inference = inference_only_protocol("acceptance:inference-only")
    inference_result = registry.__class__([inference]).evaluate(
        inference.paper_id,
        PaperProtocolContext(asha_track="training"),
    )
    assert inference_result.allows_asha_registration is False
    assert "inference_only_excluded_from_training_asha" in inference_result.reason_codes


def test_protocol_mismatch_has_no_paired_delta() -> None:
    def metric(value: float, *, baseline: bool, protocol: str) -> MetricEvidence:
        return MetricEvidence.model_validate(
            {
                "run_id": "acceptance-paired",
                "origin_run_id": "acceptance-paired",
                "inheritance_depth": 0,
                "candidate_id": "baseline" if baseline else "candidate",
                "node_id": "baseline" if baseline else "candidate",
                "evidence_role": "baseline_reference" if baseline else "current_observation",
                "dataset_manifest_sha256": "dataset",
                "protocol_hash": protocol,
                "subset_manifest_sha256": "subset",
                "seed": 1,
                "epochs": 3,
                "fidelity": "pilot_3",
                "batch_policy_hash": "batch",
                "ultralytics_version": "mock",
                "imgsz": 640,
                "eval_protocol_hash": "eval",
                "split": "val2017",
                "metric_name": "map50_95",
                "value": value,
                "source": "offline_mock",
                "verified": True,
            }
        )

    result = build_paired_experiment_result(
        run_id="acceptance-paired",
        candidate_id="candidate",
        candidate_node_id="candidate",
        metric_records=[
            metric(0.39, baseline=True, protocol="old"),
            metric(0.40, baseline=False, protocol="current"),
        ],
        error_facts=[],
    )
    assert result.verified is False
    assert result.metric_deltas == {}
    assert "protocol_hash_mismatch" in result.blockers


def test_pre_registered_deferred_identity_recovers_without_assignment(
    tmp_path: Path,
    production_inventory,
) -> None:  # type: ignore[no-untyped-def]
    group = _mock_ready_records(production_inventory)[0]
    candidate = _mock_node(tmp_path, group, 0)
    baseline = _mock_baseline(tmp_path)
    scheduler = ASHAScheduler.create("deferred-recovery")
    reserved = scheduler.pre_register_trial(
        trial_id="deferred:mock_paper_0",
        candidate_id=candidate.candidate_config.candidate_id,
        source_run_id="deferred-recovery",
        source_node=candidate,
        baseline_control_node=baseline,
        paper_ids=[item.paper_id for item in group],
        blockers=["deferred_budget"],
    )
    assert reserved.readiness_state == "pre_registered"
    assert reserved.status == "needs_evidence"
    assert scheduler.next_assignment() is None
    active = scheduler.register_trial(
        trial_id=reserved.trial_id,
        candidate_id=candidate.candidate_config.candidate_id,
        source_run_id="deferred-recovery",
        source_node=candidate,
        baseline_control_node=baseline,
        target_error_facts=candidate.candidate_config.target_error_facts,
        paper_ids=[item.paper_id for item in group],
        readiness_state="asha_eligible",
        readiness_blockers=[],
    )
    assert active is reserved
    assert active.readiness_state == "asha_eligible"
    assert scheduler.next_assignment() is not None


def test_candidate_failure_does_not_remove_other_eligible_trials(
    tmp_path: Path,
    production_inventory,
) -> None:  # type: ignore[no-untyped-def]
    groups = _mock_ready_records(production_inventory)[:2]
    nodes = [_mock_node(tmp_path, group, index) for index, group in enumerate(groups)]
    baseline = _mock_baseline(tmp_path)
    scheduler = ASHAScheduler.create("failure-isolation-83")
    for node, group in zip(nodes, groups):
        scheduler.register_trial(
            trial_id=f"failure:{node.candidate_config.candidate_id}",
            candidate_id=node.candidate_config.candidate_id,
            source_run_id="failure-isolation-83",
            source_node=node,
            baseline_control_node=baseline,
            paper_ids=[item.paper_id for item in group],
        )
    failed = scheduler.study.trials[0]
    scheduler.report(
        failed.trial_id,
        ASHAObservation(
            stage_id="pilot_3",
            node_id=nodes[0].node_id,
            seed=1,
            failure_reason="mock_candidate_failed",
            evidence_complete=False,
        ),
    )
    assert scheduler.study.trial(failed.trial_id).status == "failed"
    assert len(scheduler.study.trials) == 2
    assert scheduler.study.trials[1].status == "waiting"
