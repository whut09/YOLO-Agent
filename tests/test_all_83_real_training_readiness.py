"""Production-only acceptance for the final 83-paper training gate.

This suite reads persisted production artifacts.  The small scheduler checks
reuse explicitly labelled mock nodes only to verify state-machine invariants;
they never contribute to production readiness or training counts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from yolo_agent.agents.asha_scheduler import ASHAObservation, ASHAScheduler
from yolo_agent.certification.paper_readiness import PaperReadinessReport
from yolo_agent.core.experiment_graph import MetricEvidence
from yolo_agent.core.execution_fingerprint import execution_fingerprint
from yolo_agent.core.paired_experiment import build_paired_experiment_result
from yolo_agent.core.paper_training_readiness import PaperTrainingReadinessReport
from yolo_agent.research.paper_asset_schemas import PaperAssetRegistry
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_execution_schemas import PaperExecutionInventory

from tests.test_all_83_paper_execution_acceptance import (
    _mock_baseline,
    _mock_node,
    _mock_ready_records,
)


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "runs" / "coverage-audit" / "paper_execution_inventory.yaml"
REQUIREMENTS_PATH = (
    ROOT / "runs" / "coverage-audit" / "paper_execution_requirements.yaml"
)
ASSETS_PATH = ROOT / "runs" / "paper-readiness" / "paper_asset_registry.yaml"
READINESS_PATH = ROOT / "runs" / "paper-readiness" / "paper_readiness_report.yaml"
FINAL_PATH = ROOT / "runs" / "paper-readiness" / "paper_training_readiness.yaml"


def _production_artifacts() -> tuple[
    PaperExecutionInventory,
    PaperExecutionRequirementsMatrix,
    PaperAssetRegistry,
    PaperReadinessReport,
    PaperTrainingReadinessReport,
]:
    return (
        PaperExecutionInventory.from_yaml(INVENTORY_PATH),
        PaperExecutionRequirementsMatrix.from_yaml(REQUIREMENTS_PATH),
        PaperAssetRegistry.from_yaml(ASSETS_PATH),
        PaperReadinessReport.from_yaml(READINESS_PATH),
        PaperTrainingReadinessReport.from_yaml(FINAL_PATH),
    )


def test_all_83_papers_have_inventory_requirements_assets_and_readiness() -> None:
    inventory, requirements, assets, readiness, final = _production_artifacts()
    inventory_ids = {item.paper_id for item in inventory.records}
    assert inventory.compatible_paper_count == 83
    assert requirements.compatible_paper_count == 83
    assert assets.compatible_paper_count == 83
    assert readiness.paper_count == 83
    assert final.inventory_count == 83
    assert len(inventory_ids) == 83
    assert inventory_ids == {
        item.paper_id for item in requirements.requirements
    }
    assert inventory_ids == {item.paper_id for item in assets.records}
    assert inventory_ids == {item.paper_id for item in readiness.records}
    assert inventory_ids == {item.paper_id for item in final.records}


def test_every_paper_has_specific_mechanism_or_explicit_unresolved_reason() -> None:
    inventory, requirements, _, readiness, _ = _production_artifacts()
    requirement_by_id = {item.paper_id: item for item in requirements.requirements}
    for item in inventory.records:
        assert item.paper_mechanism_resolutions
        assert item.paper_specific_mechanism_ids or any(
            not resolution.resolved and resolution.unresolved_reason
            for resolution in item.paper_mechanism_resolutions
        )
        requirement = requirement_by_id[item.paper_id]
        assert requirement.paper_specific_mechanism
        assert requirement.paper_specific_mechanism not in {
            "distillation.yolo26_teacher_student",
            "domain_adaptation.general",
            "quality_alignment.general",
        }
    assert all(record.paper_id for record in readiness.records)


def test_production_assets_have_real_disposition_and_no_mock_authorization() -> None:
    inventory, _, assets, _, final = _production_artifacts()
    assert all(record.availability in {"available", "unavailable"} for record in assets.records)
    assert all(
        record.availability == "available" or record.exact_blocker
        for record in assets.records
    )
    assert all(
        record.availability == "unavailable" for record in assets.records
    )
    assert final.actual_trained_count == 0
    assert final.exact_reproduction_count == inventory.exact_reproduction_candidates == 0
    assert final.training_allowed is False
    assert final.training_cohort_fingerprints == []


def test_cpu_ready_is_not_runtime_ready_or_asha_eligible() -> None:
    _, _, _, readiness, final = _production_artifacts()
    cpu_ready = [item for item in readiness.records if item.cpu_checks_passed]
    assert cpu_ready
    assert any(
        item.cpu_checks_passed and not item.runtime_checks_passed
        for item in readiness.records
    )
    assert final.cpu_ready_count == len(cpu_ready)
    assert final.runtime_ready_count == 0
    assert final.asha_eligible_count == 0
    assert all(not item.asha_eligibility for item in readiness.records)


def test_asset_specific_blockers_cannot_become_eligible() -> None:
    _, requirements, assets, readiness, final = _production_artifacts()
    assets_by_id = {item.paper_id: item for item in assets.records}
    readiness_by_id = {item.paper_id: item for item in readiness.records}
    for requirement in requirements.requirements:
        asset = assets_by_id[requirement.paper_id]
        preflight = readiness_by_id[requirement.paper_id]
        mechanisms = set(requirement.paper_specific_mechanism_ids)
        if requirement.required_teacher_assets:
            assert not asset.teacher_checkpoint or not asset.teacher_sha256
            assert not preflight.asha_eligibility
        if requirement.required_domain_assets:
            assert not asset.source_dataset_manifest or not asset.target_dataset_manifest
            assert not preflight.asha_eligibility
        if requirement.required_manifest_assets:
            assert not asset.hard_negative_manifest
            assert not preflight.asha_eligibility
        if requirement.execution_route == "inference":
            assert not preflight.asha_eligibility
            assert not any(
                record.paper_id == requirement.paper_id
                and record.asha_eligibility
                for record in final.records
            )
        if any("distillation" in mechanism for mechanism in mechanisms):
            assert not asset.teacher_checkpoint
            assert not preflight.asha_eligibility
    assert final.inference_only_count == 1


def test_missing_matched_baselines_and_protocols_cannot_create_delta() -> None:
    _, _, assets, readiness, final = _production_artifacts()
    readiness_by_id = {item.paper_id: item for item in readiness.records}
    assert final.matched_control_ready_count == 0
    assert all(not item.matched_control_readiness.passed for item in readiness.records)
    assert all(
        "matched_baseline_artifact_missing" in record.exact_blocker
        or not readiness_by_id[record.paper_id].asha_eligibility
        for record in assets.records
    )
    assert final.actual_trained_count == 0


def _metric(
    *, baseline: bool, protocol: str = "protocol", split: str = "val2017", imgsz: int = 640
) -> MetricEvidence:
    return MetricEvidence.model_validate(
        {
            "run_id": "real-readiness-pair",
            "origin_run_id": "real-readiness-pair",
            "inheritance_depth": 0,
            "candidate_id": "baseline" if baseline else "candidate",
            "node_id": "baseline" if baseline else "candidate",
            "evidence_role": (
                "baseline_reference" if baseline else "current_observation"
            ),
            "dataset_manifest_sha256": "dataset",
            "protocol_hash": protocol,
            "subset_manifest_sha256": "subset",
            "seed": 1,
            "epochs": 3,
            "fidelity": "pilot_3",
            "batch_policy_hash": "batch",
            "ultralytics_version": "production-audit",
            "imgsz": imgsz,
            "eval_protocol_hash": "eval",
            "split": split,
            "metric_name": "map50_95",
            "value": 0.39 if baseline else 0.40,
            "source": "production_artifact_audit",
            "verified": True,
        }
    )


@pytest.mark.parametrize(
    ("baseline_overrides", "expected_blocker"),
    [
        ({"protocol": "old"}, "protocol_hash_mismatch"),
        ({"split": "test2017"}, "split_mismatch"),
        ({"imgsz": 1280}, "baseline_missing_imgsz_not_fixed_640"),
    ],
)
def test_paired_delta_rejects_protocol_split_or_imgsz_mismatch(
    baseline_overrides: dict[str, object], expected_blocker: str
) -> None:
    result = build_paired_experiment_result(
        run_id="real-readiness-pair",
        candidate_id="candidate",
        candidate_node_id="candidate",
        metric_records=[
            _metric(baseline=True, **baseline_overrides),
            _metric(baseline=False),
        ],
        error_facts=[],
    )
    assert result.metric_deltas == {}
    assert result.verified is False
    assert expected_blocker in result.blockers


def test_missing_baseline_produces_no_paired_delta() -> None:
    result = build_paired_experiment_result(
        run_id="real-readiness-pair",
        candidate_id="candidate",
        candidate_node_id="candidate",
        metric_records=[_metric(baseline=False)],
        error_facts=[],
    )
    assert result.metric_deltas == {}
    assert result.verified is False


def test_blocked_papers_retain_pre_registration_identity() -> None:
    _, _, _, readiness, final = _production_artifacts()
    blocked = [
        item
        for item in readiness.records
        if item.final_disposition in {"blocked_runtime", "evidence_recovery", "incompatible"}
    ]
    assert len(blocked) == 83
    assert all(item.pre_registered for item in blocked)
    assert final.pre_registered_count == 0
    assert final.asha_eligible_count == 0


def test_execution_fingerprint_merges_provenance_without_collapsing_identity(
    tmp_path: Path,
) -> None:
    inventory, _, _, _, _ = _production_artifacts()
    groups: dict[str, list[object]] = defaultdict(list)
    for item in inventory.records:
        groups[item.execution_fingerprint].append(item)
    assert len(groups) < len(inventory.records)
    assert any(len(items) > 1 for items in groups.values())
    ready_groups = _mock_ready_records(inventory)
    nodes = [_mock_node(tmp_path, group, index) for index, group in enumerate(ready_groups)]
    baseline = _mock_baseline(tmp_path)
    scheduler = ASHAScheduler.create("real-readiness-fingerprint-check")
    for node, group in zip(nodes, ready_groups):
        scheduler.register_trial(
            trial_id=f"fingerprint:{node.candidate_config.candidate_id}",
            candidate_id=node.candidate_config.candidate_id,
            source_run_id="real-readiness-mock-routing",
            source_node=node,
            baseline_control_node=baseline,
            paper_ids=[item.paper_id for item in group],
        )
    node_fingerprints = {execution_fingerprint(node) for node in nodes}
    assert len(scheduler.study.trials) == len(node_fingerprints)
    assert len(
        {trial.execution_fingerprint for trial in scheduler.study.trials}
    ) == len(node_fingerprints)
    assert all(
        len(trial.paper_ids) >= 1 for trial in scheduler.study.trials
    )
    assert any(len(trial.paper_ids) > 1 for trial in scheduler.study.trials)


def test_deferred_identity_recovers_and_failure_is_isolated(tmp_path: Path) -> None:
    inventory, _, _, _, _ = _production_artifacts()
    groups = _mock_ready_records(inventory)[:2]
    nodes = [_mock_node(tmp_path, group, index) for index, group in enumerate(groups)]
    baseline = _mock_baseline(tmp_path)
    scheduler = ASHAScheduler.create("real-readiness-state-check")
    deferred = scheduler.pre_register_trial(
        trial_id="deferred:paper",
        candidate_id=nodes[0].candidate_config.candidate_id,
        source_run_id="real-readiness-mock-routing",
        source_node=nodes[0],
        baseline_control_node=baseline,
        paper_ids=[item.paper_id for item in groups[0]],
        blockers=["deferred_budget"],
    )
    assert deferred.readiness_state == "pre_registered"
    assert scheduler.next_assignment() is None
    scheduler.register_trial(
        trial_id=deferred.trial_id,
        candidate_id=nodes[0].candidate_config.candidate_id,
        source_run_id="real-readiness-mock-routing",
        source_node=nodes[0],
        baseline_control_node=baseline,
        paper_ids=[item.paper_id for item in groups[0]],
        readiness_state="asha_eligible",
        readiness_blockers=[],
    )
    assert scheduler.next_assignment() is not None
    scheduler.register_trial(
        trial_id="failure:paper",
        candidate_id=nodes[1].candidate_config.candidate_id,
        source_run_id="real-readiness-mock-routing",
        source_node=nodes[1],
        baseline_control_node=baseline,
        paper_ids=[item.paper_id for item in groups[1]],
    )
    scheduler.report(
        "failure:paper",
        ASHAObservation(
            stage_id="pilot_3",
            node_id=nodes[1].node_id,
            seed=1,
            failure_reason="mock_candidate_failed",
            evidence_complete=False,
        ),
    )
    assert len(scheduler.study.trials) == 2
    assert scheduler.study.trial("deferred:paper").status in {"running", "waiting"}
