"""Offline acceptance for all compatible paper execution routes.

This suite deliberately uses the frozen production coverage plus CPU-only
mock execution.  A mock-ready candidate proves routing, not model quality or
actual training.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from yolo_agent.agents.asha_scheduler import ASHAObservation, ASHAScheduler
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.paper_recipe_planner import _missing_recipe_evidence
from yolo_agent.components.contracts import load_contracts
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_fingerprint import execution_fingerprint
from yolo_agent.core.experiment_graph import ExperimentNode, MetricEvidence
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
from yolo_agent.research.paper_protocol_catalog import (
    inference_only_protocol,
)
from yolo_agent.research.paper_protocol_contract import (
    PaperProtocolContext,
    default_paper_protocol_registry,
)
from yolo_agent.research.paper_registry import PaperRegistry
from yolo_agent.resources import ResourcePaths


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_COVERAGE = ROOT / "research" / "production" / "paper_method_coverage.yaml"
PROTOCOL_HASH = "mock-paper-protocol-640"
DATASET_HASH = "mock-coco-manifest"


@pytest.fixture(scope="module")
def production_inventory():  # type: ignore[no-untyped-def]
    method_coverage = PaperMethodCoverageReport.from_yaml(PRODUCTION_COVERAGE)
    aliases = ComponentAliasResolver.from_yaml()
    executable = ExecutablePaperCoverageAuditor(
        contracts=aliases.contracts,
    ).build(
        method_coverage,
        source_method_coverage_hash=method_coverage_file_hash(PRODUCTION_COVERAGE),
        source_taxonomy_hash="offline-acceptance-taxonomy",
    )
    recipes = RecipeRegistry.from_paths(
        [
            ResourcePaths.RECIPE_BUNDLES,
            *sorted(ResourcePaths.RECIPES_DIR.glob("*.yaml")),
        ],
        strict=False,
    )
    return PaperExecutionInventoryBuilder().build(
        method_coverage,
        executable,
        PaperRegistry("research").list(),
        recipes.list(),
        expected_compatible_count=83,
    )


def test_all_83_papers_have_unique_inventory_specs(production_inventory) -> None:  # type: ignore[no-untyped-def]
    records = production_inventory.records
    assert production_inventory.compatible_paper_count == 83
    assert len(records) == 83
    assert len({item.paper_id for item in records}) == 83
    assert len({item.profile_id for item in records}) == 83
    assert all(item.current_disposition for item in records)
    assert production_inventory.exact_reproduction_candidates == 0


def test_every_paper_has_specific_resolution_or_explicit_unresolved_reason(
    production_inventory,
) -> None:  # type: ignore[no-untyped-def]
    allowed_dispositions = {
        "queued",
        "runtime_ready",
        "already_tested",
        "evidence_recovery",
        "implementation_request",
        "incompatible",
        "blocked_runtime",
        "deferred_budget",
    }
    for record in production_inventory.records:
        assert record.current_disposition in allowed_dispositions
        assert record.paper_mechanism_resolutions
        assert record.recipe_ids or record.current_disposition in {
            "implementation_request",
            "evidence_recovery",
            "incompatible",
            "blocked_runtime",
            "deferred_budget",
        }
        for resolution in record.paper_mechanism_resolutions:
            assert resolution.resolved or resolution.unresolved_reason
            if not resolution.resolved:
                assert not resolution.executable_candidate
            assert not resolution.paper_specific_mechanism_id.endswith(".general") if resolution.paper_specific_mechanism_id else True


def test_generic_domain_and_distillation_do_not_replace_paper_specific_methods(
    production_inventory,
) -> None:  # type: ignore[no-untyped-def]
    domain = [
        item
        for item in production_inventory.records
        if "domain_adaptation.general" in item.generic_component_ids
    ]
    distillation = [
        item
        for item in production_inventory.records
        if "distillation.yolo26_teacher_student" in item.generic_component_ids
    ]
    assert len(domain) == 40
    assert len(distillation) == 32
    for records in (domain, distillation):
        assert all(
            generic not in item.paper_specific_mechanism_ids
            for item in records
            for generic in item.generic_component_ids
        )
        assert all(
            item.paper_specific_mechanism_ids
            or any(not resolution.resolved for resolution in item.paper_mechanism_resolutions)
            for item in records
        )


def _mock_ready_records(production_inventory):  # type: ignore[no-untyped-def]
    excluded_prefixes = ("domain_adaptation.", "distillation.")
    records = [
        item
        for item in production_inventory.records
        if item.recipe_ids
        and item.paper_specific_mechanism_ids
        and len(item.canonical_component_ids) == 1
        and item.canonical_component_ids[0] != "inference.sahi_slicing"
        and not any(
            item.canonical_component_ids[0].startswith(prefix)
            for prefix in excluded_prefixes
        )
    ]
    grouped: dict[str, list[object]] = defaultdict(list)
    for record in records:
        grouped[record.execution_fingerprint].append(record)
    return sorted(grouped.values(), key=lambda group: group[0].paper_id)


def _mock_node(tmp_path: Path, records: list[object], index: int) -> ExperimentNode:
    first = records[0]
    paper_ids = [item.paper_id for item in records]
    recipe_id = first.recipe_ids[0]
    candidate_id = f"mock_paper_{index}_{first.execution_fingerprint[:8]}"
    metadata = {
        "paper_ids": ",".join(paper_ids),
        "method_profile_ids": ",".join(item.profile_id for item in records),
        "adapter_runtime_entrypoint": "mock.paper.adapter",
        "component_recipe_id": recipe_id,
        "component_recipe_version": "mock.v1",
        "run_protocol_hash": PROTOCOL_HASH,
        "baseline_protocol_hash": PROTOCOL_HASH,
        "dataset_manifest_sha256": DATASET_HASH,
        "fidelity": "pilot_3",
        "split": "val2017",
        "imgsz": 640,
        "mock_runtime_ready": True,
        "paper_readiness_state": "asha_eligible",
        "paper_readiness_blockers": "[]",
    }
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        project=tmp_path / "ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=640,
        batch=2,
        seed=1,
        metadata=metadata,
    )
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            action_domain="paper",
            action_id=recipe_id,
            search_tier="method",
            components=list(first.canonical_component_ids),
            target_error_facts=[
                {"fact_type": "localization_error", "subject": "overall"}
            ],
        ),
        data_version=DATASET_HASH,
        seed=1,
        command=command.display(),
        command_spec=command,
        changed_variables={f"paper.{recipe_id}.enabled": True},
    )


def _mock_baseline(tmp_path: Path) -> ExperimentNode:
    metadata = {
        "matched_baseline_control": True,
        "run_protocol_hash": PROTOCOL_HASH,
        "baseline_protocol_hash": PROTOCOL_HASH,
        "dataset_manifest_sha256": DATASET_HASH,
        "fidelity": "pilot_3",
        "split": "val2017",
        "imgsz": 640,
    }
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        project=tmp_path / "ultralytics",
        name="matched_baseline",
        epochs=3,
        imgsz=640,
        batch=2,
        seed=1,
        metadata=metadata,
    )
    return ExperimentNode(
        node_id="node_matched_baseline",
        candidate_config=CandidateConfig(
            candidate_id="matched_baseline",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            action_domain="baseline",
            action_id="baseline",
            search_tier="method",
        ),
        data_version=DATASET_HASH,
        seed=1,
        command=command.display(),
        command_spec=command,
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
        run_id="all-83-offline-mock",
        nodes=nodes[:2],
        deferred_candidate_nodes=nodes[2:],
        baseline_control_node=baseline,
        ranks={node.candidate_config.candidate_id: index for index, node in enumerate(nodes)},
        primary_metric="map50_95",
    )
    planned_source_ids = {
        item.source_node_id
        for item in plan.assignments
        if item.role == "candidate"
    } | {
        item.node_id
        for item in plan.deferred_nodes
        if item.candidate_config.action_domain == "paper"
    }
    assert planned_source_ids == {node.node_id for node in nodes}
    assert len(plan.execution_nodes) == 3

    scheduler = ASHAScheduler.create("all-83-offline-mock")
    for node, group in zip(nodes, groups):
        scheduler.register_trial(
            trial_id=f"mock:{node.candidate_config.candidate_id}",
            candidate_id=node.candidate_config.candidate_id,
            source_run_id="all-83-offline-mock",
            source_node=node,
            baseline_control_node=baseline,
            paper_ids=[item.paper_id for item in group],
            method_profile_ids=[item.profile_id for item in group],
            mechanism_ids=[
                mechanism
                for item in group
                for mechanism in item.paper_specific_mechanism_ids
            ],
            required_evidence=[
                evidence
                for item in group
                for evidence in item.required_evidence
            ],
        )
    # ASHA is keyed by execution identity: distinct paper provenance may share
    # one trial, while every unique mock execution fingerprint must remain.
    assert len(scheduler.study.trials) == len(
        {execution_fingerprint(node) for node in nodes}
    )
    assert {
        trial.execution_fingerprint
        for trial in scheduler.study.trials
    } == {execution_fingerprint(node) for node in nodes}
    assert all(trial.baseline_control_node is not None for trial in scheduler.study.trials)


def test_coupled_candidate_keeps_four_arms_and_matched_controls() -> None:
    from yolo_agent.recipes.coupled_library import (
        CouplingEvidence,
        EvidenceBoundCoupledRecipeLibrary,
    )

    components = [
        "neck.rtmdet_large_kernel",
        "loss.quality.correlation",
    ]
    result = EvidenceBoundCoupledRecipeLibrary().materialize(
        component_ids=components,
        evidence=CouplingEvidence(
            evidence_kind="local_diagnosis",
            source_id="offline:coupling",
            component_ids=components,
            reason="verified localization interaction",
            source_locations=["offline fixture"],
            paper_ids=["paper:coupled"],
            error_fact_ids=["fact:localization"],
            error_fact_types=["localization_error"],
            mechanism_ids=["rtmdet_large_kernel", "quality_correlation"],
            paper_specific_configuration={
                item: {"changed_variable": f"{item}.weight"}
                for item in components
            },
            required_evidence=["verified_coupling_diagnosis"],
            verified=True,
        ),
    )
    assert result.decision == "materialized"
    assert result.recipe is not None
    assert [item["arm_id"] for item in result.recipe.internal_ablation_plan] == [
        "baseline",
        "arm_A",
        "arm_B",
        "arm_A_plus_B",
    ]
    assert all(
        item.get("matched_control_arm_id") == "baseline"
        for item in result.recipe.internal_ablation_plan[1:]
    )
    assert result.recipe.paper_ids == ["paper:coupled"]
    assert result.recipe.combination_fingerprint


def test_inference_domain_distillation_and_hard_negative_gates_fail_closed(
) -> None:
    inference_contract = inference_only_protocol("fixture:inference-only")
    inference = default_paper_protocol_registry().__class__(
        [inference_contract]
    ).evaluate(
        inference_contract.paper_id,
        PaperProtocolContext(asha_track="training"),
    )
    assert inference.execution_class == "inference_candidate"
    assert not inference.allows_asha_registration
    assert "inference_only_excluded_from_training_asha" in inference.reason_codes

    registry = default_paper_protocol_registry()
    domain_id = next(
        paper_id
        for paper_id in registry.paper_ids
        if registry.require(paper_id).protocol_family == "domain_adaptation"
    )
    domain = registry.evaluate(domain_id, PaperProtocolContext())
    assert not domain.allows_asha_registration
    assert "domain_adaptation_blocked_from_coco_map_training" in domain.reason_codes
    assert domain.disposition == "evidence_recovery"

    distillation_id = next(
        paper_id
        for paper_id in registry.paper_ids
        if registry.require(paper_id).protocol_family == "distillation"
    )
    distillation = registry.evaluate(distillation_id, PaperProtocolContext())
    assert not distillation.allows_asha_registration
    assert "teacher_checkpoint_missing" in distillation.reason_codes
    assert distillation.disposition == "evidence_recovery"

    contracts = load_contracts(
        "configs/components/data_pipeline/paper_data_adapters.yaml"
    )
    recipes = RecipeRegistry.from_path(
        "configs/recipes/yolo26_data_pipeline.yaml",
        component_contracts=contracts,
    )
    replay = recipes.get("yolo26_hard_negative_replay")
    assert replay is not None
    recovery = _missing_recipe_evidence(replay, [], None, {})
    assert recovery
    assert "recover_train_hard_negative_evidence" in recovery
    assert "bind_train_dataset_manifest_hash" in recovery


def _metric(
    value: float,
    *,
    baseline: bool,
    protocol_hash: str = PROTOCOL_HASH,
) -> MetricEvidence:
    return MetricEvidence.model_validate(
        {
            "run_id": "paired-offline",
            "origin_run_id": "paired-offline",
            "inheritance_depth": 0,
            "candidate_id": "baseline" if baseline else "candidate",
            "node_id": "node_baseline" if baseline else "node_candidate",
            "evidence_role": "baseline_reference" if baseline else "current_observation",
            "dataset_manifest_sha256": DATASET_HASH,
            "protocol_hash": protocol_hash,
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


def test_protocol_or_split_mismatch_produces_no_paired_delta() -> None:
    result = build_paired_experiment_result(
        run_id="paired-offline",
        candidate_id="candidate",
        candidate_node_id="node_candidate",
        metric_records=[
            _metric(0.39, baseline=True, protocol_hash="old-protocol"),
            _metric(0.40, baseline=False),
        ],
        error_facts=[],
    )
    assert result.verified is False
    assert result.metric_deltas == {}
    assert "protocol_hash_mismatch" in result.blockers


def test_candidate_failure_isolated_from_other_paper_trials(
    tmp_path: Path,
    production_inventory,
) -> None:  # type: ignore[no-untyped-def]
    groups = _mock_ready_records(production_inventory)[:2]
    nodes = [_mock_node(tmp_path, group, index) for index, group in enumerate(groups)]
    baseline = _mock_baseline(tmp_path)
    scheduler = ASHAScheduler.create("failure-isolation")
    for node, group in zip(nodes, groups):
        scheduler.register_trial(
            trial_id=f"failure:{node.candidate_config.candidate_id}",
            candidate_id=node.candidate_config.candidate_id,
            source_run_id="failure-isolation",
            source_node=node,
            baseline_control_node=baseline,
            paper_ids=[item.paper_id for item in group],
        )
    failed_id = scheduler.study.trials[0].trial_id
    scheduler.report(
        failed_id,
        ASHAObservation(
            stage_id="pilot_3",
            node_id=nodes[0].node_id,
            seed=1,
            failure_reason="mock_candidate_failed",
            evidence_complete=False,
        ),
    )
    assert scheduler.study.trial(failed_id).status == "failed"
    assert len(scheduler.study.trials) == 2
    assert scheduler.study.trials[1].status == "waiting"


def test_inventory_has_no_silent_drop_or_actual_training_claim(
    production_inventory,
) -> None:  # type: ignore[no-untyped-def]
    assert production_inventory.disposition_counts == {
        "queued": 0,
        "runtime_ready": 0,
        "already_tested": 0,
        "evidence_recovery": 0,
        "implementation_request": 68,
        "incompatible": 0,
        "blocked_runtime": 15,
        "deferred_budget": 0,
    }
    assert all(not item.exact_reproduction_possible for item in production_inventory.records)
    # Inventory/readiness artifacts contain no actual candidate metric.  A
    # paired result is the only accepted evidence for actual trained coverage.
    assert not any(
        "trained" in item.disposition_reason.lower()
        for item in production_inventory.records
    )
