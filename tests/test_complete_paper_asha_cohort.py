"""CPU mock coverage for registering the complete compatible-paper cohort."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from yolo_agent.agents.asha_scheduler import ASHAScheduler
from yolo_agent.agents.auto_optimization_loop import _register_guarded_pilot_trials
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.orchestrator import LoopOrchestrator
from yolo_agent.agents.paper_proposal_ledger import (
    PaperCandidateCoverage,
    PaperCandidateCoverageLedger,
    planned_recipe_disposition,
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.optimization_objective import OptimizationObjective
from yolo_agent.core.round_execution_plan import build_round_execution_plan
from yolo_agent.core.run_context import RunContext
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)


def _node(tmp_path: Path, index: int, paper_id: str, *, baseline: bool = False) -> ExperimentNode:
    candidate_id = "matched_baseline_control" if baseline else f"paper_candidate_{index}"
    recipe_id = "baseline" if baseline else f"paper_recipe_{index}"
    metadata: dict[str, object] = {
        "matched_baseline_control": baseline,
        "run_protocol_hash": "protocol-640",
        "baseline_protocol_hash": "protocol-640",
        "dataset_manifest_sha256": "dataset-83",
        "fidelity": "pilot_3",
        "split": "val2017",
    }
    if not baseline:
        metadata.update(
            {
                "paper_id": paper_id,
                "method_profile_ids": f"profile:{paper_id}",
                "adapter_runtime_entrypoint": "mock.paper.runtime",
                "component_recipe_id": recipe_id,
                "component_recipe_version": "v1",
                "paper_readiness_state": "asha_eligible",
                "paper_readiness_blockers": "[]",
            }
        )
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        project=tmp_path / "ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=640,
        batch=4,
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
            components=[] if baseline else ["loss.quality.correlation"],
            target_error_facts=[]
            if baseline
            else [{"fact_type": "localization_error", "subject": "overall"}],
        ),
        data_version="dataset-83",
        seed=1,
        command=command.display(),
        command_spec=command,
        changed_variables={}
        if baseline
        else {f"paper.{index}.enabled": True},
    )


def _inventory(paper_ids: list[str]) -> PaperExecutionInventory:
    records = [
        PaperExecutionSpec(
            paper_id=paper_id,
            profile_id=f"profile:{paper_id}",
            title=f"Fixture {paper_id}",
            source_locations=[f"fixture#{paper_id}"],
            canonical_component_ids=["loss.quality.correlation"],
            paper_specific_mechanism_ids=["quality_correlation"],
            required_evidence=["target_error_facts"],
            recipe_ids=[f"paper_recipe_{index}"],
            execution_fingerprint=hashlib.sha256(paper_id.encode()).hexdigest(),
            current_disposition="evidence_recovery",
            disposition_reason="fixture paper candidate",
        )
        for index, paper_id in enumerate(paper_ids)
    ]
    return PaperExecutionInventory(
        source_method_coverage_hash="a" * 64,
        all_paper_count=728,
        compatible_paper_count=len(records),
        exact_reproduction_candidates=0,
        records=sorted(records, key=lambda item: item.paper_id),
    ).with_hash()


def _allow_mock_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: type("Readiness", (), {"allowed": True})(),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: type(
            "Certification",
            (),
            {
                "allowed": True,
                "blockers": [],
                "report_path": None,
                "report_hash": "mock",
            },
        )(),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )


def test_all_83_papers_register_as_mock_asha_trials_without_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paper_ids = [f"paper:{index:03d}" for index in range(83)]
    context = RunContext(
        run_id="asha-83-mock",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
        dataset_version="dataset-83",
        dataset_manifest_sha256="dataset-83",
    )
    child = LoopOrchestrator(context)
    objective = OptimizationObjective(
        goal_description="Improve overall mAP",
        primary_metric="map50_95",
        baseline_run_id=context.run_id,
        baseline_candidate_id="matched_baseline_control",
        baseline_protocol_hash="protocol-640",
    )
    objective_path = context.artifact_path("optimization_objective.yaml")
    objective.to_yaml(objective_path)
    context.metadata["optimization_objective_path"] = objective_path.as_posix()
    ledger = PaperCandidateCoverageLedger(
        context.artifact_path("paper_candidate_coverage.yaml"),
        run_id=context.run_id,
        protocol_hash="protocol-640",
        dataset_manifest_hash="dataset-83",
    )
    inventory = _inventory(paper_ids)
    ledger.seed_inventory(inventory)
    candidates = [_node(tmp_path, index, paper_id) for index, paper_id in enumerate(paper_ids)]
    ledger.upsert_many(
        [
            planned_recipe_disposition(
                run_id=context.run_id,
                round_index=1,
                recipe_id=str(node.command_spec.metadata["component_recipe_id"]),
                recipe_version="v1",
                component_ids=node.candidate_config.components,
                decision="selected",
                reasons=[],
                related_papers=[str(node.command_spec.metadata["paper_id"])],
                method_profile_ids=[
                    str(node.command_spec.metadata["method_profile_ids"])
                ],
                execution_fingerprint=hashlib.sha256(
                    node.candidate_config.candidate_id.encode()
                ).hexdigest(),
                candidate_id=node.candidate_config.candidate_id,
                protocol_hash="protocol-640",
                dataset_manifest_hash="dataset-83",
            )
            for node in candidates
        ]
    )
    baseline = _node(tmp_path, 0, "baseline", baseline=True)
    round_plan = build_round_execution_plan(
        run_id=context.run_id,
        nodes=candidates[:6],
        deferred_candidate_nodes=candidates[6:],
        baseline_control_node=baseline,
        ranks={node.candidate_config.candidate_id: index for index, node in enumerate(candidates)},
        primary_metric="map50_95",
    )
    round_plan.to_yaml(context.artifact_path("round_execution_plan.yaml"))

    _allow_mock_registration(monkeypatch)

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(scheduler, child, candidates)

    coverage = PaperCandidateCoverage.from_yaml(ledger.path)
    assert registered == 83
    assert len(scheduler.study.trials) == 83
    assert all(trial.baseline_control_node is not None for trial in scheduler.study.trials)
    assert coverage.expected_paper_count == 83
    assert len(coverage.current_by_paper) == 83
    assert sum(
        coverage.current_by_paper[paper_id].disposition == "queued"
        for paper_id in paper_ids
    ) == 6
    assert sum(
        coverage.current_by_paper[paper_id].disposition == "deferred_budget"
        for paper_id in paper_ids
    ) == 77
    assert context.metadata["asha_registration_paper_summary"]["asha_trials_registered"] == 83
    assert context.metadata["asha_registration_paper_summary"] == {
        "inventory_count": 83,
        "runtime_ready_count": 83,
        "eligible_count": 83,
        "pre_registered_count": 0,
        "queued_count": 6,
        "deferred_count": 77,
        "blocked_count": 0,
        "evidence_recovery_count": 0,
        "asha_trials_registered": 83,
    }


def test_same_execution_merges_paper_provenance_into_one_asha_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="asha-provenance-merge",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
        dataset_version="dataset-83",
        dataset_manifest_sha256="dataset-83",
    )
    child = LoopOrchestrator(context)
    first = _node(tmp_path, 1, "paper:first")
    second = _node(tmp_path, 2, "paper:second")
    second.changed_variables = dict(first.changed_variables)
    second.candidate_config.action_id = first.candidate_config.action_id
    second.command_spec.metadata["component_recipe_id"] = first.command_spec.metadata[
        "component_recipe_id"
    ]
    baseline = _node(tmp_path, 0, "baseline", baseline=True)
    build_round_execution_plan(
        run_id=context.run_id,
        nodes=[first, second],
        baseline_control_node=baseline,
        primary_metric="map50_95",
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_registration(monkeypatch)

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [first, second],
    )

    assert registered == 1
    assert len(scheduler.study.trials) == 1
    assert scheduler.study.trials[0].paper_ids == ["paper:first", "paper:second"]
    assert scheduler.study.trials[0].method_profile_ids == [
        "profile:paper:first",
        "profile:paper:second",
    ]


def test_same_paper_with_distinct_execution_fingerprints_keeps_both_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="asha-distinct-executions",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
        dataset_version="dataset-83",
        dataset_manifest_sha256="dataset-83",
    )
    child = LoopOrchestrator(context)
    first = _node(tmp_path, 1, "paper:shared")
    second = _node(tmp_path, 2, "paper:shared")
    baseline = _node(tmp_path, 0, "baseline", baseline=True)
    build_round_execution_plan(
        run_id=context.run_id,
        nodes=[first, second],
        baseline_control_node=baseline,
        primary_metric="map50_95",
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_registration(monkeypatch)

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [first, second],
    )

    assert registered == 2
    assert len(scheduler.study.trials) == 2
    assert len({trial.execution_fingerprint for trial in scheduler.study.trials}) == 2
    assert all(trial.paper_ids == ["paper:shared"] for trial in scheduler.study.trials)


def test_missing_target_facts_preserves_paper_as_evidence_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="asha-evidence-recovery",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
        dataset_version="dataset-83",
        dataset_manifest_sha256="dataset-83",
    )
    child = LoopOrchestrator(context)
    candidate = _node(tmp_path, 1, "paper:needs-evidence")
    candidate.candidate_config.target_error_facts = []
    candidate.command_spec.metadata["required_error_fact_types"] = "localization_error"
    baseline = _node(tmp_path, 0, "baseline", baseline=True)
    build_round_execution_plan(
        run_id=context.run_id,
        nodes=[candidate],
        baseline_control_node=baseline,
        primary_metric="map50_95",
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_registration(monkeypatch)

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(scheduler, child, [candidate])

    assert registered == 0
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    record = coverage.records[0]
    assert record.paper_ids == ["paper:needs-evidence"]
    assert record.disposition == "evidence_recovery"
    assert record.reason_codes == [
        "target_error_facts_missing",
        "required_error_fact:localization_error",
    ]
    assert context.metadata["asha_registration_failures_by_paper_id"] == {
        "paper:needs-evidence": 1,
    }


def test_authoritative_plan_registers_cohort_when_caller_list_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="asha-plan-authority",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
        dataset_version="dataset-83",
        dataset_manifest_sha256="dataset-83",
    )
    child = LoopOrchestrator(context)
    candidate = _node(tmp_path, 1, "paper:from-plan")
    baseline = _node(tmp_path, 0, "baseline", baseline=True)
    build_round_execution_plan(
        run_id=context.run_id,
        nodes=[],
        deferred_candidate_nodes=[candidate],
        baseline_control_node=baseline,
        primary_metric="map50_95",
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_registration(monkeypatch)

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(scheduler, child, [])

    assert registered == 1
    assert [trial.candidate_id for trial in scheduler.study.trials] == [
        "paper_candidate_1"
    ]


def test_asha_registration_failure_isolated_and_attributed_to_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="asha-registration-isolation",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
        dataset_version="dataset-83",
        dataset_manifest_sha256="dataset-83",
    )
    child = LoopOrchestrator(context)
    failed = _node(tmp_path, 1, "paper:registration-failed")
    ready = _node(tmp_path, 2, "paper:registration-ready")
    baseline = _node(tmp_path, 0, "baseline", baseline=True)
    build_round_execution_plan(
        run_id=context.run_id,
        nodes=[failed, ready],
        baseline_control_node=baseline,
        primary_metric="map50_95",
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_registration(monkeypatch)

    scheduler = ASHAScheduler.create(context.run_id)
    register_trial = scheduler.register_trial

    def register_with_one_failure(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["candidate_id"] == failed.candidate_config.candidate_id:
            raise ValueError("mock invalid trial")
        return register_trial(**kwargs)

    monkeypatch.setattr(scheduler, "register_trial", register_with_one_failure)
    registered = _register_guarded_pilot_trials(
        scheduler,
        child,
        [failed, ready],
    )

    assert registered == 1
    assert [trial.candidate_id for trial in scheduler.study.trials] == [
        failed.candidate_config.candidate_id,
        ready.candidate_config.candidate_id,
    ]
    failed_trial = scheduler.study.trial(
        f"{context.run_id}:{failed.candidate_config.candidate_id}"
    )
    assert failed_trial.readiness_state == "pre_registered"
    assert failed_trial.status == "needs_evidence"
    assert failed_trial.pending_stage is None
    assert scheduler.next_assignment() is not None
    coverage = PaperCandidateCoverage.from_yaml(
        context.artifact_path("paper_candidate_coverage.yaml")
    )
    dispositions = {record.paper_ids[0]: record for record in coverage.records}
    assert dispositions["paper:registration-failed"].disposition == "blocked_runtime"
    assert dispositions["paper:registration-failed"].reason_codes == [
        "asha_registration_failed:ValueError"
    ]
    assert dispositions["paper:registration-ready"].disposition == "queued"
    assert context.metadata["asha_registration_failures_by_paper_id"] == {
        "paper:registration-failed": 1,
    }


def test_all_eligible_registration_failures_raise_invariant_and_keep_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RunContext(
        run_id="asha-registration-invariant",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
        dataset_version="dataset-83",
        dataset_manifest_sha256="dataset-83",
    )
    child = LoopOrchestrator(context)
    candidates = [_node(tmp_path, 1, "paper:one"), _node(tmp_path, 2, "paper:two")]
    baseline = _node(tmp_path, 0, "baseline", baseline=True)
    build_round_execution_plan(
        run_id=context.run_id,
        nodes=candidates,
        baseline_control_node=baseline,
        primary_metric="map50_95",
    ).to_yaml(context.artifact_path("round_execution_plan.yaml"))
    _allow_mock_registration(monkeypatch)
    scheduler = ASHAScheduler.create(context.run_id)

    def fail_registration(**kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("mock registration failure")

    monkeypatch.setattr(scheduler, "register_trial", fail_registration)
    with pytest.raises(RuntimeError, match="no runnable trial for eligible"):
        _register_guarded_pilot_trials(scheduler, child, candidates)

    assert len(scheduler.study.trials) == 2
    assert all(
        trial.readiness_state == "pre_registered"
        and trial.status == "needs_evidence"
        and trial.pending_stage is None
        for trial in scheduler.study.trials
    )


def test_pre_registered_identity_upgrades_only_through_formal_registration(
    tmp_path: Path,
) -> None:
    context = RunContext(
        run_id="asha-preregister-upgrade",
        run_root=tmp_path / "runs",
        task_path=tmp_path / "task.yaml",
        data_yaml=tmp_path / "coco.yaml",
        dataset_version="dataset-83",
        dataset_manifest_sha256="dataset-83",
    )
    candidate = _node(tmp_path, 1, "paper:upgrade")
    baseline = _node(tmp_path, 0, "baseline", baseline=True)
    scheduler = ASHAScheduler.create(context.run_id)
    reserved = scheduler.pre_register_trial(
        trial_id=f"{context.run_id}:paper_candidate_1",
        candidate_id="paper_candidate_1",
        source_run_id=context.run_id,
        source_node=candidate,
        baseline_control_node=baseline,
        blockers=["teacher_checkpoint_missing"],
    )
    assert reserved.readiness_state == "pre_registered"
    assert scheduler.next_assignment() is None

    active = scheduler.register_trial(
        trial_id=reserved.trial_id,
        candidate_id=candidate.candidate_config.candidate_id,
        source_run_id=context.run_id,
        source_node=candidate,
        baseline_control_node=baseline,
        target_error_facts=candidate.candidate_config.target_error_facts,
        paper_ids=["paper:upgrade"],
        method_profile_ids=["profile:paper:upgrade"],
        readiness_state="asha_eligible",
        readiness_blockers=[],
    )
    assert active is reserved
    assert active.readiness_state == "asha_eligible"
    assert active.status == "waiting"
    assert active.pending_stage == "pilot_3"
    assert scheduler.next_assignment() is not None
