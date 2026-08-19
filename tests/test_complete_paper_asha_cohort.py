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
    for node in candidates:
        metadata = node.command_spec.metadata
        ledger.upsert(
            planned_recipe_disposition(
                run_id=context.run_id,
                round_index=1,
                recipe_id=str(metadata["component_recipe_id"]),
                recipe_version="v1",
                component_ids=node.candidate_config.components,
                decision="selected",
                reasons=[],
                related_papers=[str(metadata["paper_id"])],
                method_profile_ids=[str(metadata["method_profile_ids"])],
                execution_fingerprint=hashlib.sha256(
                    node.candidate_config.candidate_id.encode()
                ).hexdigest(),
                candidate_id=node.candidate_config.candidate_id,
                protocol_hash="protocol-640",
                dataset_manifest_hash="dataset-83",
            )
        )
    baseline = _node(tmp_path, 0, "baseline", baseline=True)
    round_plan = build_round_execution_plan(
        run_id=context.run_id,
        nodes=candidates,
        baseline_control_node=baseline,
        ranks={node.candidate_config.candidate_id: index for index, node in enumerate(candidates)},
        primary_metric="map50_95",
    )
    round_plan.to_yaml(context.artifact_path("round_execution_plan.yaml"))

    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.AutomaticRuntimeReadinessGate.evaluate_node",
        lambda self, node: type("Readiness", (), {"allowed": True})(),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.ComponentQueueCertificationGate.evaluate",
        lambda *args, **kwargs: type(
            "Certification", (), {"allowed": True, "blockers": [], "report_path": None, "report_hash": "mock"}
        )(),
    )
    monkeypatch.setattr(
        "yolo_agent.agents.auto_optimization_loop.validate_certified_runtime_node",
        lambda node: [],
    )

    scheduler = ASHAScheduler.create(context.run_id)
    registered = _register_guarded_pilot_trials(scheduler, child, candidates)

    coverage = PaperCandidateCoverage.from_yaml(ledger.path)
    assert registered == 83
    assert len(scheduler.study.trials) == 83
    assert all(trial.baseline_control_node is not None for trial in scheduler.study.trials)
    assert coverage.expected_paper_count == 83
    assert len(coverage.current_by_paper) == 83
    assert all(
        coverage.current_by_paper[paper_id].disposition == "queued"
        for paper_id in paper_ids
    )
    assert context.metadata["asha_registration_paper_summary"]["asha_trials_registered"] == 83
    assert context.metadata["asha_registration_paper_summary"] == {
        "inventory_count": 83,
        "eligible_count": 83,
        "queued_count": 83,
        "deferred_count": 0,
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
