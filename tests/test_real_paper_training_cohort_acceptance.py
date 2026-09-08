"""Offline dry-run acceptance for the current paper training cohort.

The production artifacts provide paper identity, requirements, real asset
dispositions, and CPU/runtime readiness.  The temporary nodes and scheduler
state in this module only prove that a currently admissible COCO route can be
planned with a matched control.  They are never written to production
readiness artifacts and never execute a command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from yolo_agent.agents.asha_scheduler import ASHAScheduler
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.certification.paper_readiness import PaperReadinessReport
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.executor import DryRunExecutor
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.execution_fingerprint import execution_fingerprint
from yolo_agent.core.matched_baseline import assess_matched_control_plan
from yolo_agent.core.paper_training_readiness import PaperTrainingReadinessReport
from yolo_agent.core.round_execution_plan import build_round_execution_plan
from yolo_agent.research.paper_asset_dependencies import (
    requires_domain_assets,
    requires_hard_negative_replay,
    requires_teacher_checkpoint,
)
from yolo_agent.research.paper_asset_schemas import PaperAssetRegistry
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_execution_schemas import PaperExecutionInventory


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


def _mechanisms(item: Any, requirement: Any) -> set[str]:
    return set(item.canonical_component_ids) | set(
        item.paper_specific_mechanism_ids
    ) | set(requirement.paper_specific_mechanism_ids)


def _coco_dry_run_rows(
    inventory: PaperExecutionInventory,
    requirements: PaperExecutionRequirementsMatrix,
    assets: PaperAssetRegistry,
    readiness: PaperReadinessReport,
) -> list[tuple[Any, Any, Any, Any]]:
    requirements_by_id = {item.paper_id: item for item in requirements.requirements}
    assets_by_id = {item.paper_id: item for item in assets.records}
    readiness_by_id = {item.paper_id: item for item in readiness.records}
    rows = []
    for item in inventory.records:
        requirement = requirements_by_id[item.paper_id]
        asset = assets_by_id[item.paper_id]
        preflight = readiness_by_id[item.paper_id]
        mechanisms = _mechanisms(item, requirement)
        if not preflight.asha_eligibility:
            continue
        if requirement.execution_route == "inference":
            continue
        if not requirement.required_adapter:
            continue
        if not requirement.required_changed_variables:
            continue
        if not requirement.required_runtime_payload:
            continue
        if requirement.required_runtime_payload.get("mode") == "shadow":
            continue
        if not requirement.compatible_with_yolo26:
            continue
        if asset.availability != "available":
            continue
        if (
            requires_teacher_checkpoint(mechanisms)
            or requires_domain_assets(mechanisms)
            or requires_hard_negative_replay(mechanisms)
        ):
            continue
        rows.append((item, requirement, asset, preflight))
    return rows


def _protocol_value(preflight: Any, requirement: Any, name: str, fallback: str) -> str:
    value = getattr(preflight, name, None)
    if value and value != "missing":
        return str(value)
    value = getattr(requirement, name, None)
    if value and value != "missing":
        return str(value)
    return fallback


def _candidate_node(
    tmp_path: Path,
    index: int,
    item: Any,
    requirement: Any,
    asset: Any,
    preflight: Any,
) -> ExperimentNode:
    recipe_id = requirement.recipe_ids[0]
    protocol_hash = _protocol_value(
        preflight, requirement, "protocol_hash", "a" * 64
    )
    dataset_hash = _protocol_value(
        preflight, requirement, "dataset_manifest_hash", "b" * 64
    )
    candidate_id = f"paper_dry_run_{index:02d}_{item.paper_id.replace(':', '_')}"
    metadata: dict[str, object] = {
        "paper_ids": item.paper_id,
        "method_profile_ids": item.profile_id,
        "adapter_runtime_entrypoint": requirement.required_adapter,
        "component_recipe_id": recipe_id,
        "component_recipe_version": "production-recipe",
        "run_protocol_hash": protocol_hash,
        "baseline_protocol_hash": protocol_hash,
        "dataset_manifest_sha256": dataset_hash,
        "fidelity": "pilot_3",
        "split": "coco_val",
        "imgsz": 640,
        "paper_readiness_state": "asha_eligible",
        "paper_readiness_blockers": "[]",
        "dry_run_only": True,
    }
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=asset.source_dataset_manifest or "coco.yaml",
        project=tmp_path / "dry-run-ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=640,
        batch=2,
        seed=1,
        metadata=metadata,  # type: ignore[arg-type]
    )
    changed_variables = {
        variable: 0.1 if "weight" in variable else True
        for variable in requirement.required_changed_variables
    }
    return ExperimentNode(
        node_id=f"node-{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            components=list(item.canonical_component_ids),
            action_domain="paper",
            action_id=recipe_id,
            search_tier="method",
            target_error_facts=[
                {"fact_type": evidence, "subject": "overall"}
                for evidence in requirement.required_evidence[:1]
            ],
        ),
        data_version=dataset_hash,
        seed=1,
        command=command.display(),
        command_spec=command,
        changed_variables=changed_variables,
    )


def _baseline_node(
    tmp_path: Path,
    candidate: ExperimentNode,
) -> ExperimentNode:
    assert candidate.command_spec is not None
    baseline_id = (
        "matched-baseline-dry-run-"
        + candidate.candidate_config.candidate_id.removeprefix("paper_dry_run_")
    )
    metadata = dict(candidate.command_spec.metadata)
    metadata.update(
        {
            "matched_baseline_control": True,
            "paper_readiness_state": "pre_registered",
            "paper_readiness_blockers": "[]",
        }
    )
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=next(
            (argument.split("=", 1)[1] for argument in candidate.command_spec.args if argument.startswith("data=")),
            "coco.yaml",
        ),
        project=tmp_path / "dry-run-ultralytics",
        name=baseline_id,
        epochs=3,
        imgsz=640,
        batch=2,
        seed=1,
        metadata=metadata,  # type: ignore[arg-type]
    )
    return ExperimentNode(
        node_id=f"node-{baseline_id}",
        candidate_config=CandidateConfig(
            candidate_id=baseline_id,
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            action_domain="baseline",
            action_id="matched-baseline",
            search_tier="method",
        ),
        data_version=candidate.data_version,
        seed=1,
        command=command.display(),
        command_spec=command,
    )


def test_production_identity_and_scoped_blockers_are_preserved() -> None:
    inventory, requirements, assets, readiness, final = _production_artifacts()
    inventory_ids = {item.paper_id for item in inventory.records}
    assert inventory.compatible_paper_count == 83
    assert len(inventory_ids) == 83
    assert inventory_ids == {item.paper_id for item in requirements.requirements}
    assert inventory_ids == {item.paper_id for item in assets.records}
    assert inventory_ids == {item.paper_id for item in readiness.records}
    assert final.actual_trained_count == 0
    assert final.gpu_probe == "not_run"

    requirements_by_id = {item.paper_id: item for item in requirements.requirements}
    assets_by_id = {item.paper_id: item for item in assets.records}
    readiness_by_id = {item.paper_id: item for item in readiness.records}
    for paper_id, requirement in requirements_by_id.items():
        mechanisms = _mechanisms(
            next(item for item in inventory.records if item.paper_id == paper_id),
            requirement,
        )
        asset = assets_by_id[paper_id]
        preflight = readiness_by_id[paper_id]
        if requires_teacher_checkpoint(mechanisms) and not asset.teacher_checkpoint:
            assert not preflight.asha_eligibility
        if requires_domain_assets(mechanisms):
            assert not preflight.asha_eligibility
            assert not (
                asset.source_dataset_manifest and asset.target_dataset_manifest
            )
        if requires_hard_negative_replay(mechanisms) and not asset.hard_negative_manifest:
            assert not preflight.asha_eligibility
            assert preflight.final_disposition in {
                "evidence_recovery",
                "blocked_runtime",
            }


def test_dry_run_cohort_has_matched_control_and_asha_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, requirements, assets, readiness, _ = _production_artifacts()
    rows = _coco_dry_run_rows(inventory, requirements, assets, readiness)
    assert rows, "production readiness must expose a COCO dry-run candidate"

    def fail_if_subprocess_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run acceptance must not invoke subprocess")

    monkeypatch.setattr("yolo_agent.core.executor.subprocess.run", fail_if_subprocess_called)
    executor = DryRunExecutor()
    scheduler = ASHAScheduler.create("real-paper-cohort-dry-run")
    trainable_fingerprints: set[str] = set()
    for index, (item, requirement, asset, preflight) in enumerate(rows):
        candidate = _candidate_node(
            tmp_path, index, item, requirement, asset, preflight
        )
        baseline = _baseline_node(tmp_path, candidate)
        plan = build_round_execution_plan(
            run_id=f"real-paper-cohort-dry-run-{index}",
            nodes=[candidate],
            baseline_control_node=baseline,
            ranks={candidate.candidate_config.candidate_id: 1},
            run_protocol_hash=preflight.protocol_hash,
            primary_metric="map50_95",
        )
        assert plan.status == "ready"
        assert len(plan.execution_nodes) == 2
        assert {
            bool(
                node.command_spec
                and node.command_spec.metadata.get("matched_baseline_control")
            )
            for node in plan.execution_nodes
        } == {False, True}
        assessment = assess_matched_control_plan(
            candidate,
            baseline,
            required_protocol_hash=preflight.protocol_hash,
        )
        assert assessment.matched_control_plan_ready
        assert assessment.plan is not None
        assert assessment.plan.imgsz == 640
        assert assessment.plan.protocol_hash == preflight.protocol_hash
        assert assessment.plan.dataset_manifest_hash == preflight.dataset_manifest_hash
        assert assessment.plan.fidelity == "pilot_3"

        results = [
            executor.execute(node, run_id=f"real-paper-cohort-dry-run-{index}")
            for node in plan.execution_nodes
        ]
        assert all(result.status == "dry_run" for result in results)

        trial = scheduler.register_trial(
            trial_id=f"paper-dry-run-trial-{index}",
            candidate_id=candidate.candidate_config.candidate_id,
            source_run_id="real-paper-cohort-dry-run",
            source_node=candidate,
            baseline_control_node=baseline,
            target_error_facts=candidate.candidate_config.target_error_facts,
            paper_ids=[item.paper_id],
            method_profile_ids=[item.profile_id],
            mechanism_ids=list(item.paper_specific_mechanism_ids),
            required_evidence=list(requirement.required_evidence),
        )
        trainable_fingerprints.add(execution_fingerprint(candidate))
        assert trial.matched_control_plan_ready
        assert trial.matched_control_result_ready is False
        assert not trial.observations
        assert trial.baseline_control_node is not None

    assert trainable_fingerprints <= scheduler.registered_execution_fingerprints
    assert len(scheduler.study.trials) == len(trainable_fingerprints)


def test_dry_run_does_not_write_production_report() -> None:
    before = FINAL_PATH.read_bytes()
    _production_artifacts()
    assert FINAL_PATH.read_bytes() == before
