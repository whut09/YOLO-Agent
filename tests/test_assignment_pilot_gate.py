from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import yolo_agent.agents.auto_optimization_loop as auto_loop
from yolo_agent.agents.asha_scheduler import ASHATrial
from yolo_agent.agents.asha_scheduler import ASHAScheduler
from tests.assignment_fixtures import assignment_node, assignment_recipes, run_one_shadow_batch
from tests.maturity_helpers import with_smoke_artifact
from yolo_agent.certification.assignment_pilot_gate import (
    AssignmentActivePilotMaterializer,
)
from yolo_agent.components.contracts import load_contracts
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.core.round_execution_plan import build_asha_assignment_plan


@pytest.mark.parametrize(
    ("recipe_id", "method"),
    [
        ("yolo26_tood_tal_assignment_shadow", "tood_tal"),
        ("yolo26_ota_assignment_shadow", "ota"),
        ("yolo26_dsla_assignment_shadow", "dsla"),
        (
            "yolo26_task_aligned_weighting_shadow",
            "task_aligned_weighting",
        ),
        ("yolo26_dynamic_topk_assignment_shadow", "dynamic_topk"),
        ("yolo26_quality_aware_assignment_shadow", "quality_aware"),
        ("yolo26_soft_label_assignment_shadow", "soft_label"),
        ("yolo26_dual_path_assignment_shadow", "dual_path"),
        ("yolo26_conflict_aware_assignment_shadow", "conflict_aware"),
    ],
)
def test_active_assignment_recipe_requires_matching_shadow_and_control(
    recipe_id: str,
    method: str,
    tmp_path: Path,
) -> None:
    shadow_dir = tmp_path / method
    shadow_dir.mkdir()
    assignment_path = "both" if method == "dual_path" else "one_to_many"
    run_one_shadow_batch(shadow_dir, method, assignment_path=assignment_path)
    evidence = shadow_dir / f"assignment_{method}_shadow_evidence.json"
    recipe = next(item for item in assignment_recipes() if item.recipe_id == recipe_id)
    protocol_hash = f"protocol-{method}"

    decision = AssignmentActivePilotMaterializer().materialize(
        shadow_recipe=recipe,
        shadow_evidence_path=evidence,
        candidate_protocol_hash=protocol_hash,
        control_protocol_hash=protocol_hash,
        matched_control_available=True,
    )

    assert decision.allowed is True
    assert decision.active_recipe is not None
    assert decision.active_recipe.train_overrides[recipe.primary_changed_variable] == "active"
    assert decision.active_recipe.train_overrides["assignment.shadow_evidence_path"] == str(
        evidence.resolve()
    )
    assert (
        decision.active_recipe.train_overrides["assignment.shadow_payload_hash"]
        == f"payload-{method}-shadow"
    )
    assert "matched_control" in decision.active_recipe.promotion_requirements
    assert "ASHA_only" in decision.active_recipe.promotion_requirements
    assert decision.shadow_evidence_sha256
    assert decision.assignment_path == assignment_path


def test_active_assignment_recipe_rejects_missing_or_unmatched_control(
    tmp_path: Path,
) -> None:
    recipe = assignment_recipes()[0]
    missing = AssignmentActivePilotMaterializer().materialize(
        shadow_recipe=recipe,
        shadow_evidence_path=tmp_path / "missing.json",
        candidate_protocol_hash="candidate",
        control_protocol_hash="control",
        matched_control_available=False,
    )

    assert missing.allowed is False
    assert "matched_control_missing" in missing.blocked_by
    assert "matched_control_protocol_mismatch" in missing.blocked_by
    assert "shadow_evidence_missing" in missing.blocked_by


def test_active_assignment_recipe_rejects_shadow_protocol_mismatch(
    tmp_path: Path,
) -> None:
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    run_one_shadow_batch(shadow_dir, "tood_tal")
    recipe = assignment_recipes()[0]

    decision = AssignmentActivePilotMaterializer().materialize(
        shadow_recipe=recipe,
        shadow_evidence_path=shadow_dir / "assignment_tood_tal_shadow_evidence.json",
        candidate_protocol_hash="different-protocol",
        control_protocol_hash="different-protocol",
        matched_control_available=True,
    )

    assert decision.allowed is False
    assert "shadow_evidence_protocol_mismatch" in decision.blocked_by


def test_active_assignment_recipe_builds_rankable_runtime_payload(tmp_path: Path) -> None:
    shadow_dir = tmp_path / "shadow-active"
    shadow_dir.mkdir()
    run_one_shadow_batch(shadow_dir, "dsla")
    shadow_recipe = next(
        item for item in assignment_recipes() if item.recipe_id == "yolo26_dsla_assignment_shadow"
    )
    evidence = shadow_dir / "assignment_dsla_shadow_evidence.json"
    decision = AssignmentActivePilotMaterializer().materialize(
        shadow_recipe=shadow_recipe,
        shadow_evidence_path=evidence,
        candidate_protocol_hash="protocol-dsla",
        control_protocol_hash="protocol-dsla",
        matched_control_available=True,
    )
    assert decision.active_recipe is not None
    contract = next(
        with_smoke_artifact(item)
        for item in load_contracts("configs/components/assigner/yolo26_assignment.yaml")
        if item.component_id == "assigner.dynamic_smooth_label"
    )

    result = ComponentExecutionBridge().prepare(
        recipe=decision.active_recipe,
        node=assignment_node(decision.active_recipe, tmp_path),
        contracts={contract.component_id: contract},
        training_config=dict(decision.active_recipe.train_overrides),
        workspace=tmp_path / "active",
        protocol_hash="protocol-dsla",
    )

    assert result.status == "executable", result.blocked_by
    assert result.node.command_spec is not None
    metadata = result.node.command_spec.metadata
    assert metadata["assignment_execution_mode"] == "active"
    assert metadata["evidence_only"] is False
    assert metadata["optimization_metric_eligible"] is True
    assert result.runtime_payload_path is not None
    payload = yaml.safe_load(result.runtime_payload_path.read_text(encoding="utf-8"))
    options = payload["assigner_plugin"][0]["options"]
    assert options["mode"] == "active"
    assert options["shadow_evidence_path"] == str(evidence.resolve())
    assert options["shadow_payload_hash"] == "payload-dsla-shadow"


def test_auto_loop_materializes_active_assignment_from_matching_runtime_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = next(
        item for item in assignment_recipes() if item.recipe_id == "yolo26_dsla_assignment_shadow"
    )
    recipe = recipe.model_copy(
        update={
            "train_overrides": {
                **recipe.train_overrides,
                "assignment.minimum_shadow_batches": 1,
            }
        }
    )
    contract = next(
        with_smoke_artifact(item)
        for item in load_contracts("configs/components/assigner/yolo26_assignment.yaml")
        if item.component_id == "assigner.dynamic_smooth_label"
    )
    source = assignment_node(recipe, tmp_path)
    shadow = ComponentExecutionBridge().prepare(
        recipe=recipe,
        node=source,
        contracts={contract.component_id: contract},
        training_config=dict(recipe.train_overrides),
        workspace=tmp_path / "runtime-shadow",
        protocol_hash="protocol-dsla",
    )
    assert shadow.runtime_payload_path is not None
    evidence_dir = shadow.runtime_payload_path.parent
    run_one_shadow_batch(evidence_dir, "dsla")
    evidence = evidence_dir / "assignment_dsla_shadow_evidence.json"
    raw = json.loads(evidence.read_text(encoding="utf-8"))
    raw["runtime_payload_hash"] = shadow.runtime_payload_hash
    evidence.write_text(json.dumps(raw), encoding="utf-8")
    control = assignment_node(recipe, tmp_path).model_copy(
        update={"node_id": "node_matched_control"}
    )
    assert control.command_spec is not None
    control.command_spec.metadata["baseline_protocol_hash"] = "protocol-dsla"
    trial = ASHATrial(
        trial_id="shadow-trial",
        candidate_id=shadow.node.candidate_config.candidate_id,
        source_run_id="fixture",
        source_node=shadow.node,
        baseline_control_node=control,
    )
    monkeypatch.setattr(auto_loop, "_load_execution_contracts", lambda _: [contract])
    context = SimpleNamespace(
        run_id="fixture",
        artifact_path=lambda name: tmp_path / "artifacts" / name,
    )
    orchestrator = SimpleNamespace(context=context, evidence_store=None)

    prepared, blockers = auto_loop._materialize_active_assignment_node(
        orchestrator,
        trial=trial,
    )

    assert blockers == []
    assert prepared is not None
    active_node, active_recipe = prepared
    assert active_recipe.train_overrides[recipe.primary_changed_variable] == "active"
    assert active_node.candidate_config.candidate_id.endswith("_active")
    assert active_node.command_spec is not None
    assert active_node.command_spec.metadata["assignment_execution_mode"] == "active"
    assert active_node.command_spec.metadata["optimization_metric_eligible"] is True


def test_auto_loop_rejects_stale_assignment_shadow_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = next(
        item for item in assignment_recipes() if item.recipe_id == "yolo26_dsla_assignment_shadow"
    )
    recipe = recipe.model_copy(
        update={
            "train_overrides": {
                **recipe.train_overrides,
                "assignment.minimum_shadow_batches": 1,
            }
        }
    )
    contract = next(
        with_smoke_artifact(item)
        for item in load_contracts("configs/components/assigner/yolo26_assignment.yaml")
        if item.component_id == "assigner.dynamic_smooth_label"
    )
    shadow = ComponentExecutionBridge().prepare(
        recipe=recipe,
        node=assignment_node(recipe, tmp_path),
        contracts={contract.component_id: contract},
        training_config=dict(recipe.train_overrides),
        workspace=tmp_path / "runtime-stale",
        protocol_hash="protocol-dsla",
    )
    assert shadow.runtime_payload_path is not None
    run_one_shadow_batch(shadow.runtime_payload_path.parent, "dsla")
    control = assignment_node(recipe, tmp_path)
    assert control.command_spec is not None
    control.command_spec.metadata["baseline_protocol_hash"] = "protocol-dsla"
    trial = ASHATrial(
        trial_id="stale-shadow-trial",
        candidate_id=shadow.node.candidate_config.candidate_id,
        source_run_id="fixture",
        source_node=shadow.node,
        baseline_control_node=control,
    )
    monkeypatch.setattr(auto_loop, "_load_execution_contracts", lambda _: [contract])
    orchestrator = SimpleNamespace(
        context=SimpleNamespace(
            run_id="fixture",
            artifact_path=lambda name: tmp_path / "artifacts" / name,
        ),
        evidence_store=None,
    )

    prepared, blockers = auto_loop._materialize_active_assignment_node(
        orchestrator,
        trial=trial,
    )

    assert prepared is None
    assert blockers == ["assignment_shadow_source_payload_mismatch"]


def test_shadow_assignment_plan_runs_evidence_only_without_matched_baseline(
    tmp_path: Path,
) -> None:
    recipe = next(
        item for item in assignment_recipes() if item.recipe_id == "yolo26_dsla_assignment_shadow"
    )
    candidate = assignment_node(recipe, tmp_path)
    control = assignment_node(recipe, tmp_path).model_copy(
        update={"node_id": "node_matched_baseline_control"}
    )
    assert control.command_spec is not None
    control.command_spec.metadata["matched_baseline_control"] = True
    plan = build_asha_assignment_plan(
        run_id="shadow-run",
        source_node=candidate,
        stage_id="pilot_3",
        epochs=1,
        fraction=0.01,
        seed=42,
        baseline_control_node=control,
    )

    evidence_plan = auto_loop._evidence_only_assignment_plan(plan)

    assert len(evidence_plan.execution_nodes) == 1
    assert len(evidence_plan.assignments) == 1
    assert evidence_plan.assignments[0].matched_control_execution_node_id is None
    assert evidence_plan.assignments[0].reason == "assignment_shadow_evidence_only"
    node = evidence_plan.execution_nodes[0]
    assert node.command_spec is not None
    assert node.command_spec.expected_metrics == []
    assert node.command_spec.metadata["evidence_only"] is True
    assert node.command_spec.metadata["matched_pilot_required"] is False
    assert evidence_plan.primary_metric == "assignment_shadow_evidence"


def test_auto_loop_does_not_migrate_running_assignment_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = next(
        item for item in assignment_recipes() if item.recipe_id == "yolo26_dsla_assignment_shadow"
    )
    contract = next(
        with_smoke_artifact(item)
        for item in load_contracts("configs/components/assigner/yolo26_assignment.yaml")
        if item.component_id == "assigner.dynamic_smooth_label"
    )
    shadow = ComponentExecutionBridge().prepare(
        recipe=recipe,
        node=assignment_node(recipe, tmp_path),
        contracts={contract.component_id: contract},
        training_config=dict(recipe.train_overrides),
        workspace=tmp_path / "runtime-running",
        protocol_hash="protocol-dsla",
    )
    scheduler = ASHAScheduler.create("running-shadow")
    scheduler.register_trial(
        trial_id="running-shadow:dsla",
        candidate_id=shadow.node.candidate_config.candidate_id,
        source_run_id="fixture",
        source_node=shadow.node,
        baseline_control_node=assignment_node(recipe, tmp_path),
    )
    assignment = scheduler.next_assignment()
    assert assignment is not None
    scheduler.mark_running(assignment, run_id="running-r1", node_id="shadow-node")
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> tuple[bool, list[str]]:
        nonlocal called
        called = True
        return False, []

    monkeypatch.setattr(auto_loop, "_activate_assignment_shadow_trial", fail_if_called)
    orchestrator = SimpleNamespace(context=SimpleNamespace())

    changed = auto_loop._activate_completed_assignment_shadows(
        orchestrator,
        scheduler,
    )

    assert changed is False
    assert called is False
    assert assignment.status == "running"
