"""End-to-end offline tests for the component adapter execution bridge."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.agents.auto_optimization_loop import assess_candidate_execution
from yolo_agent.agents.loop_policy_evaluator import LoopPolicyEvaluation, LoopPolicyEvaluationReport
from yolo_agent.components.adapters import ComponentAdapterRegistry, DummyAdapter
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.adapters.ultralytics.training import UltralyticsRunImporter
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.schemas import AtomicRecipe


def _contract(maturity: str = "smoke_passed") -> ComponentContract:
    return ComponentContract(
        component_id="dummy.component",
        display_name="Dummy Component",
        category="augmentation",
        implementation_path="yolo_agent.components.adapters.dummy",
        adapter_class="DummyAdapter",
        maturity=maturity,
        fixed_imgsz_compatible=True,
        checkpoint_compatibility="unchanged_graph",
        supports_amp=True,
    )


def _recipe() -> AtomicRecipe:
    return AtomicRecipe(
        recipe_id="dummy_recipe",
        version="v1",
        target_metrics=["map50_95"],
        component_ids=["dummy.component"],
        train_overrides={"imgsz": 640, "epochs": 3},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="adapter_marker",
        stop_conditions=["no_gain"],
        maturity="smoke_passed",
    )


def _node(tmp_path: Path) -> ExperimentNode:
    candidate = CandidateConfig(
        candidate_id="dummy_candidate",
        base_model="yolo26n.pt",
        scale="n",
        framework="ultralytics",
        components=["dummy.component"],
    )
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        project=tmp_path / "runs",
        name="dummy_candidate",
        epochs=3,
        imgsz=640,
    )
    return ExperimentNode(
        node_id="node_dummy_candidate",
        candidate_config=candidate,
        data_version="coco2017",
        seed=1,
        command=command.display(),
        command_spec=command,
    )


def test_dummy_adapter_bridges_recipe_to_executable_node_and_evidence(tmp_path: Path) -> None:
    registry = ComponentAdapterRegistry()
    registry.register("dummy.component", DummyAdapter)
    store = EvidenceStore(tmp_path / "runs")
    node = _node(tmp_path)

    result = ComponentExecutionBridge(adapter_registry=registry).prepare(
        recipe=_recipe(),
        node=node,
        contracts={"dummy.component": _contract()},
        model_config={"model": "yolo26n.pt"},
        training_config={"epochs": 3, "imgsz": 640},
        workspace=tmp_path / "runs" / "run-1" / "artifacts" / "component_execution" / node.node_id,
        evidence_store=store,
        run_id="run-1",
        protocol_hash="protocol-1",
    )

    assert result.status == "executable"
    assert result.aggregate_patch_hash
    assert result.adapters[0].adapter_version == "dummy.v1"
    assert result.adapters[0].source_commit == "local-test"
    assert result.changed_variables == {"training_config.adapter_marker": "dummy.component"}
    assert result.adapters[0].rollback_plan.actions == ["discard generated adapter patch"]
    assert result.node.command_spec is not None
    assert result.node.command_spec.metadata["adapter_patch_hash"] == result.aggregate_patch_hash
    assert result.node.command_spec.metadata["adapter_versions"] == '{"dummy.component": "dummy.v1"}'
    assert result.evidence_path is not None and result.evidence_path.is_file()

    evidence = store.load_run("run-1")
    smoke = [item for item in evidence.metric_records if item.metric_name == "adapter_smoke_passed"]
    assert len(smoke) == 1 and smoke[0].value is True
    manifest = next(item for item in evidence.artifact_manifest if item.name.endswith("component_execution"))
    assert manifest.node_id == node.node_id
    assert manifest.protocol_hash == "protocol-1"


def test_bridge_is_idempotent_for_same_recipe_and_node(tmp_path: Path) -> None:
    registry = ComponentAdapterRegistry()
    registry.register("dummy.component", DummyAdapter)
    bridge = ComponentExecutionBridge(adapter_registry=registry)

    first = bridge.prepare(
        recipe=_recipe(), node=_node(tmp_path), contracts={"dummy.component": _contract()}, workspace=tmp_path / "bridge"
    )
    second = bridge.prepare(
        recipe=_recipe(), node=_node(tmp_path), contracts={"dummy.component": _contract()}, workspace=tmp_path / "bridge"
    )

    assert first.aggregate_patch_hash == second.aggregate_patch_hash
    assert first.adapters[0].patch_hash == second.adapters[0].patch_hash


def test_bridge_blocks_metadata_only_component_before_adapter_creation(tmp_path: Path) -> None:
    result = ComponentExecutionBridge().prepare(
        recipe=_recipe(),
        node=_node(tmp_path),
        contracts={"dummy.component": _contract("metadata_only")},
        workspace=tmp_path / "bridge",
    )

    assert result.status == "adapter_required"
    assert result.adapters == []
    assert result.blocked_by == ["component_maturity_below_smoke_passed:dummy.component:metadata_only"]


def test_bridge_resolves_adapter_from_contract_without_manual_registration(tmp_path: Path) -> None:
    result = ComponentExecutionBridge().prepare(
        recipe=_recipe(),
        node=_node(tmp_path),
        contracts={"dummy.component": _contract()},
        workspace=tmp_path / "bridge",
    )

    assert result.status == "executable"
    assert result.adapters[0].adapter_class == "DummyAdapter"


def test_completed_training_imports_adapter_execution_evidence(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "runs")
    node = _node(tmp_path)
    prepared = ComponentExecutionBridge().prepare(
        recipe=_recipe(),
        node=node,
        contracts={"dummy.component": _contract()},
        workspace=tmp_path / "runs" / "run-1" / "artifacts" / "component_execution" / node.node_id,
    )

    metrics = UltralyticsRunImporter(store)._import_adapter_execution_evidence(
        run_id="run-1",
        node=prepared.node,
        source="ultralytics_train",
        verified=True,
        matched_identity={"protocol_hash": "protocol-1"},
    )

    assert metrics["adapter_training_completed"] is True
    assert metrics["adapter_patch_hash"] == prepared.aggregate_patch_hash
    evidence = store.load_run("run-1")
    completed = [item for item in evidence.metric_records if item.metric_name == "adapter_training_completed"]
    assert len(completed) == 1
    assert completed[0].protocol_hash == "protocol-1"
    assert completed[0].source_artifact == prepared.evidence_path


def test_policy_assessment_uses_contract_and_bridge_instead_of_component_prefix(tmp_path: Path) -> None:
    node = _node(tmp_path)
    report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="dummy_policy",
                decision="accepted",
                candidate_config=node.candidate_config,
                experiment_node=node,
                changed_variables={"adapter_marker": "dummy.component"},
                fixed_variables={"imgsz": 640},
            )
        ]
    )

    assessments = assess_candidate_execution(
        report,
        component_contracts=[_contract()],
        workspace=tmp_path / "bridge",
    )

    assert assessments[0].execution_class == "executable"
    assert assessments[0].required_adapters == []
    patched = report.evaluations[0].experiment_node
    assert patched is not None and patched.command_spec is not None
    assert patched.command_spec.metadata["component_ids"] == "dummy.component"


def test_policy_assessment_blocks_by_maturity_not_component_name_prefix(tmp_path: Path) -> None:
    node = _node(tmp_path)
    report = LoopPolicyEvaluationReport(
        evaluations=[
            LoopPolicyEvaluation(
                policy_id="dummy_policy",
                decision="accepted",
                candidate_config=node.candidate_config,
                experiment_node=node,
            )
        ]
    )

    assessment = assess_candidate_execution(
        report,
        component_contracts=[_contract("unit_tested")],
        workspace=tmp_path / "bridge",
    )[0]

    assert assessment.execution_class == "adapter_required"
    assert assessment.required_adapters == ["component_adapter:dummy.component"]
    assert any("component_maturity_below_smoke_passed" in reason for reason in assessment.reasons)
