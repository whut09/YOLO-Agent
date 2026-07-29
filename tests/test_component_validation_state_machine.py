from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.components.adapters import ComponentAdapterRegistry, DummyAdapter
from yolo_agent.components.adapters.base import AdapterContext, SmokeTestResult
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.components.validation_bridge import ComponentValidationBridge
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.recipes.schemas import AtomicRecipe


class StatefulValidationAdapter(DummyAdapter):
    fail_smoke = False

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        if self.fail_smoke:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                errors=["synthetic local smoke failure"],
            )
        return super().smoke_test(context).model_copy(
            update={"evidence_kind": "local"}
        )


class MissingRuntimePayloadAdapter(DummyAdapter):
    def build_runtime_payload(self, *args: object, **kwargs: object) -> None:
        return None


class AlternateSourceAdapter(StatefulValidationAdapter):
    adapter_version = "dummy.v2"
    source_commit = "alternate-source"


def _contract(
    *,
    component_id: str = "validation.stateful",
    maturity: str = "adapter_implemented",
) -> ComponentContract:
    return ComponentContract(
        component_id=component_id,
        display_name="Stateful validation adapter",
        category="augmentation",
        implementation_path="yolo_agent.components.adapters.dummy",
        adapter_class="DummyAdapter",
        maturity=maturity,
        fixed_imgsz_compatible=True,
        checkpoint_compatibility="unchanged_graph",
        supports_amp=True,
    )


def _registry(
    adapter: type[DummyAdapter] = StatefulValidationAdapter,
    *,
    component_id: str = "validation.stateful",
) -> ComponentAdapterRegistry:
    registry = ComponentAdapterRegistry()
    registry.register(component_id, adapter)
    return registry


def _validate(
    tmp_path: Path,
    *,
    contract: ComponentContract | None = None,
    registry: ComponentAdapterRegistry | None = None,
    protocol_hash: str = "protocol-1",
    smoke_evidence: str = "local",
    training_config: dict[str, object] | None = None,
):
    return ComponentValidationBridge(adapter_registry=registry or _registry()).validate(
        contract=contract or _contract(),
        workspace=tmp_path,
        protocol_hash=protocol_hash,
        base_command=[
            "yolo",
            "detect",
            "train",
            "model=yolo26n.pt",
            "data=coco.yaml",
            "imgsz=640",
        ],
        training_config=training_config,
        target_maturity="smoke_passed",
        smoke_evidence=smoke_evidence,
    )


def test_full_offline_validation_advances_each_artifact_backed_stage(
    tmp_path: Path,
) -> None:
    result = _validate(tmp_path)

    assert result.status == "completed"
    assert result.final_maturity == "smoke_passed"
    assert result.contract.can_execute
    artifacts = {
        item.target_maturity: item
        for item in result.contract.maturity_artifacts
        if item.status == "passed" and not item.mock
    }
    assert set(artifacts) == {"runtime_integrated", "unit_tested", "smoke_passed"}
    assert len({item.artifact_path for item in artifacts.values()}) == 3
    assert result.patch_preview_path not in {
        item.artifact_path for item in artifacts.values()
    }
    for stage, artifact in artifacts.items():
        assert artifact.protocol_hash == "protocol-1"
        assert artifact.metadata["validation_key"] == result.validation_key
        artifact.verify()
        assert result.stage_reports[stage].is_file()


def test_same_validation_resumes_without_duplicate_artifacts(tmp_path: Path) -> None:
    first = _validate(tmp_path)
    second = _validate(tmp_path)

    assert second.status == "completed"
    assert second.validation_key == first.validation_key
    assert second.stage_reports == first.stage_reports
    assert second.runtime_payload_path == first.runtime_payload_path
    assert second.contract.maturity_artifacts == first.contract.maturity_artifacts


def test_mock_smoke_is_retained_then_local_smoke_can_resume(tmp_path: Path) -> None:
    mock = _validate(tmp_path, smoke_evidence="mock")

    assert mock.status == "blocked"
    assert mock.final_maturity == "unit_tested"
    assert mock.blocked_by == ["mock_smoke_evidence_cannot_promote"]
    assert not mock.contract.can_execute
    mock_artifact = mock.contract.maturity_artifacts[-1]
    assert mock_artifact.target_maturity == "smoke_passed"
    assert mock_artifact.mock
    mock_artifact.verify()

    local = _validate(tmp_path, smoke_evidence="local")

    assert local.status == "completed"
    assert local.final_maturity == "smoke_passed"
    assert local.contract.can_execute
    smoke_artifacts = [
        item
        for item in local.contract.maturity_artifacts
        if item.target_maturity == "smoke_passed"
    ]
    assert len(smoke_artifacts) == 2
    assert {item.mock for item in smoke_artifacts} == {False, True}
    assert len({item.artifact_path for item in smoke_artifacts}) == 2
    for artifact in smoke_artifacts:
        artifact.verify()


def test_local_request_cannot_promote_adapter_reported_mock_smoke(
    tmp_path: Path,
) -> None:
    result = _validate(
        tmp_path,
        registry=_registry(DummyAdapter),
        smoke_evidence="local",
    )

    assert result.status == "blocked"
    assert result.final_maturity == "unit_tested"
    assert result.blocked_by == ["mock_smoke_evidence_cannot_promote"]
    smoke = result.contract.maturity_artifacts[-1]
    assert smoke.target_maturity == "smoke_passed"
    assert smoke.mock
    assert not result.contract.can_execute


def test_failed_smoke_is_recoverable_without_promotion(tmp_path: Path) -> None:
    StatefulValidationAdapter.fail_smoke = True
    try:
        failed = _validate(tmp_path)
    finally:
        StatefulValidationAdapter.fail_smoke = False

    assert failed.status == "failed"
    assert failed.final_maturity == "unit_tested"
    assert not failed.contract.can_execute
    failed_artifact = failed.contract.maturity_artifacts[-1]
    assert failed_artifact.target_maturity == "smoke_passed"
    assert failed_artifact.status == "failed"
    failed_artifact.verify()

    recovered = _validate(tmp_path)
    assert recovered.status == "completed"
    assert recovered.contract.can_execute
    assert len(
        [
            item
            for item in recovered.contract.maturity_artifacts
            if item.target_maturity == "smoke_passed"
        ]
    ) == 2
    failed_artifact.verify()


def test_missing_runtime_payload_stays_adapter_implemented(tmp_path: Path) -> None:
    result = _validate(
        tmp_path,
        contract=_contract(component_id="validation.no_payload"),
        registry=_registry(
            MissingRuntimePayloadAdapter,
            component_id="validation.no_payload",
        ),
    )

    assert result.status == "failed"
    assert result.final_maturity == "adapter_implemented"
    assert result.blocked_by[0].startswith("runtime_payload_validation_failed:")
    assert result.contract.maturity_artifacts[-1].target_maturity == "runtime_integrated"
    assert result.contract.maturity_artifacts[-1].status == "failed"


def test_claimed_runtime_maturity_without_artifact_is_blocked(tmp_path: Path) -> None:
    result = _validate(tmp_path, contract=_contract(maturity="runtime_integrated"))

    assert result.status == "blocked"
    assert result.blocked_by == ["source_maturity_artifact_missing:runtime_integrated"]
    assert result.final_maturity == "runtime_integrated"
    assert not result.contract.can_execute


def test_protocol_and_config_changes_do_not_reuse_validation_state(
    tmp_path: Path,
) -> None:
    first = _validate(tmp_path, protocol_hash="protocol-1")
    changed_protocol = _validate(tmp_path, protocol_hash="protocol-2")
    changed_config = _validate(
        tmp_path,
        protocol_hash="protocol-2",
        training_config={"imgsz": 640, "epochs": 10},
    )
    changed_source = _validate(
        tmp_path,
        protocol_hash="protocol-2",
        training_config={"imgsz": 640, "epochs": 10},
        registry=_registry(AlternateSourceAdapter),
    )

    assert len({
        first.validation_key,
        changed_protocol.validation_key,
        changed_config.validation_key,
        changed_source.validation_key,
    }) == 4
    assert first.runtime_payload_path != changed_protocol.runtime_payload_path
    assert changed_protocol.runtime_payload_path != changed_config.runtime_payload_path
    assert changed_config.runtime_payload_path != changed_source.runtime_payload_path
    assert first.contract.maturity_artifacts != changed_protocol.contract.maturity_artifacts


def test_execution_bridge_only_accepts_validated_returned_contract(
    tmp_path: Path,
) -> None:
    contract = _contract()
    registry = _registry()
    recipe = AtomicRecipe(
        recipe_id="validation_recipe",
        version="v1",
        target_metrics=["map50_95"],
        component_ids=[contract.component_id],
        train_overrides={"imgsz": 640, "epochs": 3},
        fixed_variables={"imgsz": 640},
        primary_changed_variable="adapter_marker",
        stop_conditions=["no_gain"],
        maturity="smoke_passed",
    )
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data=tmp_path / "coco.yaml",
        project=tmp_path / "runs",
        name="validation_candidate",
        epochs=3,
        imgsz=640,
    )
    node = ExperimentNode(
        node_id="node_validation_candidate",
        candidate_config=CandidateConfig(
            candidate_id="validation_candidate",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
            components=[contract.component_id],
        ),
        data_version="coco2017",
        seed=1,
        command=command.display(),
        command_spec=command,
    )
    bridge = ComponentExecutionBridge(adapter_registry=registry)

    blocked = bridge.prepare(
        recipe=recipe,
        node=node,
        contracts={contract.component_id: contract},
        workspace=tmp_path / "execution-blocked",
        protocol_hash="protocol-1",
    )
    validated = _validate(tmp_path / "validation", registry=registry)
    executable = bridge.prepare(
        recipe=recipe,
        node=node,
        contracts={contract.component_id: validated.contract},
        workspace=tmp_path / "execution-ready",
        protocol_hash="protocol-1",
    )

    assert blocked.status == "adapter_required"
    assert executable.status == "executable"
    assert executable.runtime_payload_path is not None
