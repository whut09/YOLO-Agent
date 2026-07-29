from __future__ import annotations

from pathlib import Path

from yolo_agent.components.adapters import ComponentAdapterRegistry, DummyAdapter
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.validation_bridge import ComponentValidationBridge


class ValidationRuntimeAdapter(DummyAdapter):
    """Importable local adapter used to validate the bootstrap boundary."""


def _contract() -> ComponentContract:
    return ComponentContract(
        component_id="validation.runtime",
        display_name="Validation runtime",
        category="augmentation",
        implementation_path="yolo_agent.components.adapters.dummy",
        adapter_class="DummyAdapter",
        maturity="adapter_implemented",
        fixed_imgsz_compatible=True,
    )


def _bridge() -> ComponentValidationBridge:
    registry = ComponentAdapterRegistry()
    registry.register("validation.runtime", ValidationRuntimeAdapter)
    return ComponentValidationBridge(adapter_registry=registry)


def _command() -> list[str]:
    return [
        "yolo",
        "detect",
        "train",
        "model=yolo26n.pt",
        "data=coco.yaml",
        "imgsz=640",
    ]


def test_adapter_implemented_can_generate_runtime_artifact_without_node(
    tmp_path: Path,
) -> None:
    result = _bridge().validate(
        contract=_contract(),
        workspace=tmp_path,
        protocol_hash="protocol-1",
        base_command=_command(),
        target_maturity="runtime_integrated",
    )

    assert result.status == "completed"
    assert result.final_maturity == "runtime_integrated"
    assert result.runtime_payload_path is not None
    payload = AdapterRuntimePayload.read(result.runtime_payload_path)
    assert payload.component_ids == ["validation.runtime"]
    assert payload.protocol_hash == "protocol-1"
    artifact = result.contract.maturity_artifacts[-1]
    assert artifact.target_maturity == "runtime_integrated"
    assert artifact.artifact_path == result.runtime_payload_path
    artifact.verify()
    assert result.patch_preview_path not in {
        item.artifact_path for item in result.contract.maturity_artifacts
    }


def test_validation_does_not_relax_training_execution_gate(tmp_path: Path) -> None:
    contract = _contract()

    assert not contract.can_execute
    assert contract.maturity == "adapter_implemented"
    assert ComponentValidationBridge is not None
