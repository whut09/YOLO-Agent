import json
from pathlib import Path

import pytest

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.inference.plugin import (
    ClassAwareThresholdInferenceAdapter,
    ConfidenceCalibrationInferenceAdapter,
    InferencePolicyPlugin,
    MergePolicyInferenceAdapter,
    TestTimeAugmentationAdapter,
    TiledMultiScaleInferenceAdapter,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.adapters.audit_contract import (
    validate_audited_runtime_payload,
)


def _context(tmp_path: Path, component_id: str, adapter: object) -> AdapterContext:
    return AdapterContext(
        contract=ComponentContract(
            component_id=component_id,
            display_name=component_id,
            category="inference_policy",
            implementation_path="local",
            adapter_class=type(adapter).__name__,
            inference_only=True,
            training_only=False,
            changes_model_graph=False,
            fixed_imgsz_compatible=True,
        ),
        detector_family="yolo26",
        head="one_to_one",
        workspace=tmp_path,
    )


@pytest.mark.parametrize(
    ("component_id", "adapter", "changed_variable"),
    [
        ("inference.tiled_multi_scale", TiledMultiScaleInferenceAdapter(), "inference.tiled_multi_scale_policy"),
        ("inference.test_time_augmentation", TestTimeAugmentationAdapter(), "inference.tta_policy"),
        ("inference.confidence_calibration", ConfidenceCalibrationInferenceAdapter(), "inference.confidence_calibration"),
        ("inference.class_aware_thresholding", ClassAwareThresholdInferenceAdapter(), "inference.class_thresholds"),
        ("inference.merge_policy", MergePolicyInferenceAdapter(), "inference.merge_policy"),
    ],
)
def test_reusable_adapters_build_inference_only_payloads(
    tmp_path: Path, component_id: str, adapter, changed_variable: str  # type: ignore[no-untyped-def]
) -> None:
    context = _context(tmp_path, component_id, adapter)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol",
        base_command=[
            "yolo-agent",
            "advanced",
            "certify-inference-policy",
            "--execute",
        ],
        generated_config={},
    )

    assert payload.component_ids == [component_id]
    assert set(payload.changed_variables) == {changed_variable}
    assert len(payload.inference_plugin) == 1
    assert not payload.dataloader_plugin and not payload.loss_plugin
    assert payload.supports_amp is False
    audit = validate_audited_runtime_payload(payload, component_id)
    assert audit["audited_runtime_component"] is True
    assert audit["audited_plugin_kind"] == "inference_plugin"


def test_runtime_plugin_records_identity_for_real_command(tmp_path: Path) -> None:
    adapter = TestTimeAugmentationAdapter()
    context = _context(tmp_path, "inference.test_time_augmentation", adapter)
    command = [
        "yolo-agent",
        "advanced",
        "certify-inference-policy",
        "--execute",
    ]
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol",
        base_command=command,
        generated_config={},
    )
    reference = payload.inference_plugin[0]
    plugin = InferencePolicyPlugin(**reference.options)

    actual, _ = plugin.prepare_command(payload=payload, command=command, env={})
    evidence = json.loads(
        (tmp_path / "inference_policy_runtime_evidence.json").read_text(encoding="utf-8")
    )

    assert actual == command
    assert evidence["payload_hash"] == payload.payload_hash
    assert evidence["training_attribution_allowed"] is False
    assert evidence["hook_call_counts"] == {"prepare_command": 1}


def test_runtime_plugin_refuses_uninstrumented_predict(tmp_path: Path) -> None:
    adapter = ConfidenceCalibrationInferenceAdapter()
    context = _context(tmp_path, "inference.confidence_calibration", adapter)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol",
        base_command=["yolo", "detect", "predict"],
        generated_config={},
    )
    plugin = InferencePolicyPlugin(**payload.inference_plugin[0].options)

    with pytest.raises(ValueError, match="certify-inference-policy"):
        plugin.prepare_command(
            payload=payload,
            command=["yolo", "detect", "predict"],
            env={},
        )
