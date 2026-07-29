"""Typed model-graph contract, dependency, and resource guard tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yolo_agent.components.model_graph import (  # noqa: E402
    FeaturePyramidContract,
    ModelGraphDependencyGate,
    ModelGraphResourceLimits,
    evaluate_resource_guards,
)


def test_feature_pyramid_contract_validates_stride_channel_boundary() -> None:
    contract = FeaturePyramidContract(strides=[8, 16, 32], channels=[8, 16, 32])
    features = [
        torch.zeros(1, 8, 8, 8),
        torch.zeros(1, 16, 4, 4),
        torch.zeros(1, 32, 2, 2),
    ]
    contract.validate_features(features, 64)

    with pytest.raises(ValueError, match="channels"):
        contract.validate_features([features[0], features[0], features[2]], 64)
    with pytest.raises(ValueError, match="stride"):
        contract.validate_features([torch.zeros(1, 8, 7, 7), *features[1:]], 64)


def test_resource_guards_are_independent_and_hard() -> None:
    report = evaluate_resource_guards(
        base_latency_ms=10.0,
        candidate_latency_ms=12.0,
        base_vram_estimate_mb=100.0,
        candidate_vram_estimate_mb=160.0,
        base_parameter_count=100,
        candidate_parameter_count=105,
        base_model_size_mb=5.0,
        candidate_model_size_mb=5.2,
        limits=ModelGraphResourceLimits(
            max_latency_regression=0.25,
            max_vram_regression=0.5,
            max_parameter_regression=0.1,
            max_model_size_regression=0.1,
        ),
    )

    assert report.checks == {
        "latency": True,
        "vram": False,
        "parameters": True,
        "model_size": True,
    }
    assert report.passed is False


def test_missing_deformable_operator_generates_implementation_request() -> None:
    decision = ModelGraphDependencyGate.evaluate(
        component_id="neck.experimental_deformable",
        deformable_module="missing_yolo_agent_deformable_operator",
    )

    assert decision.execution_class == "implementation_request"
    assert decision.available is False
    assert decision.implementation_request is not None
    assert decision.implementation_request.missing_dependency == (
        "missing_yolo_agent_deformable_operator"
    )


def test_component_without_deformable_operator_is_runtime_candidate() -> None:
    decision = ModelGraphDependencyGate.evaluate(
        component_id="neck.multi_scale_fusion",
        deformable_module=None,
    )

    assert decision.execution_class == "runtime_candidate"
    assert decision.available is True
    assert decision.implementation_request is None
