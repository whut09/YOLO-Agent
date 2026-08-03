from pathlib import Path

import pytest

from yolo_agent.components.adapters.inference.policy import (
    InferencePolicyConfig,
    InferencePolicyMetrics,
    InferenceResourceMetrics,
    protocol_from_policy,
)


def test_protocol_is_inference_only_and_hash_stable(tmp_path: Path) -> None:
    config = InferencePolicyConfig(
        policy_id="tta-default",
        kind="test_time_augmentation",
        scales=[0.8, 1.0, 1.2],
        horizontal_flip=True,
    )
    first = protocol_from_policy(config)
    second = protocol_from_policy(config)

    assert first.protocol_hash == second.protocol_hash
    assert first.training_attribution_allowed is False
    assert first.config.standard_imgsz == 640
    assert first.write(tmp_path / "protocol.json").is_file()


def test_one_to_one_merge_requires_explicit_cross_view_opt_in() -> None:
    with pytest.raises(ValueError, match="allow_cross_view_merge"):
        InferencePolicyConfig(
            policy_id="bad-nms",
            kind="merge_policy",
            merge_policy="nms",
        )


def test_policy_metrics_publish_only_namespaced_values() -> None:
    metrics = InferencePolicyMetrics(
        metric_namespace="tta_inference",
        map50_95=0.41,
        ap_small=0.22,
        resources=InferenceResourceMetrics(
            latency_ms=12.5,
            throughput=80.0,
            peak_vram_mb=512.0,
        ),
    )

    values = metrics.namespaced()

    assert values["tta_map50_95"] == 0.41
    assert values["tta_ap_small"] == 0.22
    assert values["tta_latency_ms"] == 12.5
    assert "map50_95" not in values
    assert "latency_ms" not in values


def test_calibration_and_class_threshold_inputs_must_change_policy() -> None:
    with pytest.raises(ValueError, match="non-neutral"):
        InferencePolicyConfig(
            policy_id="neutral",
            kind="confidence_calibration",
        )
    with pytest.raises(ValueError, match="class_thresholds"):
        InferencePolicyConfig(
            policy_id="empty",
            kind="class_aware_thresholding",
        )
