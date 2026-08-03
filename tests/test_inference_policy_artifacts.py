import json
from pathlib import Path

from yolo_agent.components.adapters.inference.policy import (
    InferencePolicyConfig,
    InferencePolicyMetrics,
    InferencePolicyResult,
    InferenceResourceMetrics,
    protocol_from_policy,
)
from yolo_agent.components.adapters.inference.policy_artifacts import (
    write_inference_policy_artifacts,
)


def test_policy_artifacts_preserve_namespace_and_resources(tmp_path: Path) -> None:
    protocol = protocol_from_policy(
        InferencePolicyConfig(
            policy_id="tta",
            kind="test_time_augmentation",
            scales=[0.8, 1.0],
        )
    )
    result = InferencePolicyResult(
        status="completed",
        protocol=protocol,
        predictions=[{"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 0.8}],
        metrics=InferencePolicyMetrics(
            metric_namespace="tta_inference",
            map50_95=0.4,
            ap_small=0.2,
            resources=InferenceResourceMetrics(
                latency_ms=20.0,
                throughput=50.0,
                peak_vram_mb=256.0,
            ),
        ),
        merge_statistics={"merge_policy": "none", "input_count": 1, "output_count": 1},
    )

    paths = write_inference_policy_artifacts(result, tmp_path)
    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))

    assert paths.protocol.name == "tta_protocol.json"
    assert metrics["tta_map50_95"] == 0.4
    assert metrics["tta_latency_ms"] == 20.0
    assert "map50_95" not in metrics
    assert json.loads(paths.resources.read_text(encoding="utf-8"))["peak_vram_mb"] == 256.0
