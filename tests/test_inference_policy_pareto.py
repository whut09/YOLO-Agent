from yolo_agent.agents.pareto import (
    ParetoSelector,
    candidate_metric_variants_from_row,
)


def test_standard_sliced_and_tta_metrics_form_independent_fronts() -> None:
    row = {
        "id": "candidate",
        "base_model": "yolo26n.pt",
        "has_evidence": True,
        "metrics": {
            "map50_95": 0.40,
            "latency_ms": 10.0,
            "sliced_map50_95": 0.45,
            "sliced_latency_ms": 30.0,
            "tta_map50_95": 0.43,
            "tta_latency_ms": 20.0,
            "inference_policy_changed": True,
        },
    }

    variants = candidate_metric_variants_from_row(row)
    front = ParetoSelector().select_partitioned(variants)

    assert {item.metric_namespace for item in variants} == {
        "standard_640",
        "sliced_inference",
        "tta_inference",
    }
    assert len(front.standard_640.points) == 1
    assert len(front.sliced_inference.points) == 1
    assert len(front.tta_inference.points) == 1
    assert front.tiled_multi_scale_inference.points == []


def test_all_inference_policy_namespaces_are_partitioned() -> None:
    prefixes = {
        "tiled_multi_scale": "tiled_multi_scale_inference",
        "calibrated": "calibrated_inference",
        "class_threshold": "class_threshold_inference",
        "merged": "merged_inference",
    }
    rows = []
    for prefix in prefixes:
        rows.extend(
            candidate_metric_variants_from_row(
                {
                    "id": prefix,
                    "has_evidence": True,
                    "metrics": {
                        f"{prefix}_map50_95": 0.4,
                        f"{prefix}_latency_ms": 12.0,
                        "inference_policy_changed": True,
                    },
                }
            )
        )

    front = ParetoSelector().select_partitioned(rows)

    for namespace in prefixes.values():
        assert len(getattr(front, namespace).points) == 1
