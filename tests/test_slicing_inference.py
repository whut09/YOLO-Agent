from pathlib import Path

from yolo_agent.agents.pareto import ParetoSelector, candidate_metrics_from_row
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.inference.slicing import (
    SlicingInferenceAdapter,
    SlicingInferenceConfig,
    SlicingInferenceRunner,
    metric_evidence_from_result,
    protocol_from_config,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.recipes.registry import RecipeRegistry


def _context(tmp_path: Path, **options):
    return AdapterContext(
        contract=ComponentContract(
            component_id="inference.sahi_slicing", display_name="SAHI slicing", category="slicing",
            implementation_path="local", adapter_class="SlicingInferenceAdapter",
            inference_only=True, changes_model_graph=False, fixed_imgsz_compatible=True,
        ),
        detector_family="yolo26", head="one_to_one", workspace=tmp_path, options=options,
    )


def test_optional_sahi_missing_returns_skip(monkeypatch) -> None:
    monkeypatch.setattr(SlicingInferenceRunner, "sahi_available", staticmethod(lambda: False))
    result = SlicingInferenceRunner().run(["image.jpg"], SlicingInferenceConfig())
    assert result.status == "skipped"
    assert "not installed" in result.reason


def test_mock_backend_records_protocol_and_sliced_namespace() -> None:
    def backend(images, protocol):
        assert protocol.slice_width == 512
        assert protocol.merge_policy == "nmm"
        return ["prediction"], {
            "sliced_map50_95": 0.42,
            "sliced_ap_small": 0.26,
            "sliced_latency_ms": 40.0,
            "sliced_throughput": 25.0,
        }

    result = SlicingInferenceRunner(backend).run(
        ["image.jpg"],
        SlicingInferenceConfig(slice_height=512, slice_width=512, overlap_height_ratio=0.25, overlap_width_ratio=0.25, merge_policy="nmm"),
    )
    assert result.status == "completed"
    assert result.protocol.inference_policy_changed
    assert result.protocol.extra_nms_applied is False
    evidence = metric_evidence_from_result(result, candidate_id="candidate", node_id="node", dataset_version="coco-sha")
    assert {item.metric_name for item in evidence} == {"sliced_map50_95", "sliced_ap_small", "sliced_latency_ms", "sliced_throughput"}
    assert all(item.metric_name not in {"map50_95", "latency_ms"} for item in evidence)
    latency = next(item for item in evidence if item.metric_name == "sliced_latency_ms")
    assert latency.higher_is_better is False


def test_one_to_one_does_not_add_nms_unless_requested() -> None:
    standard = protocol_from_config(SlicingInferenceConfig(one_to_one_head=True))
    merged = protocol_from_config(SlicingInferenceConfig(one_to_one_head=True, merge_policy="nms"))
    assert standard.merge_policy == "none" and not standard.extra_nms_applied
    assert merged.extra_nms_applied


def test_adapter_patch_is_inference_only(tmp_path: Path) -> None:
    preview = SlicingInferenceAdapter().prepare_patch({}, {"epochs": 10}, _context(tmp_path, slice_height=512, slice_width=512))
    assert preview.patched_training_config["epochs"] == 10
    assert preview.patched_training_config["inference_policy"]["inference_policy_changed"] is True
    assert preview.patched_model_config == {}


def test_protocol_is_written_atomically(tmp_path: Path) -> None:
    protocol = protocol_from_config(SlicingInferenceConfig(slice_height=512, slice_width=512, merge_policy="nms"))
    path = protocol.write(tmp_path / "protocol.json")
    text = path.read_text(encoding="utf-8")
    assert '"slice_height": 512' in text
    assert '"extra_nms_applied": true' in text


def test_pareto_includes_slicing_and_marks_policy_change() -> None:
    standard = candidate_metrics_from_row({"id": "standard", "base_model": "yolo26n", "has_evidence": True, "metrics": {"map50_95": 0.38, "latency_ms": 8.0}})
    sliced = candidate_metrics_from_row({"id": "sliced", "base_model": "yolo26n", "has_evidence": True, "metrics": {"map50_95": 0.38, "latency_ms": 8.0, "sliced_map50_95": 0.43, "sliced_latency_ms": 35.0}})
    assert standard is not None and sliced is not None
    front = ParetoSelector().select([standard, sliced])
    assert {point.candidate_id for point in front.points} == {"standard", "sliced"}
    assert next(point for point in front.points if point.candidate_id == "sliced").inference_policy_changed


def test_recipe_is_inference_only_and_fixed_640() -> None:
    recipe = RecipeRegistry.from_path(Path("configs/recipes/sahi_inference.yaml")).get("sahi_slicing_inference")
    assert recipe is not None
    assert recipe.primary_changed_variable == "inference_policy"
    assert recipe.fixed_variables["imgsz"] == 640
    assert recipe.inference_actions == ["sahi_slicing"]
