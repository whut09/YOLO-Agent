from pathlib import Path

import pytest

from yolo_agent.agents.pareto import ParetoSelector, candidate_metrics_from_row
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.inference.slicing import (
    SahiInferenceRuntimePlugin,
    SahiRuntimeEvidence,
    SlicingInferenceAdapter,
    SlicingInferenceConfig,
    SlicingInferenceRunner,
    metric_evidence_from_result,
    policy_config_from_slicing,
    protocol_from_config,
)
from yolo_agent.components.adapters.inference.plugin import InferencePolicyPlugin
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


def test_sahi_is_a_shared_inference_policy_specialization() -> None:
    config = policy_config_from_slicing(
        SlicingInferenceConfig(
            slice_height=512,
            slice_width=640,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.25,
            merge_policy="nmm",
        )
    )

    assert issubclass(SahiInferenceRuntimePlugin, InferencePolicyPlugin)
    assert config.kind == "sahi_slicing"
    assert config.metric_namespace == "sliced_inference"
    assert config.tile_sizes == [512, 640]
    assert config.allow_cross_view_merge is True


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
    assert "inference_policy" not in preview.patched_training_config
    assert preview.patched_model_config == {}
    assert {item.name for item in preview.expected_artifacts} == {
        "slicing_inference_protocol",
        "sliced_predictions",
        "sliced_metrics",
        "sahi_certification_report",
        "sahi_runtime_evidence",
    }


def test_sahi_runtime_plugin_fails_closed_without_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SlicingInferenceAdapter()
    payload = adapter.build_runtime_payload(
        _context(tmp_path),
        protocol_hash="sahi-protocol",
        base_command=["yolo-agent", "advanced", "certify-sahi", "--execute"],
        generated_config={},
    )
    reference = payload.inference_plugin[0]
    plugin = reference.resolve()(**reference.options)
    monkeypatch.setattr(SlicingInferenceRunner, "sahi_available", staticmethod(lambda: False))

    with pytest.raises(RuntimeError, match="not installed"):
        plugin.prepare_command(
            payload=payload,
            command=payload.base_command,
            env={},
        )
    assert not (tmp_path / "sahi_runtime_evidence.json").exists()


def test_sahi_runtime_plugin_records_payload_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SlicingInferenceAdapter()
    payload = adapter.build_runtime_payload(
        _context(tmp_path, slice_height=512, slice_width=512),
        protocol_hash="sahi-protocol",
        base_command=["yolo-agent", "advanced", "certify-sahi", "--execute"],
        generated_config={},
    )
    reference = payload.inference_plugin[0]
    plugin = reference.resolve()(**reference.options)
    monkeypatch.setattr(SlicingInferenceRunner, "sahi_available", staticmethod(lambda: True))
    monkeypatch.setattr("importlib.metadata.version", lambda _: "0.11.test")

    command, _ = plugin.prepare_command(
        payload=payload,
        command=payload.base_command,
        env={},
    )

    assert command == payload.base_command
    evidence = SahiRuntimeEvidence.model_validate_json(
        (tmp_path / "sahi_runtime_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence.payload_hash == payload.payload_hash
    assert evidence.changed_variables == payload.changed_variables
    assert evidence.hook_call_counts == {"prepare_command": 1}
    assert evidence.training_attribution_allowed is False


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
