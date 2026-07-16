from pathlib import Path

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.sampling.small_object_sampling import SmallObjectSample, SmallObjectSampler, SmallObjectSamplingAdapter
from yolo_agent.components.contracts import ComponentContract


def test_sampler_boosts_small_objects_and_bounds_weights() -> None:
    values, manifest = SmallObjectSampler().weights([
        SmallObjectSample(image_path="small.jpg", normalized_areas=[0.005], class_ids=[1]),
        SmallObjectSample(image_path="large.jpg", normalized_areas=[0.2], class_ids=[1]),
    ])
    assert values[0] > values[1]
    assert max(values) <= 3.0
    assert manifest.small_image_count == 1
    assert manifest.val_unchanged


def test_validation_samples_are_not_resampled() -> None:
    values, manifest = SmallObjectSampler().weights([
        SmallObjectSample(image_path="train.jpg", split="train", normalized_areas=[0.01], class_ids=[1]),
        SmallObjectSample(image_path="val.jpg", split="val", normalized_areas=[0.001], class_ids=[1]),
    ])
    assert len(values) == 1
    assert manifest.val_unchanged
    assert "val.jpg" not in manifest.weights


def test_sampler_adapter_is_dry_run_safe(tmp_path: Path) -> None:
    context = AdapterContext(contract=ComponentContract(
        component_id="sampling.small_object", display_name="Sampler", category="sampling",
        implementation_path="local", adapter_class="SmallObjectSamplingAdapter", fixed_imgsz_compatible=True,
    ), workspace=tmp_path)
    adapter = SmallObjectSamplingAdapter()
    preview = adapter.prepare_patch({}, {}, context)
    assert preview.operations[0].field == "data_sampling"
    assert adapter.smoke_test(context).passed
