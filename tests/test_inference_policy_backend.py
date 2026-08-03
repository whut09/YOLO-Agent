from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from yolo_agent.components.adapters.inference.backend import (
    InferenceImage,
    UltralyticsInferenceBackend,
)
from yolo_agent.components.adapters.inference.policy import (
    InferencePolicyConfig,
    protocol_from_policy,
)


class _Model:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        boxes = SimpleNamespace(xyxy=[[1.0, 2.0, 11.0, 12.0]], conf=[0.8], cls=[0.0])
        return [SimpleNamespace(boxes=boxes)]


def _image(tmp_path: Path, *, width: int = 64, height: int = 64) -> InferenceImage:
    path = tmp_path / "image.jpg"
    Image.new("RGB", (width, height), "white").save(path)
    return InferenceImage(image_id=7, path=path)


def test_calibration_calls_real_predict_at_standard_640(tmp_path: Path) -> None:
    model = _Model()
    backend = UltralyticsInferenceBackend(lambda _: model)
    protocol = protocol_from_policy(
        InferencePolicyConfig(
            policy_id="temperature-2",
            kind="confidence_calibration",
            model_path="model.pt",
            temperature=2.0,
        )
    )

    result = backend.run([_image(tmp_path)], protocol, category_ids=[42])

    assert model.calls[0]["imgsz"] == 640
    assert result.predictions[0]["category_id"] == 42
    assert result.predictions[0]["score"] < 0.8


def test_tta_executes_scale_and_flip_views(tmp_path: Path) -> None:
    model = _Model()
    backend = UltralyticsInferenceBackend(lambda _: model)
    protocol = protocol_from_policy(
        InferencePolicyConfig(
            policy_id="tta",
            kind="test_time_augmentation",
            model_path="model.pt",
            scales=[0.5, 1.0],
            horizontal_flip=True,
        )
    )

    result = backend.run([_image(tmp_path)], protocol, category_ids=[1])

    assert [call["imgsz"] for call in model.calls] == [320, 640, 640]
    assert len(result.predictions) == 3
    assert result.predictions[-1]["bbox"][0] == 53.0


def test_tiled_multiscale_maps_tiles_to_original_coordinates(tmp_path: Path) -> None:
    model = _Model()
    backend = UltralyticsInferenceBackend(lambda _: model)
    protocol = protocol_from_policy(
        InferencePolicyConfig(
            policy_id="tiles",
            kind="tiled_multi_scale",
            model_path="model.pt",
            tile_sizes=[32],
            overlap_ratio=0.0,
        )
    )

    result = backend.run(
        [_image(tmp_path, width=64, height=64)], protocol, category_ids=[1]
    )

    assert len(model.calls) == 4
    assert {tuple(item["bbox"][:2]) for item in result.predictions} == {
        (1.0, 2.0),
        (33.0, 2.0),
        (1.0, 34.0),
        (33.0, 34.0),
    }


def test_class_threshold_policy_filters_predictions(tmp_path: Path) -> None:
    model = _Model()
    backend = UltralyticsInferenceBackend(lambda _: model)
    protocol = protocol_from_policy(
        InferencePolicyConfig(
            policy_id="class-threshold",
            kind="class_aware_thresholding",
            model_path="model.pt",
            class_thresholds={42: 0.9},
        )
    )

    result = backend.run([_image(tmp_path)], protocol, category_ids=[42])

    assert result.predictions == []


def test_merge_variant_executes_fixed_multi_view_source(tmp_path: Path) -> None:
    model = _Model()
    backend = UltralyticsInferenceBackend(lambda _: model)
    protocol = protocol_from_policy(
        InferencePolicyConfig(
            policy_id="nmm",
            kind="merge_policy",
            model_path="model.pt",
            scales=[0.8, 1.0],
            merge_policy="nmm",
            allow_cross_view_merge=True,
        )
    )

    result = backend.run([_image(tmp_path)], protocol, category_ids=[1])

    assert [call["imgsz"] for call in model.calls] == [512, 640]
    assert result.merge_statistics["input_count"] == 2
    assert result.merge_statistics["output_count"] == 1
