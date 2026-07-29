from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from yolo_agent.components.adapters.inference.sahi_backend import (
    SahiSlicingBackend,
    SlicingImage,
)
from yolo_agent.components.adapters.inference.slicing import (
    SlicingInferenceConfig,
    protocol_from_config,
)


class _Prediction:
    bbox = SimpleNamespace(to_xywh=lambda: [1, 2, 3, 4])
    score = SimpleNamespace(value=0.75)
    category = SimpleNamespace(id=7)


def _slice_image(**kwargs):  # type: ignore[no-untyped-def]
    assert kwargs["auto_slice_resolution"] is False
    return SimpleNamespace(
        images=["tile-a", "tile-b"],
        starting_pixels=[[0, 0], [320, 0]],
        original_image_height=480,
        original_image_width=960,
    )


def test_backend_binds_model_and_keeps_default_one_to_one_path_nms_free(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    calls: dict[str, object] = {"predictions": []}

    def model_factory(**kwargs):  # type: ignore[no-untyped-def]
        calls["model"] = kwargs
        return "bound-model"

    def predict(**kwargs):  # type: ignore[no-untyped-def]
        calls["predictions"].append(kwargs)  # type: ignore[union-attr]
        return SimpleNamespace(object_prediction_list=[_Prediction()])

    def forbidden_postprocess(_protocol):  # type: ignore[no-untyped-def]
        raise AssertionError("default one-to-one slicing must not add NMS")

    protocol = protocol_from_config(
        SlicingInferenceConfig(model_path="yolo26n.pt", merge_policy="none")
    )
    predictions, metrics = SahiSlicingBackend(
        model_factory=model_factory,
        slice_image_fn=_slice_image,
        prediction_fn=predict,
        postprocess_factory=forbidden_postprocess,
    )([SlicingImage(image_id=42, path=image)], protocol)

    assert calls["model"] == {
        "model_type": "ultralytics",
        "model_path": "yolo26n.pt",
        "confidence_threshold": 0.001,
        "device": "cpu",
    }
    assert [item["image_id"] for item in predictions] == [42, 42]
    assert predictions[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert metrics["sliced_latency_ms"] is not None
    assert metrics["sliced_throughput"] is not None


def test_explicit_cross_slice_merge_invokes_requested_postprocessor(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    merged: list[int] = []

    def postprocess_factory(protocol):  # type: ignore[no-untyped-def]
        assert protocol.merge_policy == "nms"

        def postprocess(predictions):  # type: ignore[no-untyped-def]
            merged.append(len(predictions))
            return predictions[:1]

        return postprocess

    backend = SahiSlicingBackend(
        model_factory=lambda **_: "model",
        slice_image_fn=_slice_image,
        prediction_fn=lambda **_: SimpleNamespace(object_prediction_list=[_Prediction()]),
        postprocess_factory=postprocess_factory,
    )
    protocol = protocol_from_config(
        SlicingInferenceConfig(model_path="yolo26n.pt", merge_policy="nms")
    )
    predictions, _ = backend([SlicingImage(image_id=5, path=image)], protocol)

    assert merged == [2]
    assert len(predictions) == 1
    assert protocol.extra_nms_applied is True


def test_backend_requires_bound_model() -> None:
    backend = SahiSlicingBackend(
        model_factory=lambda **_: "model",
        slice_image_fn=_slice_image,
        prediction_fn=lambda **_: SimpleNamespace(object_prediction_list=[]),
    )
    with pytest.raises(ValueError, match="model_path"):
        backend([], protocol_from_config(SlicingInferenceConfig()))
