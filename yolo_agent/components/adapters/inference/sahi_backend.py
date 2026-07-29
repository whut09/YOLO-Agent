"""Optional, model-bound SAHI slicing backend."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from yolo_agent.components.adapters.inference.slicing import SlicingInferenceProtocol


class SlicingImage(BaseModel):
    """One image with the stable COCO identifier used by evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_id: int
    path: Path


class SahiSlicingBackend:
    """Run SAHI slices without changing the detector's standard inference path."""

    def __init__(
        self,
        *,
        model_factory: Callable[..., Any] | None = None,
        slice_image_fn: Callable[..., Any] | None = None,
        prediction_fn: Callable[..., Any] | None = None,
        postprocess_factory: Callable[[SlicingInferenceProtocol], Any] | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._slice_image_fn = slice_image_fn
        self._prediction_fn = prediction_fn
        self._postprocess_factory = postprocess_factory

    def __call__(
        self,
        images: list[Any],
        protocol: SlicingInferenceProtocol,
    ) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
        if not protocol.model_path:
            raise ValueError("SAHI execution requires model_path")
        normalized = [_normalize_image(item, index) for index, item in enumerate(images)]
        model_factory, slice_image_fn, prediction_fn = self._resolve_api()
        detection_model = model_factory(
            model_type=protocol.model_type,
            model_path=protocol.model_path,
            confidence_threshold=protocol.confidence_threshold,
            device=protocol.device,
        )
        predictions: list[dict[str, Any]] = []
        started = time.perf_counter()
        for image in normalized:
            sliced = slice_image_fn(
                image=str(image.path),
                slice_height=protocol.slice_height,
                slice_width=protocol.slice_width,
                overlap_height_ratio=protocol.overlap_height_ratio,
                overlap_width_ratio=protocol.overlap_width_ratio,
                auto_slice_resolution=False,
            )
            object_predictions: list[Any] = []
            for tile, shift in zip(sliced.images, sliced.starting_pixels, strict=True):
                result = prediction_fn(
                    image=tile,
                    detection_model=detection_model,
                    shift_amount=list(shift),
                    full_shape=[sliced.original_image_height, sliced.original_image_width],
                    postprocess=None,
                    verbose=0,
                )
                object_predictions.extend(result.object_prediction_list)
            if protocol.merge_policy != "none":
                object_predictions = list(self._postprocess(protocol)(object_predictions))
            predictions.extend(_to_coco_prediction(image.image_id, item) for item in object_predictions)
        elapsed = time.perf_counter() - started
        image_count = max(len(normalized), 1)
        return predictions, {
            "sliced_latency_ms": elapsed * 1000.0 / image_count,
            "sliced_throughput": len(normalized) / elapsed if elapsed > 0 else 0.0,
        }

    def _resolve_api(self) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        if self._model_factory and self._slice_image_fn and self._prediction_fn:
            return self._model_factory, self._slice_image_fn, self._prediction_fn
        try:
            from sahi import AutoDetectionModel  # type: ignore[import-not-found]
            from sahi.predict import get_prediction  # type: ignore[import-not-found]
            from sahi.slicing import slice_image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("optional dependency 'sahi' is not installed") from exc
        return AutoDetectionModel.from_pretrained, slice_image, get_prediction

    def _postprocess(self, protocol: SlicingInferenceProtocol) -> Any:
        if self._postprocess_factory is not None:
            return self._postprocess_factory(protocol)
        try:
            from sahi.postprocess.combine import (  # type: ignore[import-not-found]
                NMMPostprocess,
                NMSPostprocess,
            )
        except ImportError as exc:
            raise RuntimeError("installed SAHI does not expose merge postprocessors") from exc
        postprocess_type = NMSPostprocess if protocol.merge_policy == "nms" else NMMPostprocess
        return postprocess_type(
            match_threshold=protocol.merge_match_threshold,
            match_metric=protocol.merge_match_metric.upper(),
            class_agnostic=False,
        )


def _normalize_image(value: Any, index: int) -> SlicingImage:
    if isinstance(value, SlicingImage):
        return value
    if isinstance(value, dict):
        return SlicingImage.model_validate(value)
    return SlicingImage(image_id=index, path=Path(value))


def _to_coco_prediction(image_id: int, prediction: Any) -> dict[str, Any]:
    bbox = prediction.bbox.to_xywh()
    score = prediction.score.value
    category_id = prediction.category.id
    return {
        "image_id": image_id,
        "category_id": int(category_id),
        "bbox": [float(value) for value in bbox],
        "score": float(score),
    }


__all__ = ["SahiSlicingBackend", "SlicingImage"]
