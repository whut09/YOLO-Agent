"""Standalone execution and certification for inference-only paper policies."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Protocol

from yolo_agent.certification.inference_policy_schemas import (
    InferencePolicyCertificationReport,
)
from yolo_agent.components.adapters.inference.backend import (
    BackendResult,
    InferenceImage,
    UltralyticsInferenceBackend,
)
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


class InferencePolicyBackend(Protocol):
    def run(
        self,
        images: list[InferenceImage],
        protocol: Any,
        *,
        category_ids: list[int],
    ) -> BackendResult: ...


AccuracyEvaluator = Callable[[Path, list[dict[str, Any]]], dict[str, float]]


class InferencePolicyCertificationRunner:
    """Evaluate an inference policy without creating a training node."""

    def run(
        self,
        *,
        workdir: Path,
        model: str,
        images: Path,
        annotations: Path,
        config: InferencePolicyConfig,
        standard_metrics: dict[str, float] | Path | None = None,
        execute: bool = False,
        backend: InferencePolicyBackend | None = None,
        evaluator: AccuracyEvaluator | None = None,
    ) -> InferencePolicyCertificationReport:
        workdir.mkdir(parents=True, exist_ok=True)
        protocol = protocol_from_policy(config.model_copy(update={"model_path": model}))
        standard = _standard_metrics(standard_metrics)
        if not execute:
            return self._write_report(
                workdir,
                InferencePolicyCertificationReport(
                    status="skipped",
                    model=model,
                    annotations=str(annotations),
                    protocol=protocol,
                    protocol_hash=protocol.protocol_hash,
                    standard_640_metrics=standard,
                    reason="real inference policy execution requires explicit --execute",
                ),
            )
        try:
            inputs, category_ids = _coco_inputs(images, annotations)
            actual_backend = backend or UltralyticsInferenceBackend()
            backend_result = actual_backend.run(
                inputs, protocol, category_ids=category_ids
            )
            accuracy = (evaluator or _evaluate_coco)(annotations, backend_result.predictions)
            metrics = InferencePolicyMetrics(
                metric_namespace=protocol.metric_namespace,
                map50_95=_optional_float(accuracy.get("AP")),
                ap_small=_optional_float(accuracy.get("AP_small")),
                recall=_optional_float(accuracy.get("AR_100")),
                resources=InferenceResourceMetrics(
                    latency_ms=backend_result.latency_ms,
                    throughput=backend_result.throughput,
                    peak_vram_mb=backend_result.peak_vram_mb,
                ),
            )
            result = InferencePolicyResult(
                status="completed",
                protocol=protocol,
                metrics=metrics,
                predictions=backend_result.predictions,
                merge_statistics=backend_result.merge_statistics,
            )
            paths = write_inference_policy_artifacts(result, workdir / "artifacts")
            checks = {
                "backend_completed": True,
                "policy_metrics_complete": metrics.map50_95 is not None
                and metrics.ap_small is not None,
                "standard_metrics_unchanged": standard
                == _standard_metrics(standard_metrics),
                "training_attribution_isolated": True,
                "inference_policy_marked": protocol.inference_policy_changed,
                "standard_imgsz_fixed": protocol.config.standard_imgsz == 640,
                "one_to_one_nms_policy_valid": (
                    protocol.config.merge_policy != "nms"
                    or protocol.config.allow_cross_view_merge
                ),
            }
            report = InferencePolicyCertificationReport(
                status="passed" if all(checks.values()) else "failed",
                model=model,
                annotations=str(annotations),
                protocol=protocol,
                protocol_hash=protocol.protocol_hash,
                standard_640_metrics=standard,
                policy_metrics=metrics,
                checks=checks,
                artifacts={
                    "protocol": str(paths.protocol),
                    "predictions": str(paths.predictions),
                    "metrics": str(paths.metrics),
                    "resources": str(paths.resources),
                    "merge_statistics": str(paths.merge_statistics),
                    "report": str(workdir / "inference_policy_certification_report.yaml"),
                },
            )
        except Exception as exc:
            report = InferencePolicyCertificationReport(
                status="failed",
                model=model,
                annotations=str(annotations),
                protocol=protocol,
                protocol_hash=protocol.protocol_hash,
                standard_640_metrics=standard,
                reason=str(exc),
            )
        return self._write_report(workdir, report)

    @staticmethod
    def _write_report(
        workdir: Path, report: InferencePolicyCertificationReport
    ) -> InferencePolicyCertificationReport:
        report.write(workdir / "inference_policy_certification_report.yaml")
        return report


def _coco_inputs(images: Path, annotations: Path) -> tuple[list[InferenceImage], list[int]]:
    payload = json.loads(annotations.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("COCO annotations must contain a mapping")
    records = payload.get("images", [])
    categories = payload.get("categories", [])
    if not isinstance(records, list) or not records:
        raise ValueError("COCO annotations contain no images")
    category_ids = sorted(
        int(item["id"])
        for item in categories
        if isinstance(item, dict) and "id" in item
    )
    if not category_ids:
        raise ValueError("COCO annotations contain no categories")
    output = [
        InferenceImage(image_id=int(item["id"]), path=images / str(item["file_name"]))
        for item in records
        if isinstance(item, dict) and "id" in item and "file_name" in item
    ]
    missing = [str(item.path) for item in output if not item.path.is_file()]
    if missing:
        raise FileNotFoundError("COCO images are missing: " + ", ".join(missing[:3]))
    return output, category_ids


def _evaluate_coco(
    annotations: Path, predictions: list[dict[str, Any]]
) -> dict[str, float]:
    from yolo_agent.adapters.ultralytics.coco_post_eval import write_coco_eval_report

    with tempfile.TemporaryDirectory(prefix="yolo-agent-inference-policy-") as directory:
        root = Path(directory)
        predictions_path = root / "predictions.json"
        report_path = root / "coco_eval.json"
        predictions_path.write_text(
            json.dumps(predictions, sort_keys=True), encoding="utf-8"
        )
        write_coco_eval_report(
            annotations_path=annotations,
            predictions_path=predictions_path,
            output_path=report_path,
        )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        str(key): float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def _standard_metrics(value: dict[str, float] | Path | None) -> dict[str, float]:
    if value is None:
        return {}
    payload: Any = (
        json.loads(value.read_text(encoding="utf-8-sig"))
        if isinstance(value, Path)
        else value
    )
    if not isinstance(payload, dict):
        raise ValueError("standard metrics must be a JSON mapping")
    policy_prefixes = (
        "sliced_",
        "tiled_multi_scale_",
        "tta_",
        "calibrated_",
        "class_threshold_",
        "merged_",
    )
    if any(str(name).startswith(policy_prefixes) for name in payload):
        raise ValueError("standard metrics input cannot contain inference policy metrics")
    return {
        str(name): float(metric)
        for name, metric in payload.items()
        if isinstance(metric, (int, float))
    }


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


__all__ = [
    "AccuracyEvaluator",
    "InferencePolicyBackend",
    "InferencePolicyCertificationRunner",
]
