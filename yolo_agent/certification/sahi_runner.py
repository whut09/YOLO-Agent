"""Standalone SAHI inference certification runner."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from yolo_agent.certification.sahi_schemas import (
    SahiCertificationReport,
    SahiDependencyStatus,
    protocol_hash,
)
from yolo_agent.components.adapters.inference.artifacts import write_slicing_artifacts
from yolo_agent.components.adapters.inference.sahi_backend import SlicingImage
from yolo_agent.components.adapters.inference.sliced_coco import evaluate_sliced_coco
from yolo_agent.components.adapters.inference.slicing import (
    SlicingBackend,
    SlicingInferenceConfig,
    SlicingInferenceMetrics,
    SlicingInferenceRunner,
    protocol_from_config,
)


class SahiInferenceCertificationRunner:
    """Certify an inference policy without creating a training experiment."""

    def run(
        self,
        *,
        workdir: Path,
        model: str,
        images: Path,
        annotations: Path,
        config: SlicingInferenceConfig,
        standard_metrics: dict[str, float] | Path | None = None,
        execute: bool = False,
        backend: SlicingBackend | None = None,
    ) -> SahiCertificationReport:
        workdir.mkdir(parents=True, exist_ok=True)
        protocol = protocol_from_config(
            config.model_copy(update={"model_path": model})
        )
        dependency = _dependency_status(backend)
        standard = _standard_metrics(standard_metrics)
        if not execute:
            return self._write_report(
                workdir,
                SahiCertificationReport(
                    status="skipped",
                    model=model,
                    annotations=str(annotations),
                    protocol=protocol,
                    protocol_hash=protocol_hash(protocol),
                    dependency=dependency,
                    standard_640_metrics=standard,
                    reason="real SAHI inference requires explicit --execute",
                ),
            )
        if backend is None and not dependency.available:
            return self._write_report(
                workdir,
                SahiCertificationReport(
                    status="skipped",
                    model=model,
                    annotations=str(annotations),
                    protocol=protocol,
                    protocol_hash=protocol_hash(protocol),
                    dependency=dependency,
                    standard_640_metrics=standard,
                    reason=dependency.reason,
                ),
            )
        try:
            inputs = _coco_images(images, annotations)
            result = SlicingInferenceRunner(backend).run(inputs, config.model_copy(update={"model_path": model}))
            if result.status != "completed" or result.metrics is None:
                raise RuntimeError(result.reason or f"slicing inference ended with {result.status}")
            initial_paths = write_slicing_artifacts(result, workdir / "artifacts")
            accuracy = evaluate_sliced_coco(
                annotations_path=annotations,
                predictions_path=initial_paths.predictions,
            )
            metrics = SlicingInferenceMetrics(
                sliced_map50_95=accuracy.sliced_map50_95,
                sliced_ap_small=accuracy.sliced_ap_small,
                sliced_latency_ms=result.metrics.sliced_latency_ms,
                sliced_throughput=result.metrics.sliced_throughput,
            )
            result = result.model_copy(update={"metrics": metrics})
            paths = write_slicing_artifacts(result, workdir / "artifacts")
            checks = {
                "slicing_completed": True,
                "required_sliced_metrics_complete": all(
                    value is not None
                    for value in metrics.model_dump(exclude={"inference_policy_changed"}).values()
                ),
                "standard_metrics_unchanged": standard == _standard_metrics(standard_metrics),
                "training_attribution_isolated": True,
                "inference_policy_marked": result.protocol.inference_policy_changed,
                "one_to_one_nms_policy_valid": (
                    not result.protocol.extra_nms_applied
                    if result.protocol.merge_policy != "nms"
                    else result.protocol.extra_nms_applied
                ),
            }
            report = SahiCertificationReport(
                status="passed" if all(checks.values()) else "failed",
                model=model,
                annotations=str(annotations),
                protocol=result.protocol,
                protocol_hash=protocol_hash(result.protocol),
                dependency=dependency,
                standard_640_metrics=standard,
                sliced_inference_metrics=metrics,
                checks=checks,
                artifacts={
                    "protocol": str(paths.protocol),
                    "predictions": str(paths.predictions),
                    "metrics": str(paths.metrics),
                    "report": str(workdir / "sahi_certification_report.yaml"),
                },
            )
        except Exception as exc:
            report = SahiCertificationReport(
                status="failed",
                model=model,
                annotations=str(annotations),
                protocol=protocol,
                protocol_hash=protocol_hash(protocol),
                dependency=dependency,
                standard_640_metrics=standard,
                reason=str(exc),
            )
        return self._write_report(workdir, report)

    @staticmethod
    def _write_report(workdir: Path, report: SahiCertificationReport) -> SahiCertificationReport:
        report.write(workdir / "sahi_certification_report.yaml")
        return report


def _dependency_status(backend: SlicingBackend | None) -> SahiDependencyStatus:
    if backend is not None:
        return SahiDependencyStatus(available=True, version="injected-test-backend")
    if not SlicingInferenceRunner.sahi_available():
        return SahiDependencyStatus(
            available=False,
            reason="optional dependency 'sahi' is not installed",
        )
    try:
        version = importlib.metadata.version("sahi")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return SahiDependencyStatus(available=True, version=version)


def _standard_metrics(value: dict[str, float] | Path | None) -> dict[str, float]:
    if value is None:
        return {}
    payload: Any = json.loads(value.read_text(encoding="utf-8-sig")) if isinstance(value, Path) else value
    if not isinstance(payload, dict):
        raise ValueError("standard metrics must be a JSON mapping")
    if any(str(name).startswith("sliced_") for name in payload):
        raise ValueError("standard metrics input cannot contain sliced metrics")
    return {
        str(name): float(metric)
        for name, metric in payload.items()
        if isinstance(metric, (int, float))
    }


def _coco_images(images: Path, annotations: Path) -> list[SlicingImage]:
    payload = json.loads(annotations.read_text(encoding="utf-8-sig"))
    records = payload.get("images", []) if isinstance(payload, dict) else []
    if not isinstance(records, list) or not records:
        raise ValueError("COCO annotations contain no images")
    output = [
        SlicingImage(image_id=int(item["id"]), path=images / str(item["file_name"]))
        for item in records
        if isinstance(item, dict) and "id" in item and "file_name" in item
    ]
    missing = [str(item.path) for item in output if not item.path.is_file()]
    if missing:
        raise FileNotFoundError("COCO images are missing: " + ", ".join(missing[:3]))
    return output


__all__ = ["SahiInferenceCertificationRunner"]
