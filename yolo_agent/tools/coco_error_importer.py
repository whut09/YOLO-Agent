"""Import official COCO eval metrics as candidate/node-level evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import MetricValue


COCO_STATS_METRICS = {
    0: "coco_ap50_95",
    1: "coco_ap50",
    2: "coco_ap75",
    3: "ap_small",
    4: "ap_medium",
    5: "ap_large",
    6: "coco_ar_max1",
    7: "coco_ar_max10",
    8: "coco_ar_max100",
    9: "ar_small",
    10: "ar_medium",
    11: "ar_large",
}

COCO_KEY_ALIASES = {
    "AP": "coco_ap50_95",
    "AP50": "coco_ap50",
    "AP75": "coco_ap75",
    "AP_small": "ap_small",
    "AP_medium": "ap_medium",
    "AP_large": "ap_large",
    "AR_small": "ar_small",
    "AR_medium": "ar_medium",
    "AR_large": "ar_large",
    "map": "coco_ap50_95",
    "map50": "coco_ap50",
    "map75": "coco_ap75",
    "map_small": "ap_small",
    "map_medium": "ap_medium",
    "map_large": "ap_large",
}


class CocoEvalImportResult(BaseModel):
    """Result of importing COCO eval metrics."""

    run_id: str
    candidate_id: str
    node_id: str
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    metrics_by_node_path: Path

    @field_serializer("metrics_by_node_path")
    def serialize_path(self, value: Path) -> str:
        """Serialize paths portably."""
        return value.as_posix()


def import_coco_eval_metrics(
    eval_path: Path | str,
    evidence_store: EvidenceStore,
    run_id: str,
    candidate_id: str,
    node_id: str,
    dataset_version: str = "coco2017",
    split: str = "val2017",
    source: str = "coco_eval_importer",
    verified: bool = True,
) -> CocoEvalImportResult:
    """Parse a COCO eval file and write node-level metric evidence.

    Supported inputs are intentionally conservative:
    - JSON with a pycocotools-style ``stats`` array.
    - JSON/YAML-like mappings with COCO metric keys such as ``AP_small``.
    - JSON with ``per_class_ap`` as a mapping or list.
    - Plain text logs containing COCO summary lines.
    """
    path = Path(eval_path)
    metrics = parse_coco_eval_metrics(path)
    metrics_path = evidence_store.log_candidate_metrics(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        metrics=metrics,
        dataset_version=dataset_version,
        split=split,
        source=source,
        verified=verified,
        validator="coco_error_importer",
        source_artifact=path,
    )
    evidence_store.log_artifact_manifest(
        run_id=run_id,
        name=f"{node_id}_coco_eval",
        artifact_path=path,
        producer_stage=source,
    )
    return CocoEvalImportResult(
        run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        metrics=metrics,
        metrics_by_node_path=metrics_path,
    )


def parse_coco_eval_metrics(path: Path | str) -> dict[str, MetricValue]:
    """Parse COCO eval metrics from JSON or text."""
    eval_path = Path(path)
    text = eval_path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _parse_coco_eval_text(text)
    if not isinstance(data, dict):
        raise ValueError("COCO eval JSON must contain a mapping.")
    return _parse_coco_eval_mapping(data)


def _parse_coco_eval_mapping(data: dict[str, Any]) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    stats = data.get("stats")
    if isinstance(stats, list):
        for index, name in COCO_STATS_METRICS.items():
            if index < len(stats):
                metrics[name] = _metric_value(stats[index])

    for raw_name, metric_name in COCO_KEY_ALIASES.items():
        if raw_name in data:
            metrics[metric_name] = _metric_value(data[raw_name])

    nested_metrics = data.get("metrics")
    if isinstance(nested_metrics, dict):
        metrics.update(_parse_coco_eval_mapping(nested_metrics))

    metrics.update(_per_class_metrics(data.get("per_class_ap"), suffix="ap"))
    metrics.update(_per_class_metrics(data.get("per_class_AP"), suffix="ap"))
    metrics.update(_per_class_metrics(data.get("per_class_ap50"), suffix="ap50"))
    metrics.update(_per_class_metrics(data.get("per_class_AP50"), suffix="ap50"))
    return metrics


def _parse_coco_eval_text(text: str) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    for line in text.splitlines():
        if "Average Precision" in line:
            value = _summary_value(line)
            if value is None:
                continue
            if "IoU=0.50:0.95" in line and "area=   all" in line:
                metrics["coco_ap50_95"] = value
            elif "IoU=0.50 " in line and "area=   all" in line:
                metrics["coco_ap50"] = value
            elif "IoU=0.75 " in line and "area=   all" in line:
                metrics["coco_ap75"] = value
            elif "area= small" in line:
                metrics["ap_small"] = value
            elif "area=medium" in line:
                metrics["ap_medium"] = value
            elif "area= large" in line:
                metrics["ap_large"] = value
        elif "Average Recall" in line:
            value = _summary_value(line)
            if value is None:
                continue
            if "area= small" in line:
                metrics["ar_small"] = value
            elif "area=medium" in line:
                metrics["ar_medium"] = value
            elif "area= large" in line:
                metrics["ar_large"] = value
    metrics.update(_parse_per_class_text(text))
    return metrics


def _per_class_metrics(raw: Any, suffix: str) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    if isinstance(raw, dict):
        for class_name, value in raw.items():
            metrics[f"per_class_{suffix}/{_metric_key(class_name)}"] = _metric_value(value)
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            class_name = item.get("class") or item.get("class_name") or item.get("name")
            value = item.get("ap") if "ap" in item else item.get("AP")
            if class_name is not None and value is not None:
                metrics[f"per_class_{suffix}/{_metric_key(class_name)}"] = _metric_value(value)
    return metrics


def _parse_per_class_text(text: str) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    pattern = re.compile(r"^\s*(?P<class>[A-Za-z0-9_ .-]+)\s+(?:AP|ap)\s*[:=]\s*(?P<value>-?\d+(?:\.\d+)?)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            metrics[f"per_class_ap/{_metric_key(match.group('class'))}"] = float(match.group("value"))
    return metrics


def _summary_value(line: str) -> float | None:
    match = re.search(r"=\s*(-?\d+(?:\.\d+)?)\s*$", line)
    return float(match.group(1)) if match else None


def _metric_value(value: Any) -> MetricValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _metric_key(value: Any) -> str:
    return str(value).strip().replace("/", "_")
