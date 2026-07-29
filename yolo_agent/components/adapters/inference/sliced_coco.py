"""COCO evaluation that exposes only the sliced inference namespace."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from yolo_agent.adapters.ultralytics.coco_post_eval import write_coco_eval_report


class SlicedCocoMetrics(BaseModel):
    """Accuracy metrics from a slicing-only inference protocol."""

    sliced_map50_95: float
    sliced_ap_small: float
    inference_policy_changed: bool = True


def evaluate_sliced_coco(
    *,
    annotations_path: Path,
    predictions_path: Path,
) -> SlicedCocoMetrics:
    """Evaluate sliced predictions without publishing standard metric keys."""
    with tempfile.TemporaryDirectory(prefix="yolo-agent-sliced-coco-") as directory:
        report_path = Path(directory) / "coco_eval.json"
        write_coco_eval_report(
            annotations_path=annotations_path,
            predictions_path=predictions_path,
            output_path=report_path,
        )
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    map50_95 = report.get("AP")
    ap_small = report.get("AP_small")
    if not isinstance(map50_95, (int, float)) or not isinstance(ap_small, (int, float)):
        raise ValueError("COCO evaluation did not produce AP and AP_small")
    return SlicedCocoMetrics(
        sliced_map50_95=float(map50_95),
        sliced_ap_small=float(ap_small),
    )


__all__ = ["SlicedCocoMetrics", "evaluate_sliced_coco"]
