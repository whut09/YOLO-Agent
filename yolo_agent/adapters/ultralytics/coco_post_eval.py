"""Fixed-protocol COCO validation after pilot training."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.core.command_spec import CommandSpec, ResourceRequirements


class CocoPostEvalConfig(BaseModel):
    """Configuration for candidate-specific COCO evidence collection."""

    enabled: bool = False
    profiles: list[str] = Field(
        default_factory=lambda: [
            "pilot",
            "pilot_3",
            "pilot_10",
            "baseline_full",
            "baseline_confirm",
            "candidate_full",
            "candidate_full_seed_1",
            "candidate_full_confirmation",
        ]
    )
    imgsz: int = Field(default=640, ge=640, le=640)
    split: str = "val"
    timeout_seconds: int = Field(default=7200, gt=0)
    plots: bool = True
    save_json: bool = True
    conf: float = Field(default=0.001, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)


def should_run_coco_post_eval(profile: str | None, config: CocoPostEvalConfig) -> bool:
    """Return whether a completed training profile requires fixed COCO evaluation."""
    return bool(config.enabled and profile and profile in set(config.profiles))


def requires_fixed_coco_post_eval(profile: str | None, round_stage: str | None) -> bool:
    """Return whether a training node is forbidden to finish without COCO evidence."""
    stage = str(round_stage or "")
    return bool(
        stage in {"pilot_3", "pilot_10", "candidate_full_seed_1", "candidate_full_confirmation"}
        or profile == "candidate_full"
    )


def post_eval_cli_executable(training_spec: CommandSpec) -> str:
    """Return the Ultralytics CLI executable from a direct or runtime-wrapped train command."""
    argv = list(training_spec.argv or [training_spec.command, *training_spec.args])
    runtime_entrypoint = str(training_spec.metadata.get("adapter_runtime_entrypoint") or "")
    if runtime_entrypoint:
        try:
            separator = argv.index("--")
        except ValueError as exc:
            raise ValueError(
                "Adapter runtime training command is missing the '--' inner-command separator."
            ) from exc
        argv = argv[separator + 1 :]

    if len(argv) < 3 or argv[1:3] != ["detect", "train"]:
        raise ValueError(
            "COCO post-eval requires an Ultralytics 'detect train' source command; "
            f"received: {' '.join(argv[:3]) or '<empty>'}"
        )
    return argv[0]


def build_coco_post_eval_spec(
    *,
    executable: str,
    checkpoint: Path,
    data: Path,
    output_dir: Path,
    device: str,
    workers: int,
    config: CocoPostEvalConfig,
) -> CommandSpec:
    """Build a typed, shell-free Ultralytics validation command."""
    argv = [
        executable,
        "detect",
        "val",
        f"model={checkpoint.as_posix()}",
        f"data={data.as_posix()}",
        f"project={output_dir.parent.as_posix()}",
        f"name={output_dir.name}",
        "exist_ok=True",
        f"imgsz={config.imgsz}",
        f"split={config.split}",
        f"device={device}",
        f"workers={workers}",
        f"save_json={config.save_json}",
        f"plots={config.plots}",
        f"conf={config.conf}",
        f"iou={config.iou}",
    ]
    return CommandSpec(
        command_type="benchmark",
        command=executable,
        args=argv[1:],
        argv=argv,
        shell=False,
        timeout_seconds=config.timeout_seconds,
        expected_artifacts={"predictions_json": output_dir / "predictions.json"},
        expected_metrics=[
            "coco_ap50_95",
            "ap_small",
            "ap_medium",
            "ap_large",
            "per_class_ap/*",
            "per_class_ar/*",
        ],
        resource_requirements=ResourceRequirements(requires_gpu=True, requires_batch_tuning=False),
        metadata={
            "evaluation_protocol": "coco_val2017_fixed_640",
            "fixed_imgsz": config.imgsz,
            "full_validation_split": True,
        },
    )


def write_coco_eval_report(
    *,
    annotations_path: Path,
    predictions_path: Path,
    output_path: Path,
) -> Path:
    """Evaluate COCO predictions and persist aggregate and per-class AP/AR."""
    COCO, COCOeval, evaluator_backend = _coco_evaluator_classes()

    ground_truth = COCO(str(annotations_path))
    predictions = _load_coco_predictions(ground_truth, predictions_path, COCO)
    evaluator = COCOeval(ground_truth, predictions, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    summary_buffer = io.StringIO()
    with contextlib.redirect_stdout(summary_buffer):
        evaluator.summarize()

    categories = {int(item["id"]): str(item["name"]) for item in ground_truth.loadCats(evaluator.params.catIds)}
    per_class_ap = _per_class_ap(evaluator, categories)
    per_class_ar = _per_class_ar(evaluator, categories)
    stats = [float(value) for value in evaluator.stats]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluator_backend": evaluator_backend,
        "protocol": {
            "dataset": "COCO2017",
            "split": "val2017",
            "imgsz": 640,
            "iou_type": "bbox",
            "max_dets": list(evaluator.params.maxDets),
        },
        "stats": stats,
        "AP": _stat(stats, 0),
        "AP50": _stat(stats, 1),
        "AP75": _stat(stats, 2),
        "AP_small": _stat(stats, 3),
        "AP_medium": _stat(stats, 4),
        "AP_large": _stat(stats, 5),
        "AR_small": _stat(stats, 9),
        "AR_medium": _stat(stats, 10),
        "AR_large": _stat(stats, 11),
        "per_class_ap": per_class_ap,
        "per_class_ar": per_class_ar,
        "summary": summary_buffer.getvalue(),
        "source_predictions": predictions_path.as_posix(),
        "source_annotations": annotations_path.as_posix(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _coco_evaluator_classes() -> tuple[Any, Any, str]:
    """Load either supported COCO evaluator without changing metric semantics."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        return COCO, COCOeval, "pycocotools"
    except ImportError:
        try:
            from faster_coco_eval.core import COCO, COCOeval_faster

            return COCO, COCOeval_faster, "faster-coco-eval"
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "fixed-protocol COCO post-eval requires pycocotools or faster-coco-eval"
            ) from exc


def _load_coco_predictions(ground_truth: Any, predictions_path: Path, coco_class: Any) -> Any:
    """Treat an empty prediction list as valid all-missed detection evidence."""
    raw = json.loads(predictions_path.read_text(encoding="utf-8-sig"))
    if raw:
        return ground_truth.loadRes(str(predictions_path))
    predictions = coco_class()
    predictions.dataset = {
        "images": list(ground_truth.dataset.get("images", [])),
        "categories": list(ground_truth.dataset.get("categories", [])),
        "annotations": [],
    }
    predictions.createIndex()
    return predictions


def _per_class_ap(evaluator: Any, categories: dict[int, str]) -> dict[str, float | None]:
    precision = evaluator.eval.get("precision")
    if precision is None:
        return {}
    output: dict[str, float | None] = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        valid = values[values > -1]
        output[categories.get(int(category_id), str(category_id))] = float(valid.mean()) if valid.size else None
    return output


def _per_class_ar(evaluator: Any, categories: dict[int, str]) -> dict[str, float | None]:
    recall = evaluator.eval.get("recall")
    if recall is None:
        return {}
    output: dict[str, float | None] = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        values = recall[:, index, 0, -1]
        valid = values[values > -1]
        output[categories.get(int(category_id), str(category_id))] = float(valid.mean()) if valid.size else None
    return output


def _stat(stats: list[float], index: int) -> float | None:
    return stats[index] if len(stats) > index else None
