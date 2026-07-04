"""Ultralytics training command and result import helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import ExperimentNode, MetricValue


ULTRALYTICS_METRIC_ALIASES = {
    "metrics/mAP50-95(B)": "map50_95",
    "metrics/mAP50(B)": "map50",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "fitness": "fitness",
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
}


class UltralyticsTrainingConfig(BaseModel):
    """Typed training defaults for Ultralytics CLI execution."""

    model: str = "yolo26s.pt"
    data: Path
    project: Path = Path("runs") / "ultralytics"
    task: str = "detect"
    epochs: int = Field(default=100, ge=1)
    imgsz: int = Field(default=640, ge=1)
    batch: int | str = "auto"
    device: str = "0"
    workers: int = Field(default=8, ge=0)
    optimizer: str | None = None
    patience: int | None = None
    amp: bool = True
    resume: bool | str | Path | None = None
    timeout_seconds: int | None = None
    overrides: dict[str, str | int | float | bool | Path] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "UltralyticsTrainingConfig":
        """Load training defaults from YAML."""
        with Path(path).open("r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Training config must contain a mapping: {path}")
        training = data.get("training", data)
        if not isinstance(training, dict):
            raise ValueError(f"Training config 'training' must contain a mapping: {path}")
        return cls.model_validate(training)


class Yolo26CocoGoal(BaseModel):
    """Evidence target for YOLO26 COCO optimization."""

    baseline_model: str = "yolo26s.pt"
    dataset: str = "COCO2017"
    validation_split: str = "val2017"
    primary_metric: str = "map50_95"
    target_delta_points: float = 2.0
    minimum_seeds: int = 3
    equal_training_budget_required: bool = True
    notes: list[str] = Field(default_factory=list)


def command_from_training_config(
    node: ExperimentNode,
    config: UltralyticsTrainingConfig,
    run_id: str,
    data_path: Path | str | None = None,
) -> CommandSpec:
    """Build a typed train command for one experiment node."""
    candidate = node.candidate_config
    model = _model_for_candidate(candidate, config.model)
    overrides = {**config.overrides, **candidate.train_overrides}
    name = _safe_run_name(run_id, node.node_id)
    return CommandSpec.ultralytics_train(
        model=model,
        data=data_path or config.data,
        project=config.project,
        name=name,
        seed=node.seed,
        task=config.task,
        epochs=config.epochs,
        imgsz=int(overrides.pop("imgsz", config.imgsz)),
        batch=overrides.pop("batch", config.batch),
        device=str(overrides.pop("device", config.device)),
        workers=int(overrides.pop("workers", config.workers)),
        optimizer=str(overrides.pop("optimizer")) if "optimizer" in overrides else config.optimizer,
        patience=int(overrides.pop("patience")) if "patience" in overrides else config.patience,
        amp=_bool_override(overrides.pop("amp")) if "amp" in overrides else config.amp,
        resume=overrides.pop("resume", config.resume),
        timeout_seconds=config.timeout_seconds,
        overrides=overrides,
        metadata={
            "run_id": run_id,
            "node_id": node.node_id,
            "candidate_id": candidate.candidate_id,
            "dataset_version": node.data_version,
            "seed": node.seed,
            "training_executor": "ultralytics",
        },
    )


class UltralyticsRunImporter:
    """Import Ultralytics train/val artifacts into node-level evidence."""

    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store

    def import_run(
        self,
        run_id: str,
        node: ExperimentNode,
        run_dir: Path | str,
        source: str = "ultralytics_train",
        verified: bool = True,
    ) -> dict[str, MetricValue]:
        """Parse one Ultralytics run directory and persist metrics/artifacts."""
        directory = Path(run_dir)
        metrics = parse_ultralytics_run(directory)
        results_csv = directory / "results.csv"
        source_artifact = results_csv if results_csv.is_file() else None
        self.evidence_store.log_candidate_metrics(
            run_id=run_id,
            candidate_id=node.candidate_config.candidate_id,
            node_id=node.node_id,
            metrics=metrics,
            dataset_version=node.data_version,
            split="val",
            source=source,
            verified=verified,
            validator="ultralytics_results_importer",
            source_artifact=source_artifact,
        )
        for artifact_name, artifact_path in expected_ultralytics_artifacts(directory).items():
            if artifact_path.exists():
                self.evidence_store.log_artifact_manifest(
                    run_id=run_id,
                    name=f"{node.node_id}_{artifact_name}",
                    artifact_path=artifact_path,
                    producer_stage=source,
                )
        return metrics


def parse_ultralytics_run(run_dir: Path | str) -> dict[str, MetricValue]:
    """Parse metrics and artifact-derived facts from an Ultralytics run directory."""
    directory = Path(run_dir)
    metrics: dict[str, MetricValue] = {}
    results_csv = directory / "results.csv"
    if results_csv.is_file():
        metrics.update(parse_results_csv(results_csv))
    best_pt = directory / "weights" / "best.pt"
    if best_pt.is_file():
        metrics["model_size_mb"] = round(best_pt.stat().st_size / (1024 * 1024), 4)
    args_yaml = directory / "args.yaml"
    if args_yaml.is_file():
        args = _read_yaml_mapping(args_yaml)
        if "imgsz" in args:
            metrics.setdefault("imgsz", _coerce_metric(args["imgsz"]))
        if "epochs" in args:
            metrics.setdefault("epochs", _coerce_metric(args["epochs"]))
    return metrics


def parse_results_csv(path: Path | str) -> dict[str, MetricValue]:
    """Parse Ultralytics results.csv and select the best validation row."""
    results_path = Path(path)
    with results_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    row = max(rows, key=_row_score)
    metrics: dict[str, MetricValue] = {}
    for raw_name, value in row.items():
        name = ULTRALYTICS_METRIC_ALIASES.get(raw_name.strip())
        if name is None:
            continue
        metrics[name] = _coerce_metric(value)
    if "epoch" in row:
        metrics["best_epoch"] = _coerce_metric(row["epoch"])
    return metrics


def expected_ultralytics_artifacts(run_dir: Path | str) -> dict[str, Path]:
    """Return known Ultralytics artifact paths for one run directory."""
    directory = Path(run_dir)
    return {
        "results_csv": directory / "results.csv",
        "args_yaml": directory / "args.yaml",
        "best_pt": directory / "weights" / "best.pt",
        "last_pt": directory / "weights" / "last.pt",
    }


def _model_for_candidate(candidate: CandidateConfig, default_model: str) -> str:
    model = candidate.base_model or default_model
    return model if model.endswith((".pt", ".yaml", ".yml")) else f"{model}.pt"


def _safe_run_name(run_id: str, node_id: str) -> str:
    return f"{run_id}_{node_id}".replace("/", "_").replace("\\", "_").replace(" ", "_")


def _row_score(row: dict[str, str]) -> float:
    for key in ("metrics/mAP50-95(B)", "fitness", "metrics/mAP50(B)"):
        value = row.get(key)
        if value not in {None, ""}:
            return float(value)
    return 0.0


def _coerce_metric(value: Any) -> MetricValue:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def _bool_override(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
