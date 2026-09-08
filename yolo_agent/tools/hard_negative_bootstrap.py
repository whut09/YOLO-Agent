"""Executors for the train-side hard-negative evidence bootstrap stages.

The bootstrap is intentionally separate from candidate training.  The inference
stage may use a GPU when a real run invokes it, while manifest construction is
CPU-only.  Neither stage emits model metrics or an optimization decision.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yolo_agent.components.adapters.data_pipeline.hard_negative import (
    HardNegativeEvidenceBootstrap,
)
from yolo_agent.components.adapters.data_pipeline.hard_negative_evidence import (
    TrainHardNegativePrediction,
    TrainHardNegativePredictionBatch,
    TrainSampleIndex,
    produce_train_hard_negative_manifest,
    train_sample_index_from_yolo_data,
)
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_failure import (
    ExecutionFailure,
    classify_execution_failure,
    external_gpu_conflict_failure,
)
from yolo_agent.core.executor import ExecutionResult
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.gpu_runtime import inspect_gpu_runtime


class HardNegativeBootstrapStageError(RuntimeError):
    """A stage failed without creating production evidence."""

    def __init__(
        self,
        message: str,
        *,
        failure: ExecutionFailure | None = None,
        stdout: str = "",
        stderr: str = "",
        return_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


def execute_hard_negative_bootstrap_stage(
    node: ExperimentNode,
    run_id: str,
    command: CommandSpec,
    *,
    evidence_store: EvidenceStore | None = None,
) -> ExecutionResult:
    """Execute one explicitly typed bootstrap stage.

    This entrypoint is called by ``UltralyticsTrainExecutor`` before it tries to
    translate non-training commands into a training config.  It is also usable
    by a mock executor in CPU tests.
    """
    started = datetime.now(timezone.utc)
    started_at = time.monotonic()
    stage = str(command.metadata.get("hard_negative_bootstrap_stage") or "")
    try:
        if stage == "train_split_inference":
            artifacts = _run_train_split_inference(node, run_id, command)
        elif stage == "hard_negative_manifest":
            artifacts = _run_manifest_stage(node, run_id, command)
        else:
            raise HardNegativeBootstrapStageError(
                "unknown hard-negative bootstrap stage: " + (stage or "missing")
            )
    except HardNegativeBootstrapStageError as exc:
        _record_stage_failure(
            command,
            stage,
            str(exc),
            retryable_resource=bool(exc.failure and exc.failure.waiting_for_external_gpu),
        )
        ended = datetime.now(timezone.utc)
        result = ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status="failed",
            command=command,
            return_code=exc.return_code,
            stdout=exc.stdout,
            stderr=exc.stderr,
            started_at=started,
            ended_at=ended,
            duration_seconds=time.monotonic() - started_at,
            message=str(exc),
            failure=exc.failure,
        )
    except (OSError, TypeError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        _record_stage_failure(command, stage, str(exc), retryable_resource=False)
        ended = datetime.now(timezone.utc)
        result = ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status="failed",
            command=command,
            started_at=started,
            ended_at=ended,
            duration_seconds=time.monotonic() - started_at,
            message=f"hard-negative bootstrap failed: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive process boundary
        _record_stage_failure(command, stage, str(exc), retryable_resource=False)
        ended = datetime.now(timezone.utc)
        result = ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status="failed",
            command=command,
            started_at=started,
            ended_at=ended,
            duration_seconds=time.monotonic() - started_at,
            message=f"hard-negative bootstrap failed: {type(exc).__name__}: {exc}",
        )
    else:
        ended = datetime.now(timezone.utc)
        result = ExecutionResult(
            run_id=run_id,
            node_id=node.node_id,
            candidate_id=node.candidate_config.candidate_id,
            status="completed",
            command=command,
            return_code=0,
            started_at=started,
            ended_at=ended,
            duration_seconds=time.monotonic() - started_at,
            message=f"Hard-negative bootstrap stage completed: {stage}.",
            artifacts=artifacts,
            metrics={"optimization_metric_eligible": False},
        )
    if evidence_store is not None:
        result.log_to_evidence_store(evidence_store)
    return result


def _record_stage_failure(
    command: CommandSpec,
    stage: str,
    message: str,
    *,
    retryable_resource: bool,
) -> None:
    """Persist a stage-local failure without manufacturing evidence."""
    if stage not in {
        "baseline_train",
        "train_split_inference",
        "hard_negative_manifest",
        "hard_negative_candidate",
    }:
        return
    raw_path = command.metadata.get("hard_negative_bootstrap_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return
    path = Path(raw_path).resolve()
    if not path.is_file():
        return
    try:
        state = HardNegativeEvidenceBootstrap.from_path(path)
        stage_record = state.stage(stage)  # type: ignore[arg-type]
        reason = f"{stage}_failed"
        if retryable_resource:
            state.reason_codes = list(
                dict.fromkeys([*state.reason_codes, "external_gpu_process"])
            )
        else:
            stage_record.status = "failed"
            stage_record.reason_codes = list(
                dict.fromkeys([reason, message[:240]])
            )
            state.reason_codes = list(dict.fromkeys([*state.reason_codes, reason]))
        state.write(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        # The execution result remains the authoritative failure artifact when
        # an already-corrupt bootstrap state cannot be updated.
        return


def _run_train_split_inference(
    node: ExperimentNode,
    run_id: str,
    command: CommandSpec,
) -> dict[str, Path]:
    metadata = command.metadata
    _require_train_scope(metadata)
    if _int_value(metadata.get("imgsz"), default=640) != 640:
        raise HardNegativeBootstrapStageError(
            "hard-negative train inference requires imgsz=640"
        )
    state = _load_bootstrap_state(metadata)
    checkpoint = _required_path(metadata, "baseline_checkpoint_path")
    train_index_path = _required_path(metadata, "train_index_path")
    output_path = _required_path(metadata, "prediction_artifact_path")
    dataset_hash = _required_value(metadata, "dataset_manifest_hash", state.dataset_manifest_hash)
    if not checkpoint.is_file():
        raise HardNegativeBootstrapStageError(
            f"baseline checkpoint is unavailable for hard-negative inference: {checkpoint}"
        )
    if not train_index_path.is_file():
        data_yaml = _required_path(metadata, "data_yaml")
        try:
            train_index = train_sample_index_from_yolo_data(
                data_yaml,
                dataset_manifest_hash=dataset_hash,
                output_path=train_index_path,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise HardNegativeBootstrapStageError(
                "train sample index could not be generated from the real train split; "
                f"recover_train_sample_index_from_data_yaml: {exc}"
            ) from exc
    else:
        train_index = TrainSampleIndex.from_path(train_index_path)
    if train_index.dataset_manifest_hash != dataset_hash:
        raise HardNegativeBootstrapStageError(
            "train sample index dataset manifest hash does not match bootstrap protocol"
        )
    checkpoint_hash = _sha256_file(checkpoint)
    declared_checkpoint_hash = str(metadata.get("baseline_checkpoint_hash") or "")
    if declared_checkpoint_hash and declared_checkpoint_hash != checkpoint_hash:
        raise HardNegativeBootstrapStageError(
            "baseline checkpoint hash does not match hard-negative bootstrap"
        )
    if state.baseline_checkpoint_hash and state.baseline_checkpoint_hash != checkpoint_hash:
        raise HardNegativeBootstrapStageError(
            "baseline checkpoint hash changed during hard-negative bootstrap"
        )
    if not output_path.is_file():
        _run_inference_command(command)
        source = _find_prediction_artifact(command, output_path)
        if source is None:
            raise HardNegativeBootstrapStageError(
                "train split inference completed without a prediction artifact"
            )
    else:
        source = output_path
    batch = _normalise_prediction_artifact(
        source,
        dataset_manifest_hash=dataset_hash,
        source_run_id=str(metadata.get("source_run_id") or state.source_run_id or run_id),
        baseline_protocol_hash=_required_value(
            metadata, "baseline_protocol_hash", state.baseline_protocol_hash
        ),
        baseline_checkpoint_hash=checkpoint_hash,
        train_index_hash=train_index.index_hash,
    )
    batch.write(output_path)
    state.baseline_checkpoint_hash = checkpoint_hash
    state.train_index_hash = train_index.index_hash
    state.dataset_manifest_hash = dataset_hash
    state.baseline_protocol_hash = batch.baseline_protocol_hash
    if state.stage("baseline_train").status != "completed":
        state.advance(
            "baseline_train",
            status="completed",
            node_id=str(metadata.get("baseline_node_id") or "") or None,
            artifact_path=checkpoint,
            reason_codes=["baseline_checkpoint_verified"],
        )
    state.advance(
        "train_split_inference",
        status="completed",
        node_id=node.node_id,
        artifact_path=output_path,
        reason_codes=["train_split_inference_completed"],
    )
    state.write(_bootstrap_path(metadata))
    return {
        "train_predictions": output_path,
        "train_sample_index": train_index_path,
        "hard_negative_bootstrap": _bootstrap_path(metadata),
    }


def _run_manifest_stage(
    node: ExperimentNode,
    run_id: str,
    command: CommandSpec,
) -> dict[str, Path]:
    metadata = command.metadata
    _require_train_scope(metadata)
    state = _load_bootstrap_state(metadata)
    predictions_path = _required_path(metadata, "prediction_artifact_path")
    train_index_path = _required_path(metadata, "train_index_path")
    output_path = _required_path(metadata, "manifest_artifact_path")
    if not predictions_path.is_file():
        raise HardNegativeBootstrapStageError(
            "train split predictions are missing; run train_split_inference first"
        )
    if not train_index_path.is_file():
        raise HardNegativeBootstrapStageError(
            "train sample index is missing; recover_train_sample_index_before_manifest"
        )
    if state.stage("train_split_inference").status != "completed":
        raise HardNegativeBootstrapStageError(
            "hard-negative manifest cannot run before train_split_inference"
        )
    dataset_hash = _required_value(metadata, "dataset_manifest_hash", state.dataset_manifest_hash)
    protocol_hash = _required_value(
        metadata, "baseline_protocol_hash", state.baseline_protocol_hash
    )
    checkpoint_hash = _required_value(
        metadata, "baseline_checkpoint_hash", state.baseline_checkpoint_hash
    )
    batch = TrainHardNegativePredictionBatch.from_path(predictions_path)
    train_index = TrainSampleIndex.from_path(train_index_path)
    manifest = produce_train_hard_negative_manifest(
        batch,
        train_index,
        output_path=output_path,
        expected_dataset_manifest_hash=dataset_hash,
        expected_protocol_hash=protocol_hash,
        expected_baseline_checkpoint_hash=checkpoint_hash,
        require_provenance=True,
    )
    state.bind_manifest(
        manifest,
        path=output_path,
        baseline_checkpoint_hash=checkpoint_hash,
    )
    state.recovery_actions = [
        action
        for action in state.recovery_actions
        if action not in {"recover_train_hard_negative_evidence"}
    ]
    state.reason_codes = list(
        dict.fromkeys([*state.reason_codes, "hard_negative_manifest_completed"])
    )
    state.write(_bootstrap_path(metadata))
    return {
        "hard_negative_manifest": output_path,
        "train_predictions": predictions_path,
        "train_sample_index": train_index_path,
        "hard_negative_bootstrap": _bootstrap_path(metadata),
    }


def _run_inference_command(command: CommandSpec) -> None:
    argv = list(command.argv or [command.command, *command.args])
    if not argv:
        raise HardNegativeBootstrapStageError("train split inference command is empty")
    if _cli_value(argv, "split") != "train":
        raise HardNegativeBootstrapStageError(
            "train split inference command must use split=train"
        )
    if _cli_value(argv, "imgsz") != "640":
        raise HardNegativeBootstrapStageError(
            "train split inference command must use imgsz=640"
        )
    executable = shutil.which(argv[0]) or argv[0]
    if not Path(executable).is_file() and shutil.which(argv[0]) is None:
        raise HardNegativeBootstrapStageError(f"inference executable not found: {argv[0]}")
    snapshot = inspect_gpu_runtime(command)
    if snapshot.has_external_training_conflict:
        failure = external_gpu_conflict_failure(
            command,
            snapshot,
            summary="GPU is occupied by another process during train-side evidence inference.",
        )
        raise HardNegativeBootstrapStageError(failure.summary, failure=failure)
    try:
        completed = subprocess.run(
            argv,
            cwd=command.cwd,
            env=None,
            timeout=command.timeout_seconds,
            shell=command.shell,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HardNegativeBootstrapStageError(
            f"train split inference timed out after {command.timeout_seconds} seconds",
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "",
        ) from exc
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode == 0:
        return
    failure = classify_execution_failure(
        stdout=stdout,
        stderr=stderr,
        command=command,
        gpu_snapshot=inspect_gpu_runtime(command),
    )
    message = "train split inference command failed"
    raise HardNegativeBootstrapStageError(
        message,
        failure=failure,
        stdout=stdout,
        stderr=stderr,
        return_code=completed.returncode,
    )


def _normalise_prediction_artifact(
    source: Path,
    *,
    dataset_manifest_hash: str,
    source_run_id: str,
    baseline_protocol_hash: str,
    baseline_checkpoint_hash: str,
    train_index_hash: str,
) -> TrainHardNegativePredictionBatch:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and "predictions" in payload:
        raw_predictions = payload["predictions"]
    else:
        raw_predictions = payload
    if not isinstance(raw_predictions, list):
        raise ValueError("train split inference artifact must contain a predictions list")
    predictions: list[TrainHardNegativePrediction] = []
    for item in raw_predictions:
        if not isinstance(item, dict):
            continue
        predictions.append(
            TrainHardNegativePrediction(
                image_id=str(item["image_id"]),
                predicted_class=int(item.get("category_id", item.get("class_id", item.get("cls", 0)))),
                score=float(item.get("score", item.get("confidence", 0.0))),
                bbox=[float(value) for value in item["bbox"]],
                error_type=str(item.get("error_type") or "background_false_positive"),
            )
        )
    return TrainHardNegativePredictionBatch(
        dataset_manifest_hash=dataset_manifest_hash,
        source_split="train",
        source_run_id=source_run_id,
        baseline_protocol_hash=baseline_protocol_hash,
        baseline_checkpoint_hash=baseline_checkpoint_hash,
        predictions=predictions,
    )


def _find_prediction_artifact(command: CommandSpec, expected: Path) -> Path | None:
    declared = command.metadata.get("prediction_source_path")
    candidates = [Path(str(declared)).resolve()] if declared else []
    candidates.append(expected)
    root = expected.parent
    if root.is_dir():
        candidates.extend(
            path
            for path in sorted(root.rglob("*.json"))
            if "prediction" in path.name.lower() or "labels" in path.name.lower()
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_bootstrap_state(metadata: dict[str, Any]) -> HardNegativeEvidenceBootstrap:
    path = _bootstrap_path(metadata)
    if not path.is_file():
        raise HardNegativeBootstrapStageError(
            f"hard-negative bootstrap state is missing: {path}"
        )
    return HardNegativeEvidenceBootstrap.from_path(path)


def _bootstrap_path(metadata: dict[str, Any]) -> Path:
    return _required_path(metadata, "hard_negative_bootstrap_path")


def _required_path(metadata: dict[str, Any], key: str) -> Path:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HardNegativeBootstrapStageError(
            f"hard-negative bootstrap metadata is missing {key}"
        )
    return Path(value).resolve()


def _required_value(metadata: dict[str, Any], key: str, fallback: str | None) -> str:
    value = str(metadata.get(key) or fallback or "").strip()
    if not value:
        raise HardNegativeBootstrapStageError(
            f"hard-negative bootstrap metadata is missing {key}"
        )
    return value


def _require_train_scope(metadata: dict[str, Any]) -> None:
    if str(metadata.get("source_split") or "") != "train":
        raise HardNegativeBootstrapStageError(
            "hard-negative evidence bootstrap requires source_split=train"
        )


def _cli_value(argv: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for value in argv:
        if value.startswith(prefix):
            return value.split("=", 1)[1]
    return None


def _int_value(value: object, *, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "HardNegativeBootstrapStageError",
    "execute_hard_negative_bootstrap_stage",
]
