"""Real CUDA certification harness for paper component runtime adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import time
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from yolo_agent.certification.component_schemas import (
    ComponentGPUCertificationEvidence,
    ComponentGPUProtocol,
    ComponentGPUResources,
    ComponentSmokeWorkerRequest,
)
from yolo_agent.certification.fixture import (
    create_mini_coco_fixture,
    load_mini_coco_fixture_manifest,
)
from yolo_agent.components.adapters import AdapterContext, AdapterRuntimePayload
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.adapters.ultralytics.plugin_context import (
    PluginRuntimeEvidence,
    runtime_evidence_path,
)
from yolo_agent.adapters.ultralytics.inference_latency import (
    InferenceLatencyConfig,
    benchmark_checkpoint,
)


GPU_CERTIFICATION_COMPONENTS: tuple[str, ...] = (
    "sampling.small_object",
    "loss.quality.correlation",
    "loss.calibration.bpc",
    "loss.quality.pseudo_iou",
    "distillation.yolo26_teacher_student",
    "head.p2_small_object",
    "neck.multi_scale_fusion",
    "neck.gold_gather_distribute",
    "neck.rtmdet_large_kernel",
)


class PreparedComponentGPURun(BaseModel):
    """Fully materialized, non-executed real GPU certification input."""

    model_config = ConfigDict(extra="forbid")

    protocol: ComponentGPUProtocol
    data_yaml: Path
    fixture_manifest_path: Path
    runtime_payload_path: Path
    train_command: list[str]
    project_dir: Path
    run_name: str = "initial"


class GPUTrainingStageResult(BaseModel):
    """Artifacts returned by one initial or resumed Ultralytics invocation."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: Path
    results_csv: Path
    duration_s: float


class GPUCheckpointState(BaseModel):
    """Minimal checkpoint facts needed by the certification gate."""

    model_config = ConfigDict(extra="forbid")

    epoch: int
    amp: bool
    model_size_mb: float


class ComponentGPUExecutionBackend(Protocol):
    """Injectable boundary keeping default tests off CUDA."""

    def run_training(
        self,
        *,
        payload_path: Path,
        command: list[str],
        project_dir: Path,
        run_name: str,
    ) -> GPUTrainingStageResult: ...

    def prepare_resume_checkpoint(self, source: Path, target: Path) -> Path: ...

    def inspect_checkpoint(self, checkpoint: Path) -> GPUCheckpointState: ...

    def resource_evidence(
        self,
        *,
        checkpoint: Path,
        device: str,
        train_duration_s: float,
        resume_duration_s: float,
    ) -> ComponentGPUResources: ...


class RealComponentGPUExecutionBackend:
    """Execute the typed payload through the installed Ultralytics Trainer."""

    def run_training(
        self,
        *,
        payload_path: Path,
        command: list[str],
        project_dir: Path,
        run_name: str,
    ) -> GPUTrainingStageResult:
        from yolo_agent.adapters.ultralytics.runtime_entrypoint import (
            run_ultralytics_training,
        )

        if run_name == "initial":
            import torch

            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        exit_code = run_ultralytics_training(payload_path, command)
        duration = time.perf_counter() - started
        if exit_code != 0:
            raise RuntimeError(f"Ultralytics runtime entrypoint exited with {exit_code}")
        run_dir = project_dir / run_name
        checkpoint = run_dir / "weights" / "last.pt"
        results = run_dir / "results.csv"
        if not checkpoint.is_file() or not results.is_file():
            raise RuntimeError(
                f"training artifacts missing: checkpoint={checkpoint.is_file()} "
                f"results_csv={results.is_file()}"
            )
        return GPUTrainingStageResult(
            checkpoint=checkpoint,
            results_csv=results,
            duration_s=duration,
        )

    def prepare_resume_checkpoint(self, source: Path, target: Path) -> Path:
        import torch

        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Ultralytics resume checkpoint must be a mapping")
        train_args = checkpoint.get("train_args")
        if not isinstance(train_args, dict):
            raise ValueError("Ultralytics checkpoint is missing train_args")
        train_args["epochs"] = max(int(train_args.get("epochs", 1)) + 1, 2)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, target)
        return target

    def inspect_checkpoint(self, checkpoint: Path) -> GPUCheckpointState:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Ultralytics checkpoint must be a mapping")
        train_args = payload.get("train_args") or {}
        return GPUCheckpointState(
            epoch=int(payload.get("epoch", -1)),
            amp=bool(train_args.get("amp", False)),
            model_size_mb=checkpoint.stat().st_size / (1024 * 1024),
        )

    def resource_evidence(
        self,
        *,
        checkpoint: Path,
        device: str,
        train_duration_s: float,
        resume_duration_s: float,
    ) -> ComponentGPUResources:
        import torch

        index = int(str(device).split(",", maxsplit=1)[0])
        properties = torch.cuda.get_device_properties(index)
        latency = benchmark_checkpoint(
            checkpoint,
            device=device,
            config=InferenceLatencyConfig(
                enabled=True,
                warmup_runs=1,
                timed_runs=3,
            ),
        )
        if latency.status != "completed" or latency.latency_ms is None:
            raise RuntimeError(f"checkpoint latency benchmark failed: {latency.error}")
        return ComponentGPUResources(
            device=device,
            gpu_name=properties.name,
            total_vram_mb=properties.total_memory / (1024 * 1024),
            peak_vram_mb=torch.cuda.max_memory_allocated(index) / (1024 * 1024),
            train_duration_s=train_duration_s,
            resume_duration_s=resume_duration_s,
            latency_ms=latency.latency_ms,
            model_size_mb=checkpoint.stat().st_size / (1024 * 1024),
        )


def run_real_component_gpu_certification(
    request: ComponentSmokeWorkerRequest,
    source_payload: AdapterRuntimePayload,
    *,
    backend: ComponentGPUExecutionBackend | None = None,
) -> ComponentGPUCertificationEvidence:
    """Run train and resume through the typed payload, retaining failures."""
    execution = backend or RealComponentGPUExecutionBackend()
    prepared = prepare_component_gpu_run(request, source_payload)
    payload = AdapterRuntimePayload.read(
        prepared.runtime_payload_path,
        verify_imports=True,
    )
    checks: dict[str, bool | str | int | float] = {
        "real_ultralytics_train": False,
        "required_hooks_observed": False,
        "backward_observed": False,
        "amp_enabled": False,
        "checkpoint_saved": False,
        "resume_completed": False,
        "resume_checkpoint_saved": False,
        "adapter_hash_matched": request.adapter_hash == prepared.protocol.adapter_hash,
        "fixture_manifest_matched": False,
        "adapter_artifacts_complete": False,
        "component_profile_verified": False,
    }
    artifacts: dict[str, Path] = {
        "fixture_manifest": prepared.fixture_manifest_path,
        "runtime_payload": prepared.runtime_payload_path,
    }
    errors: list[str] = []
    resources: ComponentGPUResources | None = None
    resume_command: list[str] = []
    hook_counts: dict[str, dict[str, int]] = {}
    try:
        fixture = load_mini_coco_fixture_manifest(prepared.data_yaml.parent)
        checks["fixture_manifest_matched"] = (
            fixture.fixture_hash == prepared.protocol.fixture_manifest_hash
        )
        initial = execution.run_training(
            payload_path=prepared.runtime_payload_path,
            command=prepared.train_command,
            project_dir=prepared.project_dir,
            run_name=prepared.run_name,
        )
        checks["real_ultralytics_train"] = True
        checks["backward_observed"] = _results_show_backward(initial.results_csv)
        initial_state = execution.inspect_checkpoint(initial.checkpoint)
        checks["amp_enabled"] = initial_state.amp
        checks["checkpoint_saved"] = initial.checkpoint.is_file()
        immutable_initial = Path(request.workspace) / "initial_checkpoint.pt"
        shutil.copy2(initial.checkpoint, immutable_initial)
        artifacts["initial_checkpoint"] = immutable_initial
        plugin_evidence_path = runtime_evidence_path(prepared.runtime_payload_path)
        plugin_evidence = PluginRuntimeEvidence.model_validate_json(
            plugin_evidence_path.read_text(encoding="utf-8-sig")
        )
        hook_counts = plugin_evidence.hook_call_counts
        checks["required_hooks_observed"] = _required_hooks_observed(
            payload,
            plugin_evidence,
        )
        artifacts["plugin_runtime_evidence"] = plugin_evidence_path
        checks["adapter_artifacts_complete"] = _adapter_artifacts_complete(
            payload,
            prepared.runtime_payload_path.parent,
            artifacts,
        )

        resume_source = execution.prepare_resume_checkpoint(
            immutable_initial,
            Path(request.workspace) / "resume_source.pt",
        )
        artifacts["resume_source"] = resume_source
        resume_command = _resume_training_command(
            checkpoint=resume_source,
            data=prepared.data_yaml,
            project=prepared.project_dir,
            name=prepared.run_name,
            device=request.device,
        )
        resumed = execution.run_training(
            payload_path=prepared.runtime_payload_path,
            command=resume_command,
            project_dir=prepared.project_dir,
            run_name=prepared.run_name,
        )
        resumed_state = execution.inspect_checkpoint(resumed.checkpoint)
        checks["resume_completed"] = resumed_state.epoch > initial_state.epoch
        checks["resume_checkpoint_saved"] = resumed.checkpoint.is_file()
        artifacts["resumed_checkpoint"] = resumed.checkpoint
        resumed_evidence = PluginRuntimeEvidence.model_validate_json(
            plugin_evidence_path.read_text(encoding="utf-8-sig")
        )
        hook_counts = resumed_evidence.hook_call_counts
        checks["required_hooks_observed"] = _required_hooks_observed(
            payload,
            resumed_evidence,
        )
        checks["adapter_artifacts_complete"] = _adapter_artifacts_complete(
            payload,
            prepared.runtime_payload_path.parent,
            artifacts,
        )
        from yolo_agent.certification.component_gpu_profiles import (
            validate_component_gpu_profile,
        )

        profile_checks = validate_component_gpu_profile(
            request.contract.component_id,
            payload,
            artifacts,
        )
        checks.update(profile_checks)
        checks["component_profile_verified"] = True
        resources = execution.resource_evidence(
            checkpoint=resumed.checkpoint,
            device=request.device,
            train_duration_s=initial.duration_s,
            resume_duration_s=resumed.duration_s,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(str(exc))

    required = {
        "real_ultralytics_train",
        "required_hooks_observed",
        "backward_observed",
        "amp_enabled",
        "checkpoint_saved",
        "resume_completed",
        "resume_checkpoint_saved",
        "adapter_hash_matched",
        "fixture_manifest_matched",
        "adapter_artifacts_complete",
        "component_profile_verified",
    }
    failed = sorted(name for name in required if checks.get(name) is not True)
    errors.extend(f"gpu_contract_failed:{name}" for name in failed)
    errors = list(dict.fromkeys(errors))
    evidence = ComponentGPUCertificationEvidence(
        component_id=request.contract.component_id,
        status="failed" if errors else "passed",
        worker_protocol_hash=request.protocol_hash,
        gpu_protocol=prepared.protocol,
        runtime_payload_path=prepared.runtime_payload_path,
        runtime_payload_hash=payload.payload_hash,
        train_command=prepared.train_command,
        resume_command=resume_command,
        hook_call_counts=hook_counts,
        checks=checks,
        resources=resources,
        artifacts=artifacts,
        errors=errors,
    )
    _atomic_gpu_evidence(
        evidence,
        Path(request.workspace) / "component_gpu_evidence.yaml",
    )
    return evidence


def prepare_component_gpu_run(
    request: ComponentSmokeWorkerRequest,
    source_payload: AdapterRuntimePayload,
) -> PreparedComponentGPURun:
    """Create a fresh runtime payload bound to the real tiny training command."""
    if not request.real_gpu_training:
        raise ValueError("real_gpu_training_not_confirmed")
    if request.contract.component_id not in GPU_CERTIFICATION_COMPONENTS:
        raise ValueError(
            f"component has no real GPU certification profile: {request.contract.component_id}"
        )
    if not request.adapter_hash or not request.ultralytics_version:
        raise ValueError("GPU certification requires adapter and Ultralytics identity")
    if source_payload.component_ids != [request.contract.component_id]:
        raise ValueError("source runtime payload component identity mismatch")
    model_path = _local_checkpoint(request.model)
    root = Path(request.workspace).resolve()
    fixture_root = root / "mini_coco"
    data_yaml = create_mini_coco_fixture(fixture_root)
    fixture = load_mini_coco_fixture_manifest(fixture_root)
    runtime_root = root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    project = root / "ultralytics"
    command = _training_command(
        model=model_path,
        data=data_yaml,
        project=project,
        name="initial",
        device=request.device,
        epochs=1,
    )
    adapter = ComponentAdapterRegistry().create_for_contract(request.contract)
    context = AdapterContext(
        contract=request.contract,
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=runtime_root,
        environment={
            "certification_mode": "gpu",
            "device": request.device,
        },
        options=_gpu_options(request, model_path=model_path, data_yaml=data_yaml),
    )
    preview = adapter.prepare_patch({}, {"imgsz": 640}, context, dry_run=False)
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash=request.protocol_hash,
        base_command=command,
        generated_config={
            "model_config": preview.patched_model_config,
            "training_config": preview.patched_training_config,
        },
    )
    if payload is None:
        raise ValueError("adapter returned no GPU runtime payload")
    if payload.component_ids != [request.contract.component_id]:
        raise ValueError("GPU runtime payload component identity mismatch")
    if payload.adapter_versions != source_payload.adapter_versions:
        raise ValueError("GPU runtime payload adapter version changed after CPU smoke")
    payload.verify_imports()
    payload_path = payload.write(
        runtime_root / f"adapter_runtime_payload.{payload.payload_hash[:12]}.yaml"
    )
    protocol = ComponentGPUProtocol(
        component_id=request.contract.component_id,
        adapter_hash=request.adapter_hash,
        runtime_payload_hash=payload.payload_hash,
        fixture_manifest_hash=fixture.fixture_hash,
        model_sha256=_sha256(model_path),
        ultralytics_version=request.ultralytics_version,
        device=request.device,
    )
    return PreparedComponentGPURun(
        protocol=protocol,
        data_yaml=data_yaml,
        fixture_manifest_path=fixture_root / "fixture_manifest.json",
        runtime_payload_path=payload_path,
        train_command=command,
        project_dir=project,
    )


def _gpu_options(
    request: ComponentSmokeWorkerRequest,
    *,
    model_path: Path,
    data_yaml: Path,
) -> dict[str, Any]:
    options = dict(request.options)
    options.update({"imgsz": 640, "device": request.device})
    if request.contract.component_id == "distillation.yolo26_teacher_student":
        teacher = options.get("teacher")
        if not teacher:
            raise ValueError(
                "distillation GPU certification requires --teacher pointing to a local "
                "yolo26s.pt or yolo26m.pt checkpoint"
            )
        teacher_path = _local_checkpoint(str(teacher))
        if teacher_path.name not in {"yolo26s.pt", "yolo26m.pt"}:
            raise ValueError("distillation teacher must be yolo26s.pt or yolo26m.pt")
        options.update(
            {
                "teacher": str(teacher_path),
                "student": str(model_path),
                "teacher_data": str(data_yaml),
                "student_data": str(data_yaml),
            }
        )
    return options


def _training_command(
    *,
    model: Path,
    data: Path,
    project: Path,
    name: str,
    device: str,
    epochs: int,
) -> list[str]:
    return [
        "yolo",
        "detect",
        "train",
        f"model={model.as_posix()}",
        f"data={data.as_posix()}",
        f"project={project.as_posix()}",
        f"name={name}",
        "exist_ok=True",
        f"epochs={epochs}",
        "imgsz=640",
        "batch=2",
        f"device={device}",
        "workers=0",
        "cache=False",
        "plots=False",
        "save=True",
        "val=False",
        "amp=True",
        "seed=17",
        "deterministic=True",
    ]


def _resume_training_command(
    *,
    checkpoint: Path,
    data: Path,
    project: Path,
    name: str,
    device: str,
) -> list[str]:
    command = _training_command(
        model=checkpoint,
        data=data,
        project=project,
        name=name,
        device=device,
        epochs=2,
    )
    command.append(f"resume={checkpoint.as_posix()}")
    return command


def _results_show_backward(path: Path) -> bool:
    import csv
    import math

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    loss_values: list[float] = []
    for key, value in rows[-1].items():
        if "train/" not in key or "loss" not in key:
            continue
        try:
            loss_values.append(float(str(value).strip()))
        except ValueError:
            return False
    return bool(loss_values and all(math.isfinite(value) for value in loss_values))


def _required_hooks_observed(
    payload: AdapterRuntimePayload,
    evidence: PluginRuntimeEvidence,
) -> bool:
    if (
        evidence.payload_hash != payload.payload_hash
        or evidence.protocol_hash != payload.protocol_hash
        or evidence.component_ids != payload.component_ids
        or evidence.failures
    ):
        return False
    return all(
        evidence.hook_call_counts.get(reference.reference, {}).get(hook, 0) > 0
        for reference in payload.plugin_references
        for hook in reference.required_hooks
    )


def _adapter_artifacts_complete(
    payload: AdapterRuntimePayload,
    runtime_root: Path,
    artifacts: dict[str, Path],
) -> bool:
    complete = True
    for expected in payload.expected_artifacts:
        path = runtime_root / expected.relative_path
        if path.is_file():
            artifacts[f"adapter_{expected.name}"] = path
        elif expected.required:
            complete = False
    return complete


def _atomic_gpu_evidence(
    evidence: ComponentGPUCertificationEvidence,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    evidence.to_yaml(temporary, exclude_none=True, sort_keys=False)
    temporary.replace(path)
    return path


def _local_checkpoint(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(
            f"GPU certification requires a local checkpoint and will not download: {value}"
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ComponentGPUExecutionBackend",
    "GPU_CERTIFICATION_COMPONENTS",
    "GPUCheckpointState",
    "GPUTrainingStageResult",
    "PreparedComponentGPURun",
    "RealComponentGPUExecutionBackend",
    "prepare_component_gpu_run",
    "run_real_component_gpu_certification",
]
