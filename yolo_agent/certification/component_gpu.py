"""Real CUDA certification harness for paper component runtime adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from yolo_agent.certification.component_schemas import (
    ComponentGPUProtocol,
    ComponentSmokeWorkerRequest,
)
from yolo_agent.certification.fixture import (
    create_mini_coco_fixture,
    load_mini_coco_fixture_manifest,
)
from yolo_agent.components.adapters import AdapterContext, AdapterRuntimePayload
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry


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
    "GPU_CERTIFICATION_COMPONENTS",
    "PreparedComponentGPURun",
    "prepare_component_gpu_run",
]
