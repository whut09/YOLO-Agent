"""Isolated process entrypoint for component CPU and GPU smoke checks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from yolo_agent.certification.component_schemas import (
    ComponentSmokeWorkerReport,
    ComponentSmokeWorkerRequest,
)
from yolo_agent.components.adapters import AdapterContext, AdapterRuntimePayload
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.adapters.assigners.yolo26_assignment import ASSIGNMENT_SPECS
from yolo_agent.components.distillation import DISTILLATION_COMPONENTS
from yolo_agent.components.graph_mechanisms import GRAPH_COMPONENTS


_QUALITY_LOSS_REPORT_NAMES = {
    "loss.quality.iou_aware_classification": (
        "iou_aware_classification_cpu_golden_path.yaml"
    ),
    "loss.quality.correlation": "correlation_cpu_golden_path.yaml",
    "loss.calibration.bpc": "bpc_calibration_cpu_golden_path.yaml",
    "loss.quality.pseudo_iou": "pseudo_iou_cpu_golden_path.yaml",
    "loss.quality.localization_aware": (
        "localization_aware_classification_cpu_golden_path.yaml"
    ),
    "loss.boundary_aware": "boundary_aware_cpu_golden_path.yaml",
    "loss.localization.uncertainty_weighted": (
        "uncertainty_weighted_regression_cpu_golden_path.yaml"
    ),
    "loss.hard_negative_classification": (
        "hard_negative_classification_cpu_golden_path.yaml"
    ),
    "loss.class_balanced_focal": "class_balanced_focal_cpu_golden_path.yaml",
}


def run_component_smoke_worker(
    request: ComponentSmokeWorkerRequest,
) -> ComponentSmokeWorkerReport:
    """Execute one adapter smoke in the current isolated process."""
    errors: list[str] = []
    checks: dict[str, bool | str | int | float] = {}
    evidence_kind = "mock"
    cuda_available = False
    payload_hash = "unavailable"
    try:
        payload = AdapterRuntimePayload.read(
            request.runtime_payload_path,
            verify_imports=True,
        )
        payload_hash = payload.payload_hash
        if payload.protocol_hash != request.protocol_hash:
            raise ValueError("worker runtime payload protocol mismatch")
        if request.contract.component_id not in payload.component_ids:
            raise ValueError("worker runtime payload component mismatch")
        adapter = ComponentAdapterRegistry().create_for_contract(request.contract)
        context = AdapterContext(
            contract=request.contract,
            detector_family="yolo26",
            head="one_to_one",
            imgsz=640,
            workspace=request.workspace,
            environment={
                "certification_mode": request.mode,
                "device": request.device,
            },
            options=dict(request.options),
        )
        if request.mode == "gpu":
            cuda_available = _cuda_available()
            checks["cuda_available"] = cuda_available
            if not cuda_available:
                raise RuntimeError("cuda_not_available")
            from yolo_agent.certification.component_gpu import (
                run_real_component_gpu_certification,
            )

            gpu_evidence = run_real_component_gpu_certification(
                request,
                payload,
            )
            payload_hash = gpu_evidence.runtime_payload_hash
            evidence_kind = "local"
            checks.update(gpu_evidence.checks)
            checks["gpu_evidence_path"] = str(
                Path(request.workspace).resolve() / "component_gpu_evidence.yaml"
            )
            if gpu_evidence.resources is not None:
                checks.update(
                    {
                        "gpu_name": gpu_evidence.resources.gpu_name,
                        "total_vram_mb": gpu_evidence.resources.total_vram_mb,
                        "peak_vram_mb": gpu_evidence.resources.peak_vram_mb,
                        "latency_ms": gpu_evidence.resources.latency_ms,
                        "model_size_mb": gpu_evidence.resources.model_size_mb,
                    }
                )
            errors.extend(gpu_evidence.errors)
            smoke = None
        else:
            smoke = adapter.smoke_test(context)
        if smoke is not None:
            evidence_kind = smoke.evidence_kind
            checks.update(smoke.checks)
            errors.extend(smoke.errors)
        if (
            request.mode == "cpu"
            and request.contract.component_id == "sampling.small_object"
            and smoke is not None
            and smoke.passed
            and smoke.evidence_kind == "local"
        ):
            from yolo_agent.certification.small_object_sampling import (
                run_small_object_sampling_cpu_fixture,
            )

            golden = run_small_object_sampling_cpu_fixture(
                runtime_payload_path=request.runtime_payload_path,
                workspace=request.workspace,
            )
            checks.update(golden.checks)
            checks["cpu_golden_path_report"] = str(
                Path(request.workspace).resolve()
                / "small_object_sampling_cpu_golden_path.yaml"
            )
            errors.extend(golden.errors)
        if (
            request.mode == "cpu"
            and request.contract.component_id
            in _QUALITY_LOSS_REPORT_NAMES
            and smoke.passed
            and smoke.evidence_kind == "local"
        ):
            from yolo_agent.certification.quality_loss import (
                run_quality_loss_cpu_fixture,
            )

            golden = run_quality_loss_cpu_fixture(
                runtime_payload_path=request.runtime_payload_path,
                workspace=request.workspace,
            )
            checks.update(golden.checks)
            checks["cpu_golden_path_report"] = str(
                Path(request.workspace).resolve()
                / _QUALITY_LOSS_REPORT_NAMES[request.contract.component_id]
            )
            errors.extend(golden.errors)
        if (
            request.mode == "cpu"
            and request.contract.component_id == "inference.sahi_slicing"
            and smoke.passed
            and smoke.evidence_kind == "local"
        ):
            checks.update(
                _run_sahi_cpu_smoke(
                    payload=payload,
                    workspace=request.workspace,
                )
            )
        if (
            request.mode == "cpu"
            and request.contract.component_id
            in ASSIGNMENT_SPECS
            and smoke.passed
            and smoke.evidence_kind == "local"
        ):
            from yolo_agent.certification.assignment_shadow import (
                run_assignment_shadow_cpu_fixture,
            )

            golden = run_assignment_shadow_cpu_fixture(
                runtime_payload_path=request.runtime_payload_path,
                workspace=request.workspace,
            )
            checks.update(golden.checks)
            checks.update(
                {
                    f"shadow_{name}": value
                    for name, value in golden.metrics.items()
                }
            )
            checks["cpu_golden_path_report"] = str(
                Path(request.workspace).resolve()
                / f"assignment_{golden.method}_shadow_cpu_golden_path.yaml"
            )
            errors.extend(golden.errors)
        if (
            request.mode == "cpu"
            and request.contract.component_id
            in {"distillation.yolo26_teacher_student", *DISTILLATION_COMPONENTS}
            and smoke.passed
            and smoke.evidence_kind == "local"
        ):
            from yolo_agent.certification.distillation import (
                run_distillation_cpu_fixture,
            )

            golden = run_distillation_cpu_fixture(
                runtime_payload_path=request.runtime_payload_path,
                workspace=request.workspace,
            )
            checks.update(golden.checks)
            checks["cpu_golden_path_report"] = str(
                Path(request.workspace).resolve()
                / (
                    "distillation_cpu_golden_path.yaml"
                    if request.contract.component_id
                    == "distillation.yolo26_teacher_student"
                    else "distillation_"
                    + DISTILLATION_COMPONENTS[
                        request.contract.component_id
                    ].mechanism
                    + "_cpu_golden_path.yaml"
                )
            )
            errors.extend(golden.errors)
        if (
            request.mode == "cpu"
            and request.contract.component_id == "head.p2_small_object"
            and smoke.passed
            and smoke.evidence_kind == "local"
        ):
            from yolo_agent.certification.p2_graph import (
                run_p2_graph_cpu_fixture,
            )

            golden = run_p2_graph_cpu_fixture(
                runtime_payload_path=request.runtime_payload_path,
                workspace=request.workspace,
            )
            checks.update(golden.checks)
            checks["cpu_golden_path_report"] = str(
                Path(request.workspace).resolve()
                / "p2_graph_cpu_golden_path.yaml"
            )
            errors.extend(golden.errors)
        if (
            request.mode == "cpu"
            and request.contract.component_id
            in GRAPH_COMPONENTS
            and smoke.passed
            and smoke.evidence_kind == "local"
        ):
            from yolo_agent.certification.neck_graph import (
                run_neck_graph_cpu_fixture,
            )

            golden = run_neck_graph_cpu_fixture(
                runtime_payload_path=request.runtime_payload_path,
                workspace=request.workspace,
            )
            checks.update(golden.checks)
            checks["cpu_golden_path_report"] = str(
                Path(request.workspace).resolve()
                / (
                    request.contract.component_id.replace(".", "_")
                    + "_cpu_golden_path.yaml"
                )
            )
            errors.extend(golden.errors)
        if smoke is not None and not smoke.passed and not errors:
            errors.append("adapter smoke failed without details")
        if smoke is not None and smoke.passed and smoke.evidence_kind != "local":
            errors.append("mock_smoke_evidence_cannot_certify_component")
    except (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return ComponentSmokeWorkerReport(
        component_id=request.contract.component_id,
        mode=request.mode,
        status="failed" if errors else "passed",
        protocol_hash=request.protocol_hash,
        payload_hash=payload_hash,
        evidence_kind=evidence_kind,
        process_id=os.getpid(),
        cuda_available=cuda_available,
        device=request.device,
        checks=checks,
        errors=errors,
    )


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _run_sahi_cpu_smoke(
    *,
    payload: AdapterRuntimePayload,
    workspace: Path,
) -> dict[str, bool | str | int | float]:
    """Exercise the actual optional SAHI and Ultralytics CPU path once."""
    from PIL import Image

    from yolo_agent.components.adapters.inference.sahi_backend import SlicingImage
    from yolo_agent.components.adapters.inference.slicing import (
        SahiRuntimeEvidence,
        SlicingInferenceConfig,
        SlicingInferenceRunner,
    )

    if not SlicingInferenceRunner.sahi_available():
        raise RuntimeError("optional dependency 'sahi' is not installed")
    if len(payload.inference_plugin) != 1:
        raise ValueError("SAHI CPU smoke requires one inference plugin")
    reference = payload.inference_plugin[0]
    config = SlicingInferenceConfig.model_validate(reference.options.get("config", {}))
    model_path = Path(config.model_path) if config.model_path else _payload_model(payload)
    if model_path is None or not model_path.is_file():
        raise ValueError("SAHI CPU smoke requires a local detector checkpoint")
    root = Path(workspace).resolve()
    image_path = root / "sahi_single_image.jpg"
    Image.new("RGB", (64, 64), color=(127, 127, 127)).save(image_path)
    runtime_config = config.model_copy(
        update={
            "model_path": str(model_path.resolve()),
            "device": "cpu",
            "slice_height": 64,
            "slice_width": 64,
            "overlap_height_ratio": 0.0,
            "overlap_width_ratio": 0.0,
        }
    )
    result = SlicingInferenceRunner().run(
        [SlicingImage(image_id=1, path=image_path)],
        runtime_config,
    )
    if result.status != "completed" or result.metrics is None:
        raise RuntimeError(result.reason or f"SAHI CPU smoke ended with {result.status}")
    plugin_type = reference.resolve()
    plugin = plugin_type(**reference.options) if isinstance(plugin_type, type) else plugin_type
    command = ["yolo-agent", "advanced", "certify-sahi", "--execute"]
    plugin.prepare_command(payload=payload, command=command, env={})
    evidence_path = Path(str(reference.options["evidence_path"]))
    evidence = SahiRuntimeEvidence.model_validate_json(
        evidence_path.read_text(encoding="utf-8-sig")
    )
    return {
        "real_sahi_inference": True,
        "runtime_hook_called": evidence.hook_call_counts.get("prepare_command", 0) == 1,
        "runtime_payload_bound": evidence.payload_hash == payload.payload_hash,
        "training_attribution_isolated": not evidence.training_attribution_allowed,
        "sliced_latency_ms": result.metrics.sliced_latency_ms,
    }


def _payload_model(payload: AdapterRuntimePayload) -> Path | None:
    for token in payload.base_command:
        key, separator, value = token.partition("=")
        if separator and key == "model":
            return Path(value)
    return None


def _atomic_report(report: ComponentSmokeWorkerReport, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    report.to_yaml(temporary, exclude_none=True, sort_keys=False)
    temporary.replace(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yolo-agent-component-worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = ComponentSmokeWorkerRequest.from_yaml(args.request)
    report = run_component_smoke_worker(request)
    _atomic_report(report, args.output)
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_component_smoke_worker"]
