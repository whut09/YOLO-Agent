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
            smoke = adapter.gpu_smoke_test(context)
        else:
            smoke = adapter.smoke_test(context)
        evidence_kind = smoke.evidence_kind
        checks.update(smoke.checks)
        errors.extend(smoke.errors)
        if (
            request.mode == "cpu"
            and request.contract.component_id == "sampling.small_object"
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
        if not smoke.passed and not errors:
            errors.append("adapter smoke failed without details")
        if smoke.passed and smoke.evidence_kind != "local":
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
