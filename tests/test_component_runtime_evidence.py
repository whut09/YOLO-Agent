from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_agent.certification.component_runtime_backend import (
    build_component_runtime_launch,
)
from yolo_agent.certification.component_runtime_evidence import (
    validate_component_runtime_artifacts,
)
from yolo_agent.certification.runner import BackendRun


def _runtime_run(tmp_path: Path, component_id: str = "loss.quality.correlation") -> BackendRun:
    launch = build_component_runtime_launch(
        component_id=component_id,
        base_command=["yolo", "detect", "train", "model=yolo26n.pt", "imgsz=640"],
        workspace=tmp_path / "runtime",
        protocol_hash="protocol-1",
        options={},
    )
    for artifact in launch.payload.expected_artifacts:
        path = launch.runtime_artifacts[artifact.name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    hook_counts = {
        reference.reference: {hook: 1 for hook in reference.required_hooks}
        for reference in launch.payload.plugin_references
    }
    launch.runtime_artifacts["plugin_runtime_evidence"].write_text(
        json.dumps(
            {
                "payload_hash": launch.payload.payload_hash,
                "protocol_hash": "protocol-1",
                "component_ids": [component_id],
                "changed_variables": launch.payload.changed_variables,
                "ultralytics_version": "8.4.0",
                "signature_hash": "signature",
                "compatible": True,
                "hook_call_counts": hook_counts,
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    return BackendRun(
        candidate_id=component_id,
        node_id="candidate",
        run_dir=tmp_path,
        checkpoint=checkpoint,
        runtime_artifacts=launch.runtime_artifacts,
    )


def test_validate_component_runtime_artifacts_accepts_required_hook_evidence(
    tmp_path: Path,
) -> None:
    run = _runtime_run(tmp_path)

    payload, checks = validate_component_runtime_artifacts(
        run,
        component_id="loss.quality.correlation",
        protocol_hash="protocol-1",
    )

    assert payload.component_ids == ["loss.quality.correlation"]
    assert checks["required_hooks_complete"] is True
    assert checks["required_hook_calls"] == 1


def test_validate_component_runtime_artifacts_rejects_missing_hook(tmp_path: Path) -> None:
    run = _runtime_run(tmp_path)
    evidence_path = run.runtime_artifacts["plugin_runtime_evidence"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["hook_call_counts"] = {}
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing_hook"):
        validate_component_runtime_artifacts(
            run,
            component_id="loss.quality.correlation",
            protocol_hash="protocol-1",
        )


def test_validate_component_runtime_artifacts_rejects_missing_adapter_artifact(
    tmp_path: Path,
) -> None:
    run = _runtime_run(tmp_path)
    run.runtime_artifacts["auxiliary_loss_correlation_evidence"].unlink()

    with pytest.raises(RuntimeError, match="missing_artifact"):
        validate_component_runtime_artifacts(
            run,
            component_id="loss.quality.correlation",
            protocol_hash="protocol-1",
        )
