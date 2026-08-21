from __future__ import annotations

from yolo_agent.certification.automatic_runtime_readiness import (
    AutomaticRuntimeReadinessGate,
)
from yolo_agent.components.adapters import SmokeTestResult


def test_local_smoke_cache_requires_exact_payload_and_protocol(tmp_path) -> None:  # type: ignore[no-untyped-def]
    gate = AutomaticRuntimeReadinessGate(tmp_path / "cache")
    smoke = SmokeTestResult(
        passed=True,
        evidence_kind="local",
        checks={"shape_forward": True},
    )
    gate.record_smoke(
        component_id="loss.quality.correlation",
        adapter_hash="adapter-a",
        runtime_payload_hash="payload-a",
        protocol_hash="protocol-a",
        result=smoke,
    )

    assert gate.lookup_smoke(
        component_id="loss.quality.correlation",
        adapter_hash="adapter-a",
        runtime_payload_hash="payload-a",
        protocol_hash="protocol-a",
    ) == smoke
    assert gate.lookup_smoke(
        component_id="loss.quality.correlation",
        adapter_hash="adapter-a",
        runtime_payload_hash="payload-b",
        protocol_hash="protocol-a",
    ) is None


def test_runtime_gate_reports_cpu_ready_without_accuracy_authorization(tmp_path) -> None:  # type: ignore[no-untyped-def]
    gate = AutomaticRuntimeReadinessGate(tmp_path / "cache")
    smoke = SmokeTestResult(
        passed=True,
        evidence_kind="local",
        checks={"shape_forward": True},
    )
    path = gate.record_smoke(
        component_id="loss.quality.correlation",
        adapter_hash="adapter-a",
        runtime_payload_hash="payload-a",
        protocol_hash="protocol-a",
        result=smoke,
    )
    record = gate._read(  # type: ignore[attr-defined]
        gate._identity(  # type: ignore[attr-defined]
            scope="component_smoke",
            component_ids=["loss.quality.correlation"],
            adapter_hashes={"loss.quality.correlation": "adapter-a"},
            runtime_payload_hash="payload-a",
            protocol_hash="protocol-a",
        )
    )
    assert path.is_file()
    assert record is not None
    assert record.readiness_state == "cpu_ready"
    assert record.passed is True
    assert gate.lookup_smoke(
        component_id="loss.quality.correlation",
        adapter_hash="adapter-a",
        runtime_payload_hash="payload-a",
        protocol_hash="protocol-b",
    ) is None


def test_mock_or_failed_smoke_is_never_reused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    gate = AutomaticRuntimeReadinessGate(tmp_path / "cache")
    for result in (
        SmokeTestResult(passed=True, evidence_kind="mock"),
        SmokeTestResult(passed=False, evidence_kind="local", errors=["shape failed"]),
    ):
        gate.record_smoke(
            component_id="neck.rtmdet_large_kernel",
            adapter_hash="adapter",
            runtime_payload_hash="payload",
            protocol_hash="protocol",
            result=result,
        )
        assert gate.lookup_smoke(
            component_id="neck.rtmdet_large_kernel",
            adapter_hash="adapter",
            runtime_payload_hash="payload",
            protocol_hash="protocol",
        ) is None
