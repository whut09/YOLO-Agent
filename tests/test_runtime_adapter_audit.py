from __future__ import annotations

from pathlib import Path

from yolo_agent.tools.runtime_adapter_audit import (
    EXPECTED_RUNTIME_ADAPTERS,
    build_runtime_adapter_audit,
)


def test_runtime_adapter_audit_separates_payloads_from_observed_execution(
    tmp_path: Path,
) -> None:
    report = build_runtime_adapter_audit(
        registry_path=tmp_path / "empty_registry.yaml"
    )

    assert report.expected_count == 41
    assert report.audited_count == 41
    assert report.payload_implemented_count == 41
    assert report.runtime_observed_count == 0
    assert {item.component_id for item in report.records} == set(
        EXPECTED_RUNTIME_ADAPTERS
    )
    assert all(item.source_maturity == "adapter_implemented" for item in report.records)
    assert all(
        "artifact_backed_runtime_hook_not_observed" in item.blocked_by
        for item in report.records
    )


def test_runtime_adapter_audit_round_trips_yaml(tmp_path: Path) -> None:
    report = build_runtime_adapter_audit(
        registry_path=tmp_path / "empty_registry.yaml"
    )
    path = report.to_yaml(tmp_path / "runtime_adapter_audit.yaml")

    restored = type(report).from_yaml(path)

    assert restored == report
