"""Contracts for the small-object sampling CPU certification artifact."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.small_object_sampling_schemas import (
    SmallObjectSamplingCpuReport,
)


def test_passed_report_requires_complete_golden_path(tmp_path: Path) -> None:
    report = SmallObjectSamplingCpuReport(
        status="passed",
        protocol_hash="protocol",
        runtime_payload_hash="payload",
        sampler_manifest_path=tmp_path / "sampler_manifest.json",
        runtime_evidence_path=tmp_path / "plugin_runtime_evidence.json",
        sampler_state_path=tmp_path / "small_object_sampler_state.rank1.json",
        checks={
            "train_dataloader_hook_called": True,
            "sampler_manifest_verified": True,
            "ddp_deterministic_sharding": True,
            "resume_state_restored": True,
            "validation_loader_unchanged": True,
        },
    )

    path = report.to_yaml(tmp_path / "report.yaml", sort_keys=False)
    loaded = SmallObjectSamplingCpuReport.from_yaml(path)

    assert loaded.report_hash == report.report_hash


def test_passed_report_rejects_missing_runtime_check(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resume_state_restored"):
        SmallObjectSamplingCpuReport(
            status="passed",
            protocol_hash="protocol",
            runtime_payload_hash="payload",
            sampler_manifest_path=tmp_path / "sampler_manifest.json",
            runtime_evidence_path=tmp_path / "plugin_runtime_evidence.json",
            sampler_state_path=tmp_path / "state.json",
            checks={
                "train_dataloader_hook_called": True,
                "sampler_manifest_verified": True,
                "ddp_deterministic_sharding": True,
                "resume_state_restored": False,
                "validation_loader_unchanged": True,
            },
        )
