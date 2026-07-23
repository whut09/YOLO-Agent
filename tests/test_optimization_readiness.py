"""Offline contracts for certification-gated automatic exploration."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.code_identity import certification_code_hash
from yolo_agent.certification.schemas import (
    CertificationCapabilityClaim,
    CertificationReport,
    CertificationStage,
)
from yolo_agent.core.optimization_readiness import OptimizationReadinessGate


REQUIRED_STAGES = {
    "environment",
    "train_entrypoint",
    "debug",
    "pilot_3_control",
    "pilot_3_candidates",
    "post_eval",
    "error_facts",
    "paired_delta",
    "asha_decision",
    "pilot_10",
    "catalog_import",
    "snapshot_creation",
    "diagnosis_linked_paper_prior",
    "eligibility_gate",
    "executable_recipe",
    "policy_memory_update",
    "recipe_execution_contract",
}


def _report(path: Path, *, code_hash: str | None = None) -> Path:
    report = CertificationReport(
        certification_id="mini",
        level="mini_gpu_pilot",
        status="passed",
        model="yolo26n.pt",
        data_yaml="mini.yaml",
        device="mock",
        protocol_hash="protocol",
        certified_code_hash=code_hash or certification_code_hash(),
        executed_recipe_id="reduce_mosaic",
        executed_changed_variable="mosaic",
        stages=[CertificationStage(stage_id=item, status="passed") for item in sorted(REQUIRED_STAGES)],
        capability_claims=[
            CertificationCapabilityClaim(
                capability_id=item,
                local_reproduction="locally_pilot_reproduced",
                certification_level="mini_gpu_pilot",
                recipe_id="reduce_mosaic",
                snapshot_hash="snapshot",
                evidence_hash=f"evidence-{item}",
            )
            for item in OptimizationReadinessGate.required_capabilities
        ],
    )
    report.to_yaml(path, exclude_none=True, sort_keys=False)
    return path


def test_missing_certification_blocks_exploration(tmp_path: Path) -> None:
    result = OptimizationReadinessGate().evaluate(run_root=tmp_path, execute=True)

    assert result.ready is False
    assert result.mode == "blocked"
    assert result.blockers == ["gpu_certification_report_missing"]


def test_matching_passed_certification_allows_exploration(tmp_path: Path) -> None:
    path = _report(tmp_path / "certification_report.yaml")

    result = OptimizationReadinessGate().evaluate(
        run_root=tmp_path,
        execute=True,
        report_path=path,
    )

    assert result.ready is True
    assert result.mode == "certified_exploration"
    assert result.certification_report_hash
    assert set(result.observed_capabilities) == set(OptimizationReadinessGate.required_capabilities)


def test_code_change_invalidates_old_certification(tmp_path: Path) -> None:
    path = _report(tmp_path / "certification_report.yaml", code_hash="stale-code")

    result = OptimizationReadinessGate().evaluate(
        run_root=tmp_path,
        execute=True,
        report_path=path,
    )

    assert result.ready is False
    assert "gpu_certification_code_hash_mismatch" in result.blockers


def test_dry_run_never_claims_training_authorization(tmp_path: Path) -> None:
    result = OptimizationReadinessGate().evaluate(run_root=tmp_path, execute=False)

    assert result.ready is True
    assert result.mode == "dry_run"
    assert result.warnings == ["dry_run does not authorize candidate training"]
