"""Capability maturity manifest and generated-document tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.schemas import (
    CertificationCapabilityClaim,
    CertificationObjectiveResult,
    CertificationReport,
    CertificationStage,
)
from yolo_agent.tools.capability_matrix import (
    CapabilityEntry,
    CapabilityManifest,
    generate,
    render_detail_document,
    render_readme_matrix,
    validate_certification_claims,
    validate_source_paths,
)


CONFIG_PATH = Path("configs/capability_maturity.yaml")


def test_capability_manifest_separates_execution_and_reproduction() -> None:
    manifest = CapabilityManifest.from_yaml(CONFIG_PATH)
    capabilities = {item.capability_id: item for item in manifest.capabilities}

    assert capabilities["pilot_auto_training"].status == "executable"
    assert capabilities["candidate_coco_error_facts"].status == "incomplete"
    assert capabilities["error_delta_next_round"].status == "partial"
    assert capabilities["three_seed_confirmation"].status == "supported_not_automatic"
    assert capabilities["stable_two_map_gain"].status == "not_guaranteed"
    assert capabilities["three_seed_confirmation"].code_present is True
    assert capabilities["three_seed_confirmation"].local_reproduction == "not_claimed"


def test_capability_manifest_source_references_exist() -> None:
    manifest = CapabilityManifest.from_yaml(CONFIG_PATH)
    assert validate_source_paths(manifest) == []


def test_generated_matrix_explains_real_boundaries() -> None:
    manifest = CapabilityManifest.from_yaml(CONFIG_PATH)
    matrix = render_readme_matrix(manifest, language="zh")
    document = render_detail_document(manifest)

    assert "代码存在" in matrix
    assert "自动执行" in matrix
    assert "本地复现" in matrix
    assert "`not guaranteed`" in matrix
    assert "代码存在不代表可以自动执行" in document
    assert "yolo_agent/agents/asha_scheduler.py" in document


def test_committed_capability_docs_are_current() -> None:
    assert generate(
        config_path=CONFIG_PATH,
        document_path=Path("docs/capability-maturity.md"),
        readme_path=Path("README.md"),
        readme_en_path=Path("README.en.md"),
        check=True,
    ) is True


def _certification_report(
    root: Path,
    *,
    capability_id: str,
    reproduction: str,
    full: bool = False,
) -> Path:
    required = {
        "environment", "train_entrypoint", "debug", "pilot_3_control", "pilot_3_candidates",
        "post_eval", "error_facts", "paired_delta", "asha_decision", "pilot_10",
    }
    level = "full_coco_multi_seed" if full else "mini_gpu_pilot"
    objective = CertificationObjectiveResult(
        objective_hash="objective",
        required_delta=0.02,
        observed_delta=0.025,
        baseline_seeds=[1, 2, 3],
        candidate_seeds=[1, 2, 3],
        passed=True,
    ) if full else None
    report = CertificationReport(
        certification_id="test-certification",
        level=level,
        status="passed",
        model="yolo26n.pt",
        data_yaml="coco.yaml",
        device="0",
        protocol_hash="protocol",
        stages=[CertificationStage(stage_id=stage, status="passed") for stage in sorted(required)],
        objective=objective,
        capability_claims=[CertificationCapabilityClaim(
            capability_id=capability_id,
            local_reproduction=reproduction,  # type: ignore[arg-type]
            certification_level=level,
        )],
    )
    path = root / "certification_report.yaml"
    report.to_yaml(path, exclude_none=True)
    return path


def _promoted_manifest(report: Path | None, reproduction: str) -> CapabilityManifest:
    return CapabilityManifest(
        schema_version=1,
        reviewed_at="2026-07-17",
        capabilities=[CapabilityEntry(
            capability_id="candidate_coco_error_facts",
            name_zh="候选 COCO error facts",
            name_en="Candidate COCO error facts",
            status="executable",
            code_present=True,
            automatic_execution="guarded",
            local_reproduction=reproduction,  # type: ignore[arg-type]
            boundary_zh="认证测试",
            boundary_en="certification test",
            source_paths=[Path("pyproject.toml")],
            certification_report=report,
        )],
    )


def test_reproduction_promotion_requires_certification_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires certification_report"):
        validate_certification_claims(_promoted_manifest(None, "locally_pilot_reproduced"), root=tmp_path)


def test_valid_mini_report_allows_pilot_reproduction(tmp_path: Path) -> None:
    path = _certification_report(
        tmp_path,
        capability_id="candidate_coco_error_facts",
        reproduction="locally_pilot_reproduced",
    )
    validate_certification_claims(
        _promoted_manifest(path.relative_to(tmp_path), "locally_pilot_reproduced"),
        root=tmp_path,
    )


def test_failed_report_cannot_promote_capability(tmp_path: Path) -> None:
    report = CertificationReport(
        certification_id="failed-certification",
        level="mini_gpu_pilot",
        status="failed",
        model="yolo26n.pt",
        data_yaml="coco.yaml",
        device="0",
        protocol_hash="protocol",
        failures=["post_eval_failed"],
    )
    path = tmp_path / "failed.yaml"
    report.to_yaml(path)
    with pytest.raises(ValueError, match="did not pass"):
        validate_certification_claims(
            _promoted_manifest(path.relative_to(tmp_path), "locally_pilot_reproduced"),
            root=tmp_path,
        )


def test_mini_report_cannot_claim_confirmed_multi_seed(tmp_path: Path) -> None:
    path = _certification_report(
        tmp_path,
        capability_id="candidate_coco_error_facts",
        reproduction="locally_pilot_reproduced",
    )
    manifest = _promoted_manifest(path.relative_to(tmp_path), "confirmed_multi_seed")
    with pytest.raises(ValueError, match="does not authorize"):
        validate_certification_claims(manifest, root=tmp_path)


def test_full_report_allows_confirmed_multi_seed(tmp_path: Path) -> None:
    path = _certification_report(
        tmp_path,
        capability_id="candidate_coco_error_facts",
        reproduction="confirmed_multi_seed",
        full=True,
    )
    validate_certification_claims(
        _promoted_manifest(path.relative_to(tmp_path), "confirmed_multi_seed"),
        root=tmp_path,
    )
