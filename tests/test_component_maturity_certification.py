from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from yolo_agent.certification.schemas import (
    CertificationObjectiveResult,
    CertificationReport,
    CertificationStage,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.components.maturity_certification import apply_certification_report
from yolo_agent.components.maturity_registry import ComponentMaturityRegistry


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
}


def _artifact(component_id: str, target: str, path: Path) -> ComponentMaturityArtifact:
    artifact_types = {
        "smoke_passed": "smoke_report",
        "pilot_reproduced": "pilot_paired_result",
    }
    return ComponentMaturityArtifact(
        component_id=component_id,
        target_maturity=target,
        artifact_type=artifact_types[target],
        artifact_path=path,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status="passed",
        producer="pytest_fixture",
    )


def _contract(tmp_path: Path, maturity: str) -> ComponentContract:
    evidence = tmp_path / f"{maturity}.json"
    evidence.write_text("{}", encoding="utf-8")
    artifacts = []
    if maturity == "smoke_passed":
        artifacts = [_artifact("sampling.small_object", "smoke_passed", evidence)]
    elif maturity == "pilot_reproduced":
        smoke = tmp_path / "smoke.json"
        smoke.write_text("{}", encoding="utf-8")
        artifacts = [
            _artifact("sampling.small_object", "smoke_passed", smoke),
            _artifact("sampling.small_object", "pilot_reproduced", evidence),
        ]
    return ComponentContract(
        component_id="sampling.small_object",
        display_name="Small-object sampling",
        category="sampling",
        implementation_path="local",
        adapter_class="SmallObjectSamplingAdapter",
        maturity=maturity,
        maturity_artifacts=artifacts,
    )


def _report(path: Path, *, status: str, full: bool = False) -> Path:
    objective = None
    if full:
        objective = CertificationObjectiveResult(
            objective_hash="objective-1",
            passed=True,
            baseline_seeds=[1, 2, 3],
            candidate_seeds=[1, 2, 3],
            dataset_manifest_hash="dataset",
            subset_manifest_hash="subset",
            seed_policy_hash="seeds",
            batch_policy_hash="batch",
            ultralytics_version="test",
            eval_protocol_hash="eval",
            paired_bootstrap_ci=(0.001, 0.02),
            cross_seed_confidence_interval=(0.002, 0.018),
            latency_guard_passed=True,
            model_size_guard_passed=True,
        )
    report = CertificationReport(
        certification_id="cert-1",
        level="full_coco_multi_seed" if full else "mini_gpu_pilot",
        status=status,
        model="yolo26n.pt",
        data_yaml="coco.yaml",
        device="0",
        protocol_hash="protocol-1",
        executed_recipe_id="small-object-recipe",
        stages=(
            [CertificationStage(stage_id=item, status="passed") for item in sorted(REQUIRED_STAGES)]
            if status == "passed"
            else []
        ),
        objective=objective,
        failures=[] if status == "passed" else ["post-eval failed"],
    )
    report.to_yaml(path, exclude_none=True, sort_keys=False)
    return path


def test_failed_gpu_certification_is_retained_without_promotion(tmp_path: Path) -> None:
    contract = _contract(tmp_path, "adapter_implemented")
    result = apply_certification_report(
        contract,
        _report(tmp_path / "failed.yaml", status="failed"),
        expected_recipe_id="small-object-recipe",
    )

    assert result.contract.maturity == "adapter_implemented"
    assert result.promoted_to == []
    assert result.retained_without_promotion == ["gpu_certified"]
    assert result.contract.maturity_artifacts[-1].status == "failed"


def test_passed_gpu_certification_requires_adjacent_smoke_state(tmp_path: Path) -> None:
    report_path = _report(tmp_path / "passed.yaml", status="passed")
    blocked = apply_certification_report(
        _contract(tmp_path, "adapter_implemented"),
        report_path,
        expected_recipe_id="small-object-recipe",
    )
    promoted = apply_certification_report(
        _contract(tmp_path, "smoke_passed"),
        report_path,
        expected_recipe_id="small-object-recipe",
    )

    assert blocked.contract.maturity == "adapter_implemented"
    assert blocked.retained_without_promotion == ["gpu_certified"]
    assert promoted.contract.maturity == "gpu_certified"
    assert promoted.promoted_to == ["gpu_certified"]


def test_full_multi_seed_report_advances_only_from_pilot_evidence(tmp_path: Path) -> None:
    result = apply_certification_report(
        _contract(tmp_path, "pilot_reproduced"),
        _report(tmp_path / "full.yaml", status="passed", full=True),
        expected_recipe_id="small-object-recipe",
    )

    assert result.contract.maturity == "confirmed_multi_seed"
    assert result.promoted_to == ["full_reproduced", "confirmed_multi_seed"]


def test_certification_recipe_identity_is_mandatory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="certification recipe mismatch"):
        apply_certification_report(
            _contract(tmp_path, "smoke_passed"),
            _report(tmp_path / "passed.yaml", status="passed"),
            expected_recipe_id="another-recipe",
        )


def test_failed_certification_is_persisted_without_registry_promotion(
    tmp_path: Path,
) -> None:
    registry = ComponentMaturityRegistry(tmp_path / "maturity-registry.yaml")

    result = apply_certification_report(
        _contract(tmp_path, "adapter_implemented"),
        _report(tmp_path / "failed.yaml", status="failed"),
        expected_recipe_id="small-object-recipe",
        maturity_registry=registry,
        adapter_hash="a" * 64,
        code_commit="commit-1",
        ultralytics_version="8.4.87",
    )

    overlay = registry.load().overlays[0]
    assert result.contract.maturity == "adapter_implemented"
    assert overlay.protocol_hash == "protocol-1"
    assert overlay.artifacts[-1].target_maturity == "gpu_certified"
    assert overlay.artifacts[-1].status == "failed"
