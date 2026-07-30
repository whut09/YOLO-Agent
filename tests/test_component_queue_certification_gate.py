"""Queue admission tests for component-specific GPU certification."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.certification.code_identity import certification_code_hash
from yolo_agent.certification.component_queue_gate import (
    ComponentQueueCertificationGate,
)
from yolo_agent.certification.schemas import (
    CertificationCapabilityClaim,
    CertificationObjectiveResult,
    CertificationPromotionResult,
    CertificationReport,
    CertificationStage,
)


BASE_STAGES = {
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
    "component_runtime_certification",
    "runtime_adapter",
    "paired_bootstrap",
    "promotion_gate",
}


def _sampling_report(path: Path, *, recipe_id: str = "small_object_sampling") -> Path:
    report = CertificationReport(
        certification_id="sampling-golden",
        level="mini_gpu_pilot",
        status="passed",
        model="yolo26n.pt",
        data_yaml="mini-coco.yaml",
        device="mock",
        protocol_hash="sampling-protocol",
        certified_code_hash=certification_code_hash(),
        executed_recipe_id="small_object_sampling",
        executed_changed_variable="data.sampling_policy",
        stages=[
            CertificationStage(stage_id=stage_id, status="passed")
            for stage_id in sorted(BASE_STAGES)
        ],
        objective=CertificationObjectiveResult(
            primary_metric="ap_small",
            observed_delta=0.01,
            passed=True,
            target_error_fact_deltas={"false_negative/object": 1.0},
        ),
        promotion_results=[
            CertificationPromotionResult(
                stage_id=stage_id,
                passed=True,
                primary_metric="ap_small",
            )
            for stage_id in ("pilot_3", "pilot_10")
        ],
        capability_claims=[
            CertificationCapabilityClaim(
                capability_id="small_object_sampling_runtime",
                local_reproduction="locally_pilot_reproduced",
                certification_level="mini_gpu_pilot",
                recipe_id=recipe_id,
                snapshot_hash="snapshot",
                evidence_hash="paired-result",
            )
        ],
    )
    report.to_yaml(path, exclude_none=True, sort_keys=False)
    return path


def test_sampling_component_is_blocked_without_end_to_end_report() -> None:
    result = ComponentQueueCertificationGate().evaluate(
        component_ids=["sampling.small_object"],
        report_path=None,
    )

    assert result.allowed is False
    assert result.blockers == ["sampling_end_to_end_certification_report_missing"]


def test_matching_sampling_report_allows_queue_admission(tmp_path: Path) -> None:
    result = ComponentQueueCertificationGate().evaluate(
        component_ids=["sampling.small_object"],
        report_path=_sampling_report(tmp_path / "certification_report.yaml"),
    )

    assert result.allowed is True
    assert result.report_hash
    assert all(result.checks.values())


def test_unrelated_recipe_claim_cannot_authorize_sampling(tmp_path: Path) -> None:
    result = ComponentQueueCertificationGate().evaluate(
        component_ids=["sampling.small_object"],
        report_path=_sampling_report(
            tmp_path / "certification_report.yaml",
            recipe_id="reduce_mosaic",
        ),
    )

    assert result.allowed is False
    assert (
        "sampling_end_to_end_certification_failed:sampling_capability_claimed"
        in result.blockers
    )


def test_unrelated_components_do_not_require_sampling_certification() -> None:
    result = ComponentQueueCertificationGate().evaluate(
        component_ids=["loss.quality.correlation"],
        report_path=None,
    )

    assert result.allowed is True
