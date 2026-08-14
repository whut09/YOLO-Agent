from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.paper_proposal_ledger import (
    PaperCandidateCoverageLedger,
    planned_recipe_disposition,
)


def _queued_record():
    return planned_recipe_disposition(
        run_id="paper-run",
        round_index=1,
        recipe_id="yolo26_quality",
        recipe_version="v1.0.0",
        component_ids=["loss.quality.correlation"],
        decision="selected",
        reasons=[],
        execution_fingerprint="fingerprint-1",
        candidate_id="paper_recipe_yolo26_quality_v1_0_0",
    )


def test_evidence_recovery_update_remains_schema_valid(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(_queued_record())

    updated = ledger.update_disposition(
        execution_fingerprint="fingerprint-1",
        disposition="evidence_recovery",
        reason_codes=["target_error_facts_missing"],
        source_stage="asha_registration",
    )

    assert updated is not None
    assert updated.required_evidence == ["target_error_facts_missing"]
    reloaded = ledger.read().records[0]
    assert reloaded.disposition == "evidence_recovery"
    assert reloaded.required_evidence == ["target_error_facts_missing"]


def test_implementation_request_update_names_required_adapter(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
    )
    ledger.upsert(_queued_record())

    updated = ledger.update_disposition(
        execution_fingerprint="fingerprint-1",
        disposition="implementation_request",
        reason_codes=["runtime_adapter_missing"],
        source_stage="materialization",
    )

    assert updated is not None
    assert updated.required_adapters == ["adapter_for:loss.quality.correlation"]
    assert ledger.read().records[0].disposition == "implementation_request"


def test_reconcile_rejects_silent_candidate_drop(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
    )
    ledger.upsert(_queued_record())

    try:
        ledger.reconcile(["fingerprint-1", "fingerprint-missing"])
    except RuntimeError as exc:
        assert "fingerprint-missing" in str(exc)
    else:
        raise AssertionError("silent candidate drop was not rejected")
