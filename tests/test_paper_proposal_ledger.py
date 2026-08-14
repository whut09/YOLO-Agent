from __future__ import annotations

from pathlib import Path

import pytest

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


def test_ensure_runtime_candidate_recovers_missing_upstream_record(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )

    record = ledger.ensure_runtime_candidate(
        candidate_id="paper-candidate",
        recipe_id="paper-quality",
        recipe_version="v2",
        component_ids=["loss.quality.correlation"],
        execution_fingerprint="runtime-fingerprint",
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        source_stage="asha_registration",
        node_id="node-paper-candidate",
    )

    assert record.candidate_id == "paper-candidate"
    assert ledger.read().records[0].node_id == "node-paper-candidate"


def test_same_fingerprint_merges_paper_and_profile_provenance(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    first = _queued_record().model_copy(
        update={"paper_ids": ["paper-a"], "method_profile_ids": ["profile-a"]}
    )
    second = _queued_record().model_copy(
        update={"paper_ids": ["paper-b"], "method_profile_ids": ["profile-b"]}
    )

    merged = ledger.upsert_many([first, second]).records[0]

    assert merged.paper_ids == ["paper-a", "paper-b"]
    assert merged.method_profile_ids == ["profile-a", "profile-b"]


def test_same_fingerprint_cannot_bind_two_training_candidates(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(_queued_record())

    with pytest.raises(RuntimeError, match="candidate_id"):
        ledger.upsert(
            _queued_record().model_copy(update={"candidate_id": "different-candidate"})
        )


def test_same_fingerprint_cannot_change_recipe_identity(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(_queued_record())

    with pytest.raises(RuntimeError, match="recipe_id"):
        ledger.upsert(_queued_record().model_copy(update={"recipe_id": "other-recipe"}))


def test_ledger_rejects_artifact_from_another_protocol(tmp_path: Path) -> None:
    path = tmp_path / "paper_candidate_coverage.yaml"
    PaperCandidateCoverageLedger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-1",
    ).upsert(_queued_record())

    with pytest.raises(RuntimeError, match="protocol mismatch"):
        PaperCandidateCoverageLedger(
            path,
            run_id="paper-run",
            protocol_hash="protocol-2",
        ).read()


def test_ledger_rejects_artifact_from_another_run(tmp_path: Path) -> None:
    path = tmp_path / "paper_candidate_coverage.yaml"
    PaperCandidateCoverageLedger(path, run_id="paper-run").upsert(_queued_record())

    with pytest.raises(RuntimeError, match="run mismatch"):
        PaperCandidateCoverageLedger(path, run_id="other-run").read()


def test_disposition_updates_preserve_stage_history(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(_queued_record())

    ledger.update_disposition(
        execution_fingerprint="fingerprint-1",
        disposition="deferred_budget",
        reason_codes=["pilot_budget_exhausted"],
        source_stage="asha_registration",
    )

    record = ledger.read().records[0]
    assert [event.source_stage for event in record.stage_history] == [
        "paper_recipe_planner",
        "asha_registration",
    ]
    assert [event.disposition for event in record.stage_history] == [
        "queued",
        "deferred_budget",
    ]
