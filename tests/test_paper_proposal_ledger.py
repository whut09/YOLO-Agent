from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.agents.paper_proposal_ledger import (
    PaperCandidateCoverageLedger,
    planned_recipe_disposition,
    supersede_stale_ledger,
)
from yolo_agent.agents.paper_proposal_schemas import PaperProposalStageEvent
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
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


def _paper_inventory(*, paper_id: str = "paper-a") -> PaperExecutionInventory:
    record = PaperExecutionSpec(
        paper_id=paper_id,
        profile_id=f"profile-{paper_id}",
        title=f"Title {paper_id}",
        source_locations=[f"papers.yaml#{paper_id}"],
        canonical_component_ids=["loss.quality.correlation"],
        paper_specific_mechanism_ids=["quality_correlation"],
        recipe_ids=["yolo26_quality"],
        execution_fingerprint="a" * 64,
        current_disposition="implementation_request",
        disposition_reason="adapter evidence is incomplete",
        required_evidence=["runtime_payload"],
    )
    return PaperExecutionInventory(
        source_method_coverage_hash="b" * 64,
        all_paper_count=1,
        compatible_paper_count=1,
        exact_reproduction_candidates=0,
        records=[record],
    ).with_hash()


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


def test_runtime_candidate_replaces_reserved_trial_with_registered_identity(
    tmp_path: Path,
) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    common = {
        "candidate_id": "paper-candidate",
        "recipe_id": "paper-quality",
        "recipe_version": "v2",
        "component_ids": ["loss.quality.correlation"],
        "execution_fingerprint": "runtime-fingerprint",
        "node_id": "node-paper-candidate",
    }
    deferred = ledger.ensure_runtime_candidate(
        **common,
        disposition="deferred_budget",
        reason_codes=["round_budget_deferred"],
        source_stage="materialization_input",
    )

    registered = ledger.ensure_runtime_candidate(
        **common,
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        source_stage="asha_registration",
        asha_trial_id="paper-run:paper-candidate",
    )

    assert deferred.asha_trial_id == "paper-run:paper:paper-candidate"
    assert registered.asha_trial_id == "paper-run:paper-candidate"
    assert ledger.read().records[0].asha_trial_id == "paper-run:paper-candidate"


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
    assert ledger.execution_provenance() == {
        "fingerprint-1": ["paper-a", "paper-b"]
    }


def test_execution_cohort_audit_rejects_a_silent_fingerprint_drop(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
    )
    ledger.upsert(_queued_record())

    with pytest.raises(RuntimeError, match="silent execution drops"):
        ledger.assert_execution_cohort(["fingerprint-1", "fingerprint-missing"])


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


def test_materialized_candidate_rekeys_provisional_planner_identity(
    tmp_path: Path,
) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
    )
    ledger.upsert(
        _queued_record().model_copy(
            update={"candidate_id": "paper-candidate", "paper_ids": ["paper-a"]}
        )
    )

    updated = ledger.update_candidate_disposition(
        candidate_id="paper-candidate",
        execution_fingerprint="canonical-fingerprint",
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        source_stage="asha_registration",
        node_id="node-paper-candidate",
    )

    assert updated is not None
    assert updated.execution_fingerprint == "canonical-fingerprint"
    assert "execution_identity_reconciled" in updated.reason_codes
    records = ledger.read().records
    assert len(records) == 1
    assert records[0].execution_fingerprint == "canonical-fingerprint"


def test_failed_registration_binding_rebinds_to_the_registered_trial(
    tmp_path: Path,
) -> None:
    """A blocked_runtime binding is a recovery artifact, not a live claim.

    Regression: after the ASHA terminal-trial re-key released a failed
    registration's trial id, the successful re-registration recorded a new
    trial id and the ledger merge crashed with a fingerprint identity
    conflict on ``asha_trial_id`` even though the old binding came from a
    registration that never produced a runnable trial.
    """
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record().model_copy(
            update={
                "asha_trial_id": "old-run:paper_recipe_yolo26_quality_v1_0_0",
                "disposition": "blocked_runtime",
                "source_stage": "asha_registration",
                "reason_codes": ["asha_registration_failed:ValueError"],
            }
        )
    )

    updated = ledger.update_disposition(
        execution_fingerprint="fingerprint-1",
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        source_stage="asha_registration",
        asha_trial_id="new-run:paper:paper_recipe_yolo26_quality_v1_0_0",
    )

    assert updated is not None
    assert updated.asha_trial_id == "new-run:paper:paper_recipe_yolo26_quality_v1_0_0"
    record = ledger.read().records[0]
    assert record.asha_trial_id == "new-run:paper:paper_recipe_yolo26_quality_v1_0_0"
    assert any(
        event.boundary == "asha_registration"
        and event.asha_trial_id == "old-run:paper_recipe_yolo26_quality_v1_0_0"
        and event.disposition == "blocked_runtime"
        for event in record.stage_history
    )


def test_live_registration_binding_still_refuses_a_different_trial(
    tmp_path: Path,
) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record().model_copy(
            update={
                "asha_trial_id": "old-run:paper_recipe_yolo26_quality_v1_0_0",
                "source_stage": "asha_registration",
            }
        )
    )

    with pytest.raises(RuntimeError, match="asha_trial_id"):
        ledger.update_disposition(
            execution_fingerprint="fingerprint-1",
            disposition="queued",
            reason_codes=["asha_trial_registered"],
            source_stage="asha_registration",
            asha_trial_id="new-run:paper:paper_recipe_yolo26_quality_v1_0_0",
        )


def test_deferred_registration_binding_still_refuses_a_different_trial(
    tmp_path: Path,
) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record().model_copy(
            update={
                "asha_trial_id": "paper-run:paper:paper_recipe_yolo26_quality_v1_0_0",
                "disposition": "deferred_budget",
                "source_stage": "asha_registration",
                "reason_codes": ["asha_trial_registered_deferred_by_round_budget"],
            }
        )
    )

    with pytest.raises(RuntimeError, match="asha_trial_id"):
        ledger.update_disposition(
            execution_fingerprint="fingerprint-1",
            disposition="queued",
            reason_codes=["asha_trial_registered"],
            source_stage="asha_registration",
            asha_trial_id="paper-run:paper_recipe_yolo26_quality_v1_0_0",
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


def test_ledger_rejects_artifact_from_another_dataset_manifest(tmp_path: Path) -> None:
    path = tmp_path / "paper_candidate_coverage.yaml"
    PaperCandidateCoverageLedger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-1",
        dataset_manifest_hash="dataset-1",
    ).upsert(_queued_record())

    with pytest.raises(RuntimeError, match="dataset manifest mismatch"):
        PaperCandidateCoverageLedger(
            path,
            run_id="paper-run",
            protocol_hash="protocol-1",
            dataset_manifest_hash="dataset-2",
        ).read()


def test_ledger_rejects_artifact_from_another_run(tmp_path: Path) -> None:
    path = tmp_path / "paper_candidate_coverage.yaml"
    PaperCandidateCoverageLedger(path, run_id="paper-run").upsert(_queued_record())

    with pytest.raises(RuntimeError, match="run mismatch"):
        PaperCandidateCoverageLedger(path, run_id="other-run").read()


def test_supersede_stale_ledger_archives_superseded_protocol(tmp_path: Path) -> None:
    path = tmp_path / "paper_candidate_coverage.yaml"
    stale = PaperCandidateCoverageLedger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    stale.upsert(_queued_record())

    archived = supersede_stale_ledger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-2",
    )

    assert archived is not None
    assert archived.is_file()
    assert archived.read_text(encoding="utf-8").find("protocol-1") >= 0
    assert not path.exists()
    reopened = PaperCandidateCoverageLedger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-2",
    )
    reopened.upsert(_queued_record())
    assert reopened.read().records[0].candidate_id == "paper_recipe_yolo26_quality_v1_0_0"


def test_supersede_stale_ledger_archives_superseded_dataset_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper_candidate_coverage.yaml"
    PaperCandidateCoverageLedger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-1",
        dataset_manifest_hash="dataset-1",
    ).upsert(_queued_record())

    archived = supersede_stale_ledger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-1",
        dataset_manifest_hash="dataset-2",
    )

    assert archived is not None
    assert not path.exists()


def test_supersede_stale_ledger_keeps_current_protocol_artifact(tmp_path: Path) -> None:
    path = tmp_path / "paper_candidate_coverage.yaml"
    PaperCandidateCoverageLedger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-1",
        dataset_manifest_hash="dataset-1",
    ).upsert(_queued_record())

    archived = supersede_stale_ledger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-1",
        dataset_manifest_hash="dataset-1",
    )

    assert archived is None
    assert path.is_file()


def test_supersede_stale_ledger_tolerates_unknown_hashes(tmp_path: Path) -> None:
    path = tmp_path / "paper_candidate_coverage.yaml"
    PaperCandidateCoverageLedger(
        path,
        run_id="paper-run",
        protocol_hash="protocol-1",
    ).upsert(_queued_record())

    assert (
        supersede_stale_ledger(path, run_id="paper-run", protocol_hash="unknown")
        is None
    )
    assert (
        supersede_stale_ledger(path, run_id="other-run", protocol_hash="protocol-2")
        is None
    )
    assert path.is_file()


def test_supersede_stale_ledger_missing_file_is_noop(tmp_path: Path) -> None:
    missing = tmp_path / "paper_candidate_coverage.yaml"

    assert (
        supersede_stale_ledger(missing, run_id="paper-run", protocol_hash="protocol-1")
        is None
    )


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


def test_inventory_seed_persists_paper_denominator_across_candidate_updates(
    tmp_path: Path,
) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
        dataset_manifest_hash="dataset-1",
    )
    seeded = ledger.seed_inventory(_paper_inventory())

    ledger.upsert(_queued_record())
    reloaded = ledger.read()

    assert seeded.expected_paper_count == 1
    assert len(reloaded.paper_coverage) == 1
    assert reloaded.paper_coverage[0].paper_id == "paper-a"
    assert reloaded.paper_coverage[0].stage_history[0].boundary == "inventory"
    assert reloaded.dataset_manifest_hash == "dataset-1"


def test_inventory_seed_rejects_changed_paper_denominator(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
    )
    ledger.seed_inventory(_paper_inventory())

    with pytest.raises(RuntimeError, match="inventory hash mismatch"):
        ledger.seed_inventory(_paper_inventory(paper_id="paper-b"))


def test_planner_record_projects_to_one_current_paper_disposition(
    tmp_path: Path,
) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
        dataset_manifest_hash="dataset-1",
    )
    ledger.seed_inventory(_paper_inventory())
    ledger.upsert(
        _queued_record().model_copy(
            update={
                "paper_ids": ["paper-a"],
                "method_profile_ids": ["profile-paper-a"],
                "protocol_hash": "protocol-1",
                "dataset_manifest_hash": "dataset-1",
            }
        )
    )

    paper = ledger.read().current_by_paper["paper-a"]

    assert paper.disposition == "queued"
    assert paper.recipe_id == "yolo26_quality"
    assert [event.boundary for event in paper.stage_history] == [
        "inventory",
        "planner",
    ]


def test_boundary_seal_detects_and_fills_paper_level_silent_drop(
    tmp_path: Path,
) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
    )
    ledger.seed_inventory(_paper_inventory())

    with pytest.raises(RuntimeError, match="planner boundary has silent drops"):
        ledger.assert_boundary_complete("planner")

    sealed = ledger.seal_boundary("planner")

    assert sealed.paper_coverage[0].stage_history[-1].boundary == "planner"
    ledger.assert_boundary_complete("planner")


def test_stale_identity_trial_binding_rebinds_during_terminal_recording(
    tmp_path: Path,
) -> None:
    """A binding to a trial of a *different* execution identity is an artifact.

    Regression (fix 12): the classifier_response_distillation recipe evolved
    its execution fingerprint between child runs.  The r10 ledger kept the
    pre-evolution binding ``...:<old-fp12>`` (queued registration), while
    ASHA's fingerprint-identity rule kept the post-evolution trial
    ``...:<new-fp12>`` as the recovery record and returned it to the
    terminal-recording path.  The merge treated the stale binding as a live
    claim and crashed with a fingerprint identity conflict, killing the CLI
    after training had already completed.
    """
    new_fingerprint = "2" * 64
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record()
        .model_copy(
            update={
                "execution_fingerprint": new_fingerprint,
                "asha_trial_id": "first-training:paper:paper_recipe_yolo26_quality_v1_0_0:624eea39c7eb",
                "source_stage": "asha_registration",
            }
        )
    )

    updated = ledger.update_disposition(
        execution_fingerprint=new_fingerprint,
        disposition="already_tested",
        reason_codes=["verified_paired_result:completed"],
        source_stage="candidate_completion",
        asha_trial_id="first-training:paper:paper_recipe_yolo26_quality_v1_0_0:"
        + "2" * 12,
    )

    assert updated is not None
    assert (
        updated.asha_trial_id
        == "first-training:paper:paper_recipe_yolo26_quality_v1_0_0:" + "2" * 12
    )


def test_current_identity_trial_binding_still_refuses_a_different_trial(
    tmp_path: Path,
) -> None:
    """A binding whose fingerprint suffix matches this record stays live."""
    new_fingerprint = "2" * 64
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record()
        .model_copy(
            update={
                "execution_fingerprint": new_fingerprint,
                "asha_trial_id": "first-training:paper:paper_recipe_yolo26_quality_v1_0_0:"
                + "2" * 12,
                "source_stage": "asha_registration",
            }
        )
    )

    with pytest.raises(RuntimeError, match="asha_trial_id"):
        ledger.update_disposition(
            execution_fingerprint=new_fingerprint,
            disposition="already_tested",
            reason_codes=["verified_paired_result:completed"],
            source_stage="candidate_completion",
            asha_trial_id="first-training:paper:paper_recipe_yolo26_quality_v1_0_0:aaaaaaaaaaaa",
        )



def test_blocked_runtime_registration_does_not_block_identity_rebind(
    tmp_path: Path,
) -> None:
    """A blocked_runtime registration must not veto identity rebind (fix 12b).

    Regression: the r14 ledger record for classifier_response_distillation
    carried one legacy-format blocked_runtime registration plus live queued
    registrations of a *different* fingerprint trial.  The first exemption
    pass fail-closed on the legacy-format suffix before considering that it
    was a blocked_runtime artifact, so the terminal recording of the ASHA
    recovery trial still crashed with a fingerprint identity conflict.
    """
    new_fingerprint = "0" * 64
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    registration_events = [
        PaperProposalStageEvent(
            source_stage="asha_registration",
            boundary="asha_registration",
            disposition="queued",
            reason_codes=["asha_trial_registered"],
            execution_fingerprint=new_fingerprint,
            candidate_id="paper_recipe_yolo26_quality_v1_0_0",
            asha_trial_id=(
                "first-training:paper:paper_recipe_yolo26_quality_v1_0_0:708d98d5c900"
            ),
        ),
        PaperProposalStageEvent(
            source_stage="asha_registration",
            boundary="asha_registration",
            disposition="blocked_runtime",
            reason_codes=["asha_registration_failed:RuntimeError"],
            execution_fingerprint=new_fingerprint,
            candidate_id="paper_recipe_yolo26_quality_v1_0_0",
            asha_trial_id="first-training:paper:paper_recipe_yolo26_quality_v1_0_0",
        ),
    ]
    ledger.upsert(
        _queued_record().model_copy(
            update={
                "execution_fingerprint": new_fingerprint,
                "asha_trial_id": "first-training:paper:paper_recipe_yolo26_quality_v1_0_0:708d98d5c900",
                "source_stage": "asha_registration",
                "stage_history": registration_events,
            }
        )
    )

    updated = ledger.update_disposition(
        execution_fingerprint=new_fingerprint,
        disposition="already_tested",
        reason_codes=["verified_paired_result:completed"],
        source_stage="candidate_completion",
        asha_trial_id="first-training:paper:paper_recipe_yolo26_quality_v1_0_0:0b123c3ed1c6",
    )

    assert updated is not None
    assert (
        updated.asha_trial_id
        == "first-training:paper:paper_recipe_yolo26_quality_v1_0_0:0b123c3ed1c6"
    )


def _legacy_base_id_registration_event(fingerprint: str) -> PaperProposalStageEvent:
    """Build a queued registration bound to the pre-convention base trial id."""
    return PaperProposalStageEvent(
        source_stage="asha_registration",
        boundary="asha_registration",
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        execution_fingerprint=fingerprint,
        candidate_id="paper_recipe_yolo26_quality_v1_0_0",
        asha_trial_id="first-training:paper:paper_recipe_yolo26_quality_v1_0_0",
    )


def test_legacy_base_id_binding_rebinds_to_own_fingerprint_trial(
    tmp_path: Path,
) -> None:
    """A legacy base-id binding yields to the record's own fingerprint trial.

    Regression (r44): the distillation ledger row for fingerprint
    ``05ce612344f3c174...`` still carried the pre-convention base trial id
    (no ``:<fp12>`` suffix) from an early registration wave, while the trial
    that finally ran was the record's own fingerprint trial
    ``...:05ce612344f3``.  The merge failed closed on the legacy format and
    killed the CLI after training had already completed.  A legacy-format
    binding can never be a live claim of the record's current identity, so
    rebinding to the record's own fingerprint-suffixed trial is an identity
    correction, not a takeover.
    """
    fingerprint = "5" * 64
    base_trial = "first-training:paper:paper_recipe_yolo26_quality_v1_0_0"
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record()
        .model_copy(
            update={
                "execution_fingerprint": fingerprint,
                "asha_trial_id": base_trial,
                "source_stage": "asha_registration",
                "stage_history": [
                    _legacy_base_id_registration_event(fingerprint)
                ],
            }
        )
    )

    updated = ledger.update_disposition(
        execution_fingerprint=fingerprint,
        disposition="already_tested",
        reason_codes=["verified_paired_result:completed"],
        source_stage="candidate_completion",
        asha_trial_id=base_trial + ":" + "5" * 12,
    )

    assert updated is not None
    assert updated.asha_trial_id == base_trial + ":" + "5" * 12


def test_legacy_base_id_binding_still_refuses_a_foreign_trial(
    tmp_path: Path,
) -> None:
    """A foreign fingerprint trial must not take over a legacy-bound record."""
    fingerprint = "5" * 64
    base_trial = "first-training:paper:paper_recipe_yolo26_quality_v1_0_0"
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record()
        .model_copy(
            update={
                "execution_fingerprint": fingerprint,
                "asha_trial_id": base_trial,
                "source_stage": "asha_registration",
                "stage_history": [
                    _legacy_base_id_registration_event(fingerprint)
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="asha_trial_id"):
        ledger.update_disposition(
            execution_fingerprint=fingerprint,
            disposition="already_tested",
            reason_codes=["verified_paired_result:completed"],
            source_stage="candidate_completion",
            asha_trial_id=base_trial + ":aaaaaaaaaaaa",
        )


def _own_fingerprint_trial_registration_event(
    fingerprint: str,
) -> PaperProposalStageEvent:
    """Build a queued registration bound to the record's own fingerprint trial."""
    return PaperProposalStageEvent(
        source_stage="asha_registration",
        boundary="asha_registration",
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        execution_fingerprint=fingerprint,
        candidate_id="paper_recipe_yolo26_quality_v1_0_0",
        asha_trial_id=(
            "first-training:paper:paper_recipe_yolo26_quality_v1_0_0:"
            + fingerprint[:12]
        ),
    )


def test_own_fingerprint_trial_binding_yields_to_legacy_placeholder_id(
    tmp_path: Path,
) -> None:
    """A legacy placeholder id must not conflict with the live identity binding.

    Regression (r45, fix 14): the mark closure fabricated the legacy base id
    for identity-reserving dispositions whenever the caller did not pass a
    trial id (evidence_recovery).  The r45 ledger row for fingerprint
    ``6265a382381b...`` already bound the record's own fingerprint trial
    ``...:6265a382381b`` (reconciled registration), so the fabricated id
    raised a fingerprint identity conflict and killed the CLI during pilot
    registration - before any training ran.  A legacy-format id can never be
    a live claim of any execution, so the merge must keep the existing
    fingerprint-suffixed binding instead of failing closed.
    """
    fingerprint = "6" * 64
    base_trial = "first-training:paper:paper_recipe_yolo26_quality_v1_0_0"
    own_trial = base_trial + ":" + "6" * 12
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record()
        .model_copy(
            update={
                "execution_fingerprint": fingerprint,
                "asha_trial_id": own_trial,
                "source_stage": "asha_registration",
                "stage_history": [
                    _own_fingerprint_trial_registration_event(fingerprint)
                ],
            }
        )
    )

    updated = ledger.update_disposition(
        execution_fingerprint=fingerprint,
        disposition="evidence_recovery",
        reason_codes=["asha_trial_registered_without_valid_paired_evidence"],
        source_stage="asha_registration",
        asha_trial_id=base_trial,
    )

    assert updated is not None
    # The live identity binding is preserved, not regressed to the placeholder.
    assert updated.asha_trial_id == own_trial


def test_own_fingerprint_trial_binding_still_refuses_a_foreign_trial(
    tmp_path: Path,
) -> None:
    """A foreign fingerprint trial must not take over an identity-bound record."""
    fingerprint = "6" * 64
    base_trial = "first-training:paper:paper_recipe_yolo26_quality_v1_0_0"
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record()
        .model_copy(
            update={
                "execution_fingerprint": fingerprint,
                "asha_trial_id": base_trial + ":" + "6" * 12,
                "source_stage": "asha_registration",
                "stage_history": [
                    _own_fingerprint_trial_registration_event(fingerprint)
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="asha_trial_id"):
        ledger.update_disposition(
            execution_fingerprint=fingerprint,
            disposition="evidence_recovery",
            reason_codes=["asha_trial_registered_without_valid_paired_evidence"],
            source_stage="asha_registration",
            asha_trial_id=base_trial + ":aaaaaaaaaaaa",
        )


def test_two_legacy_base_ids_from_different_runs_merge(tmp_path: Path) -> None:
    """Two pre-convention placeholders must not fail closed against each other.

    Regression (r45, fix 17): the reconciled ledger row bound the r45
    placeholder ``first-training-r45:paper:<cand>`` while the main run's
    ASHA trial (created by an early registration wave under the same
    pre-convention id format) is ``first-training:paper:<cand>``.  Neither
    id carries a fingerprint suffix, so neither is a live claim of the
    record's identity - their collision must merge, not crash the CLI.
    """
    fingerprint = "7" * 64
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="first-training-r45",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record()
        .model_copy(
            update={
                "execution_fingerprint": fingerprint,
                "asha_trial_id": (
                    "first-training-r45:paper:paper_recipe_yolo26_quality_v1_0_0"
                ),
                "source_stage": "asha_registration",
                "stage_history": [
                    PaperProposalStageEvent(
                        source_stage="asha_registration",
                        boundary="asha_registration",
                        disposition="deferred_budget",
                        reason_codes=["native_fallback_deferred_for_adapter_methods"],
                        execution_fingerprint=fingerprint,
                        candidate_id="paper_recipe_yolo26_quality_v1_0_0",
                        asha_trial_id=(
                            "first-training:paper:paper_recipe_yolo26_quality_v1_0_0"
                        ),
                    )
                ],
            }
        )
    )

    updated = ledger.update_disposition(
        execution_fingerprint=fingerprint,
        disposition="blocked_runtime",
        reason_codes=["automatic_runtime_readiness_failed"],
        source_stage="runtime_readiness",
        asha_trial_id="first-training:paper:paper_recipe_yolo26_quality_v1_0_0",
    )

    assert updated is not None
    # The main run's binding (backed by a real ASHA trial) wins the merge.
    assert (
        updated.asha_trial_id
        == "first-training:paper:paper_recipe_yolo26_quality_v1_0_0"
    )


def test_legacy_base_id_still_refuses_foreign_fingerprint_trial(
    tmp_path: Path,
) -> None:
    """A legacy-vs-foreign-fingerprint collision stays fail-closed."""
    fingerprint = "7" * 64
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="first-training-r45",
        protocol_hash="protocol-1",
    )
    ledger.upsert(
        _queued_record()
        .model_copy(
            update={
                "execution_fingerprint": fingerprint,
                "asha_trial_id": (
                    "first-training-r45:paper:paper_recipe_yolo26_quality_v1_0_0"
                ),
                "source_stage": "asha_registration",
                "stage_history": [
                    PaperProposalStageEvent(
                        source_stage="asha_registration",
                        boundary="asha_registration",
                        disposition="deferred_budget",
                        reason_codes=["native_fallback_deferred_for_adapter_methods"],
                        execution_fingerprint=fingerprint,
                        candidate_id="paper_recipe_yolo26_quality_v1_0_0",
                        asha_trial_id=(
                            "first-training:paper:paper_recipe_yolo26_quality_v1_0_0"
                        ),
                    )
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="asha_trial_id"):
        ledger.update_disposition(
            execution_fingerprint=fingerprint,
            disposition="blocked_runtime",
            reason_codes=["automatic_runtime_readiness_failed"],
            source_stage="runtime_readiness",
            asha_trial_id="first-training:paper:paper_recipe_yolo26_quality_v1_0_0:aaaaaaaaaaaa",
        )


def test_merge_tolerates_equal_timestamp_events_with_none_fields(tmp_path: Path) -> None:
    """fix 19: equal created_at events must sort without a str/None TypeError.

    A legacy ledger event has None in optional identity fields (execution_fingerprint,
    asha_trial_id, ...) while the current event fills them.  When both share the same
    created_at, the secondary sort on _stage_event_key compared None against str and
    crashed the whole paper planner into failed_fallback_to_rule_loop.
    """
    from datetime import datetime, timezone

    stamp = datetime(2026, 10, 1, 9, 0, 0, tzinfo=timezone.utc)
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    base = _queued_record()
    legacy_event = PaperProposalStageEvent(
        source_stage="asha_registration",
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        paper_ids=[],
        execution_fingerprint=None,
        candidate_id=None,
        asha_trial_id=None,
        node_id=None,
        created_at=stamp,
    )
    legacy = base.model_copy(update={"stage_history": [legacy_event]})
    current_event = PaperProposalStageEvent(
        source_stage="asha_registration",
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        paper_ids=[],
        execution_fingerprint="fingerprint-1",
        candidate_id=base.candidate_id,
        asha_trial_id=None,
        node_id=None,
        created_at=stamp,
    )
    incoming = base.model_copy(update={"stage_history": [current_event]})

    coverage = ledger.upsert_many([legacy, incoming])

    merged = coverage.records[0]
    assert merged.execution_fingerprint == "fingerprint-1"
    stamp_events = [
        event for event in merged.stage_history if event.created_at == stamp
    ]
    assert len(stamp_events) == 2


def test_merge_tolerates_equal_timestamp_events_with_none_fields(tmp_path: Path) -> None:
    """fix 19: equal created_at events must sort without a str/None TypeError.

    A legacy ledger event has None in optional identity fields (execution_fingerprint,
    asha_trial_id, ...) while the current event fills them.  When both share the same
    created_at, the secondary sort on _stage_event_key compared None against str and
    crashed the whole paper planner into failed_fallback_to_rule_loop.
    """
    from datetime import datetime, timezone

    stamp = datetime(2026, 10, 1, 9, 0, 0, tzinfo=timezone.utc)
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="paper-run",
        protocol_hash="protocol-1",
    )
    base = _queued_record()
    legacy_event = PaperProposalStageEvent(
        source_stage="asha_registration",
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        paper_ids=[],
        execution_fingerprint=None,
        candidate_id=None,
        asha_trial_id=None,
        node_id=None,
        created_at=stamp,
    )
    legacy = base.model_copy(update={"stage_history": [legacy_event]})
    current_event = PaperProposalStageEvent(
        source_stage="asha_registration",
        disposition="queued",
        reason_codes=["asha_trial_registered"],
        paper_ids=[],
        execution_fingerprint="fingerprint-1",
        candidate_id=base.candidate_id,
        asha_trial_id=None,
        node_id=None,
        created_at=stamp,
    )
    incoming = base.model_copy(update={"stage_history": [current_event]})

    coverage = ledger.upsert_many([legacy, incoming])

    merged = coverage.records[0]
    assert merged.execution_fingerprint == "fingerprint-1"
    stamp_events = [
        event for event in merged.stage_history if event.created_at == stamp
    ]
    assert len(stamp_events) == 2
