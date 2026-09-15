"""Paper distillation/domain-adaptation audit tests.

Covers the per-paper audit contract for the teacher/domain side: scope comes
only from the frozen plan, routes resolve only to paper-specific mechanism
IDs (generic branches stay primitives), behavior probes are real CPU checks,
required-asset records never block code readiness, and the two most-repeated
generic IDs are explicitly audited as non-paper evidence.
"""

from __future__ import annotations

from yolo_agent.research.paper_domain_side import (
    PaperDomainSideAuditBuilder,
    render_paper_83_domain_side_status,
)


def test_audit_builds_and_reports_per_paper_status() -> None:
    audit = PaperDomainSideAuditBuilder(workspace=".").build()
    assert audit.paper_count == 83
    assert audit.domain_side_paper_count == 72
    assert audit.distillation_paper_count == 32
    assert audit.domain_adaptation_paper_count == 40
    assert audit.semi_supervised_paper_count == 0
    assert audit.summary["out_of_scope"] == 11
    # The plan itself blocks 14 distillation papers for missing evidence.
    assert audit.summary["blocked_missing_evidence"] == 14
    assert audit.summary["ready"] == 58
    assert audit.audit_hash

    by_id = {item.paper_id: item for item in audit.records}
    ready_da = by_id["arxiv:2210.11539"]
    assert ready_da.in_scope
    assert ready_da.status == "ready"
    assert ready_da.resolved_route_ids == ["domain_adaptation.2210_11539"]
    assert ready_da.behavior.passed

    out_of_scope = by_id["arxiv:2103.14259"]  # assignment domain
    assert out_of_scope.status == "out_of_scope"
    assert not out_of_scope.paper_specific_config
    assert out_of_scope.behavior.checks["domain_side_scope"] is False


def test_blocked_records_fail_with_missing_evidence_not_fake_mechanisms() -> None:
    audit = PaperDomainSideAuditBuilder(workspace=".").build()
    blocked = [r for r in audit.records if r.status == "blocked_missing_evidence"]
    assert len(blocked) == 14
    for record in blocked:
        assert record.blockers, record.paper_id
        assert all(
            blocker.startswith("blocked_missing_evidence:")
            for blocker in record.blockers
        )
        # A blocked paper must not pretend to have a certified paper route:
        # any resolved route is identity-recovery, i.e. an explicitly blocked
        # generic-branch fallback rather than a paper-specific mechanism.
        for status in record.method_identity_statuses:
            assert status == "identity_recovery", record.paper_id
        assert "paper_specific_domain_mechanism" in " ".join(record.blockers)


def test_required_assets_block_reproduction_not_code_readiness() -> None:
    audit = PaperDomainSideAuditBuilder(workspace=".").build()
    in_scope = [r for r in audit.records if r.in_scope]
    assert in_scope
    for record in in_scope:
        assert record.required_assets, record.paper_id
        for asset in record.required_assets:
            assert not asset.available_at_runtime
            assert asset.blocked_for_real_reproduction
            assert asset.recovery_action.strip()
        # The flag is a reproduction gap only: code readiness is unaffected.
        assert record.status in {"ready", "blocked_missing_evidence"}
        assert not any(
            "required_asset" in blocker for blocker in record.blockers
        )
    kinds = {asset.asset_kind for r in in_scope for asset in r.required_assets}
    assert "teacher_checkpoint" in kinds
    assert "target_domain_data" in kinds


def test_generic_ids_are_audited_as_primitives_not_paper_routes() -> None:
    audit = PaperDomainSideAuditBuilder(workspace=".").build()
    in_scope = [r for r in audit.records if r.in_scope]
    generic = {
        "distillation.yolo26_teacher_student",
        "domain_adaptation.general",
    }
    for record in in_scope:
        if record.status != "ready":
            continue
        assert record.resolved_route_ids
        assert not (set(record.resolved_route_ids) & generic), record.paper_id
        assert "identity_recovery" not in record.method_identity_statuses


def test_every_in_scope_record_carries_behavior_evidence() -> None:
    audit = PaperDomainSideAuditBuilder(workspace=".").build()
    for record in audit.records:
        if not record.in_scope:
            continue
        assert record.evidence_refs, record.paper_id
        assert record.paper_specific_config
        if record.status != "ready":
            continue
        assert record.behavior.passed, (record.paper_id, record.behavior.errors)
        assert record.behavior.checks
        assert record.behavior.observed_changes


def test_status_renderer_lists_all_papers_and_method_notes() -> None:
    audit = PaperDomainSideAuditBuilder(workspace=".").build()
    report = render_paper_83_domain_side_status(audit)
    assert "# Paper-83 Distillation/Domain-Adaptation Status" in report
    assert "arxiv:2210.11539" in report
    assert "out-of-scope" in report
    assert "does not train a model" in report
    assert "blocked_for_real_reproduction" in report
    assert all(record.paper_id in report for record in audit.records)
