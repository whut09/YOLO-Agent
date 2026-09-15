"""Paper loss/assignment-side audit tests.

Covers the per-paper audit contract: scope comes only from the frozen plan's
domains, routes resolve only through the provenance registry, behavior probes
are real CPU math checks, and readiness stays per paper.
"""

from __future__ import annotations

from yolo_agent.research.paper_loss_side import (
    PaperLossSideAuditBuilder,
    render_paper_83_loss_side_status,
    resolve_loss_side_route,
)


def test_audit_builds_and_reports_per_paper_status() -> None:
    audit = PaperLossSideAuditBuilder().build()
    assert audit.paper_count == 83
    assert audit.loss_side_paper_count == 8
    assert audit.summary["ready"] == 8
    assert audit.summary["blocked_missing_evidence"] == 0
    assert audit.summary["out_of_scope"] == 75
    assert audit.audit_hash

    by_id = {item.paper_id: item for item in audit.records}
    assert by_id["arxiv:2104.14082"].resolved_route_ids == ["loss.quality.pseudo_iou"]
    assert by_id["arxiv:2103.14259"].resolved_route_ids == ["assigner.optimal_transport"]
    assert by_id["arxiv:2203.16250"].resolved_route_ids == ["assigner.task_aligned"]
    assert by_id["arxiv:2208.00817"].resolved_route_ids == ["assigner.dynamic_smooth_label"]
    assert by_id["arxiv:2303.14404"].status == "ready"
    out_of_scope = by_id["arxiv:2108.07755"]  # domain head, not loss/assignment
    assert out_of_scope.status == "out_of_scope"
    assert not out_of_scope.paper_specific_config
    assert out_of_scope.behavior.checks["loss_side_scope"] is False


def test_every_in_scope_record_carries_behavior_evidence() -> None:
    audit = PaperLossSideAuditBuilder().build()
    for record in audit.records:
        if not record.in_scope:
            continue
        assert record.behavior.passed
        assert record.behavior.checks
        assert record.resolved_route_ids
        assert record.component_ids
        assert record.evidence_refs
        assert record.paper_specific_config


def test_status_renderer_lists_all_papers_and_method_notes() -> None:
    audit = PaperLossSideAuditBuilder().build()
    report = render_paper_83_loss_side_status(audit)
    assert "# Paper-83 Loss/Assignment-side Status" in report
    assert "arxiv:2103.14259" in report
    assert "out-of-scope" in report
    assert "does not train a model" in report


def test_route_resolution_rejects_unknown_mechanisms() -> None:
    assert resolve_loss_side_route("loss.bbox.wiou") is None
    assert resolve_loss_side_route("loss.bbox.mpdiou") is None
    assert resolve_loss_side_route("loss.bbox.nwd") is None
    assert resolve_loss_side_route("totally_unknown_mechanism") is None
    assert resolve_loss_side_route("data.quality_filter") is None


def test_tood_tal_paper_profile_differs_from_baseline_profile() -> None:

    from tests.assignment_fixtures import assignment_inputs
    from yolo_agent.components.assignment import compare_assignments
    from yolo_agent.loss_math.tal_reference import (
        ppyoloe_tal_profile,
        tal_reference_assignment,
        yolo_baseline_profile,
    )

    inputs = assignment_inputs()
    baseline = tal_reference_assignment(inputs, yolo_baseline_profile())
    paper = tal_reference_assignment(inputs, ppyoloe_tal_profile())
    comparison = compare_assignments(baseline, paper)
    assert comparison.foreground_disagreement_count > 0

    paper_scores = paper.target_scores[0][paper.foreground_mask[0]]
    paper_labels = paper.target_labels[0][paper.foreground_mask[0]]
    quality = paper_scores.gather(-1, paper_labels.unsqueeze(-1).long()).squeeze(-1)
    assert float(quality.max()) <= 1.0
    assert float(quality.min()) > 0.0
    baseline_scores = baseline.target_scores[0][baseline.foreground_mask[0]]
    baseline_labels = baseline.target_labels[0][baseline.foreground_mask[0]]
    baseline_quality = baseline_scores.gather(
        -1, baseline_labels.unsqueeze(-1).long()
    ).squeeze(-1)
    # Raw-IoU baseline targets vs normalized-alignment paper targets differ.
    assert abs(float(quality.mean()) - float(baseline_quality.mean())) > 1e-6


def test_loss_probe_uses_shared_math_matrix_for_every_loss_route() -> None:
    from yolo_agent.loss_math.loss_math_matrix import evaluate_loss_math_matrix
    from yolo_agent.components.auxiliary_losses import build_auxiliary_loss

    for loss_name in ("pseudo_iou", "mutual_supervision", "correlation", "bpc_calibration"):
        results = evaluate_loss_math_matrix(loss_name, build_loss=build_auxiliary_loss)
        assert results, loss_name
        for result in results:
            assert result.passed, f"{loss_name}/{result.case}: {result.errors}"
