"""Tests for the Paper-83 per-paper exactness audit.

The exactness audit may legitimately complete below 83/83 ready.  These tests
assert that the audit is still produced completely: every paper inventoried,
the 17-check matrix present, blocker categories from the fixed vocabulary, a
gap queue with actionable entries, and artifacts that round-trip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from yolo_agent.research.paper_exactness import (
    build_paper_exactness_audit,
    build_paper_exactness_gap_queue,
)
from yolo_agent.research.paper_exactness_report import (
    render_paper_83_exactness_audit,
    write_paper_exactness_artifacts,
)
from yolo_agent.research.paper_exactness_schemas import (
    BLOCKER_CATEGORY_ORDER,
    BLOCKING_STATUSES,
    EXACTNESS_CHECK_COUNT,
    EXACTNESS_CHECK_IDS,
    IMPLEMENTATION_READY,
    OUT_OF_SCOPE_STATUS,
    ExactnessCheckResult,
    ExactnessGapEntry,
    ExactnessPaperRecord,
    validate_ready_status_vocabulary,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/research/paper_83_manifest.yaml"

_AUDIT_SLOW = pytest.mark.slow if hasattr(pytest.mark, "slow") else pytest.mark.noop


@pytest.fixture(scope="module")
def audit():
    return build_paper_exactness_audit()


@pytest.fixture(scope="module")
def queue(audit):
    return build_paper_exactness_gap_queue(audit)


def test_audit_covers_all_frozen_papers(audit) -> None:
    assert audit.paper_count == 83
    assert len(audit.records) == 83
    ids = [record.paper_id for record in audit.records]
    assert ids == sorted(set(ids))
    assert audit.audit_hash


def test_every_record_inventories_all_17_checks(audit) -> None:
    for record in audit.records:
        assert set(record.checks) == set(EXACTNESS_CHECK_IDS)
        assert len(record.checks) == EXACTNESS_CHECK_COUNT
        assert record.manifest_membership_hash == audit.manifest_membership_hash


def test_status_vocabulary_is_the_fixed_blocker_set(audit) -> None:
    allowed = {IMPLEMENTATION_READY, OUT_OF_SCOPE_STATUS, *BLOCKER_CATEGORY_ORDER}
    assert allowed == {
        "implementation_ready",
        "out_of_scope",
        "blocked_missing_evidence",
        "blocked_missing_code",
        "blocked_runtime",
        "blocked_test",
        "blocked_compatibility",
        "blocked_asset",
        "blocked_license",
    }
    for record in audit.records:
        assert record.status in allowed, record.paper_id


def test_component_level_terms_are_never_ready_synonyms(audit) -> None:
    forbidden = {"covered", "mapped", "certified_adapter"}
    for record in audit.records:
        assert record.status not in forbidden
        assert record.status in {
            IMPLEMENTATION_READY,
            OUT_OF_SCOPE_STATUS,
            *BLOCKING_STATUSES,
        }
    for term in sorted(forbidden):
        assert validate_ready_status_vocabulary(term) is not None
    assert validate_ready_status_vocabulary("implementation_ready") is None


def test_summary_counts_are_consistent(audit) -> None:
    summary = audit.summary
    assert summary.total == 83
    assert summary.ready + summary.blocked + summary.out_of_scope == summary.total
    assert sum(summary.by_status.values()) == summary.total
    # The ready count must equal the campaign registry's ready count: the
    # exactness audit joins evidence, it never promotes a paper.
    from yolo_agent.research.paper_implementation_registry import (
        build_paper_implementation_registry,
    )

    registry = build_paper_implementation_registry(
        manifest_path=MANIFEST_PATH,
    )
    assert summary.ready == registry.implementation_ready_count


def test_ready_records_pass_all_checks_and_carry_no_blockers(audit) -> None:
    ready = [record for record in audit.records if record.status == IMPLEMENTATION_READY]
    assert ready, "campaign registry certifies ready papers; audit must join them"
    for record in ready:
        assert record.failed_checks == []
        assert record.passed_check_count == EXACTNESS_CHECK_COUNT
        assert record.blockers == []
        assert record.implementation_fingerprint


def test_blocked_records_carry_categorized_blockers(audit) -> None:
    blocked = [record for record in audit.records if record.status in BLOCKING_STATUSES]
    assert blocked
    for record in blocked:
        assert record.blockers, record.paper_id
        for blocker in record.blockers:
            assert blocker.startswith(record.status), blocker
        assert record.failed_checks


def test_known_fraud_classes_fail_their_checks(audit) -> None:
    """The 14 generic-only papers must be blocked_missing_code, never ready."""

    generic_only = [
        record
        for record in audit.records
        if record.evidence_inventory.get("implementation_evidence_class")
        == "generic_only"
    ]
    assert len(generic_only) == 14
    for record in generic_only:
        assert record.status == "blocked_missing_code", record.paper_id
        assert "not_generic_only" in record.failed_checks


def test_side_audit_fingerprints_are_joined(audit) -> None:
    graph_ready = [
        record
        for record in audit.records
        if record.side_status.get("graph") == "ready"
    ]
    assert len(graph_ready) == 3  # TOOD head, RTMDet neck, Gold-YOLO pyramid
    for record in graph_ready:
        assert record.side_audit_fingerprints["graph"]


def test_gap_queue_covers_every_blocked_paper(audit, queue) -> None:
    blocked_ids = {
        record.paper_id
        for record in audit.records
        if record.status in BLOCKING_STATUSES
    }
    queue_ids = {gap.paper_id for gap in queue.gaps}
    assert queue.blocked_paper_count == len(blocked_ids)
    assert queue_ids == blocked_ids
    assert queue.gaps == sorted(
        queue.gaps, key=lambda item: (item.paper_id, item.missing_requirement)
    )
    assert queue.exactness_audit_hash == audit.audit_hash
    assert queue.queue_hash == queue.calculate_hash()


def test_gap_queue_entries_are_actionable(queue) -> None:
    for gap in queue.gaps:
        assert gap.blocking_reason.strip()
        assert gap.missing_requirement.strip()
        assert gap.recommended_fix.strip()
        assert gap.estimated_scope in {"small", "medium", "large"}
        assert gap.dependency.strip()
        category, check_id = gap.missing_requirement.split(":", 1)
        assert category in BLOCKER_CATEGORY_ORDER
        assert check_id in EXACTNESS_CHECK_IDS
    assert sum(queue.gaps_by_category.values()) == len(queue.gaps)


def test_artifacts_round_trip(tmp_path, audit, queue) -> None:
    yaml_path, queue_path, markdown_path = write_paper_exactness_artifacts(
        audit,
        queue,
        yaml_path=tmp_path / "audit.yaml",
        queue_path=tmp_path / "gaps.yaml",
        markdown_path=tmp_path / "status.md",
    )
    from yolo_agent.research.paper_exactness_schemas import (
        PaperExactnessAudit,
        PaperExactnessGapQueue,
    )

    reloaded = PaperExactnessAudit.from_yaml(yaml_path)
    assert reloaded.audit_hash == audit.audit_hash
    assert reloaded.paper_count == 83

    reloaded_queue = PaperExactnessGapQueue.from_yaml(queue_path)
    assert reloaded_queue.queue_hash == queue.queue_hash
    assert len(reloaded_queue.gaps) == len(queue.gaps)

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Total: 83" in markdown
    assert f"Ready: {audit.summary.ready}" in markdown
    assert f"Blocked: {audit.summary.blocked}" in markdown


def test_rendered_report_groups_blockers_by_category(audit, queue) -> None:
    report = render_paper_83_exactness_audit(audit, queue)
    for category in BLOCKER_CATEGORY_ORDER:
        if any(record.status == category for record in audit.records):
            assert f"`{category}`" in report
    assert "Training remains locked" in report


def test_cli_handler_writes_artifacts_before_nonzero(tmp_path) -> None:
    from yolo_agent.cli import run_research_paper_83_exactness_audit_command

    args = argparse.Namespace(
        manifest=MANIFEST_PATH,
        method_coverage=ROOT / "research/production/paper_method_coverage.yaml",
        inventory=ROOT / "runs/coverage-audit/paper_execution_inventory.yaml",
        coverage=ROOT / "research/production/coverage_baseline.yaml",
        contracts=ROOT / "research/production/component_contracts.yaml",
        tests_root=ROOT / "tests",
        output=tmp_path / "audit.yaml",
        gap_queue=tmp_path / "gaps.yaml",
        markdown=tmp_path / "status.md",
        expected_paper_count=83,
    )
    exit_code = run_research_paper_83_exactness_audit_command(args)
    # The audit may be incomplete; it must still produce every artifact first.
    assert (tmp_path / "audit.yaml").is_file()
    assert (tmp_path / "gaps.yaml").is_file()
    assert (tmp_path / "status.md").is_file()
    assert exit_code in {0, 1}
    if audit_ready_count(tmp_path / "audit.yaml") < 83:
        assert exit_code == 1


def audit_ready_count(path: Path) -> int:
    import yaml

    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file)
    return data["summary"]["ready"]


# -- schema guardrails -------------------------------------------------------


def _record(
    *, status: str, checks: dict, blockers: list[str], fingerprint: str | None = None
) -> ExactnessPaperRecord:
    base_checks = {
        check_id: ExactnessCheckResult(
            check_id=check_id, passed=True, source="test"
        )
        for check_id in EXACTNESS_CHECK_IDS
    }
    base_checks.update(checks)
    return ExactnessPaperRecord(
        paper_id="arxiv:2103.14259",
        title="OTA",
        year=2021,
        primary_domain="assignment",
        manifest_membership_hash="0" * 64,
        in_scope_for_implementation=True,
        checks=base_checks,
        status=status,  # type: ignore[arg-type]
        blockers=blockers,
        implementation_fingerprint=fingerprint,
    )


def test_ready_record_with_failing_check_is_rejected() -> None:
    with pytest.raises(ValueError, match="failing checks"):
        _record(
            status="implementation_ready",
            checks={
                "not_no_op": ExactnessCheckResult(
                    check_id="not_no_op", passed=False, source="test"
                )
            },
            blockers=[],
            fingerprint="a" * 64,
        )


def test_blocked_record_without_blockers_is_rejected() -> None:
    with pytest.raises(ValueError, match="blocker"):
        _record(
            status="blocked_runtime",
            checks={
                "not_no_op": ExactnessCheckResult(
                    check_id="not_no_op", passed=False, source="test"
                )
            },
            blockers=[],
            fingerprint="a" * 64,
        )


def test_blocker_without_category_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="blocker-category prefix"):
        _record(
            status="blocked_runtime",
            checks={
                "not_no_op": ExactnessCheckResult(
                    check_id="not_no_op", passed=False, source="test"
                )
            },
            blockers=["some vague reason"],
            fingerprint="a" * 64,
        )


def test_out_of_scope_record_with_blockers_is_rejected() -> None:
    with pytest.raises(ValueError, match="out-of-scope"):
        _record(status="out_of_scope", checks={}, blockers=["blocked_runtime:x"])


def test_gap_entry_requires_every_field() -> None:
    with pytest.raises(ValueError, match="recommended_fix"):
        ExactnessGapEntry(
            paper_id="x",
            blocking_reason="blocked_runtime:x",
            missing_requirement="blocked_runtime:not_no_op",
            recommended_fix="  ",
            estimated_scope="small",
            dependency="paper implementation",
        )


def test_gap_queue_rejects_unsorted_entries() -> None:
    from yolo_agent.research.paper_exactness_schemas import PaperExactnessGapQueue

    with pytest.raises(ValueError, match="sorted"):
        PaperExactnessGapQueue(
            paper_count=2,
            blocked_paper_count=2,
            gaps=[
                ExactnessGapEntry(
                    paper_id="zzz",
                    blocking_reason="blocked_runtime:zzz",
                    missing_requirement="blocked_runtime:not_no_op",
                    recommended_fix="wire runtime",
                    estimated_scope="small",
                    dependency="runtime work",
                ),
                ExactnessGapEntry(
                    paper_id="aaa",
                    blocking_reason="blocked_runtime:aaa",
                    missing_requirement="blocked_runtime:not_no_op",
                    recommended_fix="wire runtime",
                    estimated_scope="small",
                    dependency="runtime work",
                ),
            ],
        )
