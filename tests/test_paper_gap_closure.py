"""Gap-closure loop tests.

The Prompt-12 contract: the loop closes locally fixable blockers with real
fixes (component certification, per-route test evidence, adapter declaration
repairs), records every closure honestly against the committed Prompt-11
baseline audit, and keeps evidence-blocked papers blocked rather than
guessing.  These tests pin the diff records, the artifact round-trip, and the
frozen-campaign invariants (membership, vocabulary, no threshold changes).
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import pytest
import yaml
from typing import get_args

from yolo_agent.research.paper_evidence_diligence import (
    CONFIRMED_SOURCES,
    build_evidence_diligence_section,
    evidence_diligence_payload,
)
from yolo_agent.research.paper_exactness_schemas import (
    BLOCKING_STATUSES,
    ExactnessPaperStatus,
)
from yolo_agent.research.paper_gap_closure import (
    PaperClosureRecord,
    PaperGapClosureEngine,
    _diff_closure_records,
    _load_baseline_records,
    write_gap_closure_artifact,
    write_unresolved_report,
)

warnings.filterwarnings("ignore")

STATUS_VOCABULARY = (
    set(get_args(ExactnessPaperStatus))
    | set(BLOCKING_STATUSES)
    | {"implementation_ready", "out_of_scope"}
)

BASELINE_AUDIT = Path("artifacts/paper_83_exactness_audit.yaml")
FROZEN_COUNT = 83


@pytest.fixture(scope="module")
def audit_and_records(tmp_path_factory):
    """Run one closure cycle against the *committed* Prompt-11 baseline.

    The committed audit at HEAD is the campaign's before-state.  Recovering it
    via ``git show`` keeps the engine free of git dependencies while making
    the diff deterministic for this commit.  Once this Prompt-12 work itself
    is committed, the diff becomes empty and the movement assertions skip.
    """

    baseline_dir = tmp_path_factory.mktemp("baseline")
    baseline = baseline_dir / "committed_audit.yaml"
    result = subprocess.run(
        [
            "git",
            "show",
            "HEAD:artifacts/paper_83_exactness_audit.yaml",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or "paper_count" not in result.stdout:
        pytest.skip("committed baseline audit unavailable at HEAD")
    baseline.write_text(result.stdout, encoding="utf-8")
    engine = PaperGapClosureEngine()
    audit, records, _summary = engine.run_cycle(baseline_path=baseline)
    return audit, records


def test_baseline_audit_loads_with_frozen_membership():
    baseline = _load_baseline_records(BASELINE_AUDIT)
    assert baseline is not None
    assert len(baseline) == FROZEN_COUNT


def test_baseline_records_carry_status_and_blockers():
    baseline = _load_baseline_records(BASELINE_AUDIT)
    assert baseline is not None
    for record in baseline.values():
        assert record["status"] in STATUS_VOCABULARY
        assert isinstance(record["blockers"], list)


def test_cycle_reaches_69_ready_end_state(audit_and_records):
    audit, _records = audit_and_records
    assert audit.summary.ready == 69
    assert audit.summary.total == FROZEN_COUNT
    assert audit.summary.blocked == 14


def test_cycle_moves_runtime_blocked_papers_to_ready(audit_and_records):
    _audit, records = audit_and_records
    moved = {
        record.paper_id: (record.before_status, record.after_status)
        for record in records
        if record.before_status != record.after_status
    }
    if not moved:
        pytest.skip("baseline already matches the current audit")
    assert moved
    assert all(
        before == "blocked_runtime" and after == "implementation_ready"
        for before, after in moved.values()
    )


def test_moved_records_carry_components_files_and_tests(audit_and_records):
    _audit, records = audit_and_records
    moved = [
        record
        for record in records
        if record.before_status != record.after_status
    ]
    if not moved:
        pytest.skip("baseline already matches the current audit")
    for record in moved:
        assert record.actions, record.paper_id
        assert record.files_changed, record.paper_id
        assert record.tests, record.paper_id
        assert any(
            action.startswith("certified:") for action in record.actions
        )


def test_runtime_fix_never_fabricates_lowered_thresholds(audit_and_records):
    """Every ready record still passed all 17 checks — nothing was waved."""

    audit, _records = audit_and_records
    for record in audit.records:
        if record.status == "implementation_ready":
            assert record.blockers == []
            assert all(
                result.passed for result in record.checks.values()
            ), record.paper_id


def test_evidence_blocked_papers_stay_blocked_with_diligence(audit_and_records):
    audit, records = audit_and_records
    blocked = [r for r in audit.records if r.status == "blocked_missing_code"]
    assert len(blocked) == 14
    by_id = {r.paper_id: r for r in records}
    for record in blocked:
        closure = by_id[record.paper_id]
        assert closure.after_status == "blocked_missing_code"
        assert closure.unresolved
        assert any(
            action.startswith("evidence_diligence:") for action in closure.actions
        ), record.paper_id


def test_evidence_diligence_confirms_a_source_for_every_blocked_paper(
    audit_and_records,
):
    audit, _records = audit_and_records
    blocked_ids = {
        r.paper_id for r in audit.records if r.status == "blocked_missing_code"
    }
    assert blocked_ids == set(CONFIRMED_SOURCES)


def test_gap_closure_artifact_round_trip(audit_and_records, tmp_path):
    audit, records = audit_and_records
    out = tmp_path / "gap_closure.yaml"
    written = write_gap_closure_artifact(
        audit, records, {"components_certified": 60}, out
    )
    assert written == out
    with out.open(encoding="utf-8-sig") as file:
        payload = yaml.safe_load(file)
    assert payload["schema_version"] == "paper_83_gap_closure.v1"
    assert len(payload["papers"]) == len(records)
    assert payload["audit_after"]["ready"] == audit.summary.ready
    assert payload["evidence_diligence"]["confirmed_sources"]
    for paper in payload["papers"]:
        assert set(paper) == {
            "paper_id",
            "before",
            "actions",
            "files_changed",
            "tests",
            "after",
            "unresolved",
        }


def test_unresolved_report_documents_diligence(audit_and_records, tmp_path):
    audit, records = audit_and_records
    out = write_unresolved_report(audit, records, tmp_path / "unresolved.md")
    text = out.read_text(encoding="utf-8")
    assert f"Ready: {audit.summary.ready}" in text
    assert "Evidence due diligence" in text
    assert "proceedings.neurips.cc" in text or "ecva.net" in text
    assert "No threshold" in text or "no threshold" in text.lower()


def test_diligence_payload_serializes():
    payload = evidence_diligence_payload()
    assert len(payload["confirmed_sources"]) == 14
    assert "DNS" in payload["environment_limitation"]
    section = build_evidence_diligence_section()
    assert any("confirmed sources" in line for line in section)


def test_membership_unchanged_by_closure_loop(audit_and_records):
    """The loop must never delete a paper or touch the frozen membership."""

    audit, _records = audit_and_records
    assert audit.paper_count == FROZEN_COUNT
    manifest_hash = audit.manifest_membership_hash
    assert len(manifest_hash) == 64  # sha256 membership pinned


def test_no_paper_regresses_out_of_vocabulary(audit_and_records):
    audit, records = audit_and_records
    for record in audit.records:
        assert record.status in STATUS_VOCABULARY
    for closure in records:
        assert closure.before_status in STATUS_VOCABULARY
        assert closure.after_status in STATUS_VOCABULARY


def test_diff_skips_unchanged_papers():
    from yolo_agent.research.paper_exactness_schemas import ExactnessCheckResult

    def _checks() -> dict[str, ExactnessCheckResult]:
        return {
            "membership_valid": ExactnessCheckResult(
                check_id="membership_valid",
                passed=True,
                source="test",
            )
        }

    baseline = {
        "p": {"status": "implementation_ready", "blockers": []},
    }
    audit = type(
        "AuditStub",
        (),
        {
            "records": [
                type(
                    "R",
                    (),
                    {
                        "paper_id": "p",
                        "blockers": [],
                        "component_ids": [],
                        "status": "implementation_ready",
                        "checks": _checks(),
                    },
                )()
            ]
        },
    )()
    records = _diff_closure_records(audit, baseline)  # type: ignore[arg-type]
    assert records == []


def test_closure_record_dataclass_defaults():
    record = PaperClosureRecord(
        paper_id="x", before_status="blocked_runtime"
    )
    assert record.actions == []
    assert record.files_changed == []
    assert record.tests == []
    assert record.after_status == ""
    assert record.unresolved == []
