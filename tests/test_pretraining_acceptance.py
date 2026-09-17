"""Pre-training acceptance tests.

Pins the Prompt-16 contract: the acceptance system runs all five sections
(campaign integrity, action space, autonomous loop, safety probes, tests),
always writes its artifacts on PASS *and* FAIL, never mutates the frozen
manifest, never lowers a maturity bar, and never executes real training —
the 83/83-allow probe evaluates the gate only, with the trainer stubbed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.research.pretraining_acceptance import (
    PretrainingAcceptanceRunner,
    render_pretraining_acceptance,
    write_pretraining_acceptance_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def acceptance_no_tests():
    """Full acceptance run with test probes disabled (fast, deterministic)."""

    runner = PretrainingAcceptanceRunner(run_tests=False, project_root=REPO_ROOT)
    return runner.run()


def test_fail_verdict_still_produces_complete_record(acceptance_no_tests) -> None:
    """Current campaign state is 69/83 → FAIL — but every section is filled."""

    acceptance = acceptance_no_tests
    assert acceptance.verdict == "FAIL"
    assert acceptance.real_training_executed is False
    assert acceptance.paper_campaign.manifest_paper_count == 83
    assert acceptance.paper_campaign.unique_paper_ids == 83
    assert acceptance.paper_campaign.membership_hash_valid
    # Honest counts — never inflated to reach PASS.
    assert acceptance.paper_campaign.implementation_ready == 69
    assert acceptance.paper_campaign.blocked == 14
    assert not acceptance.paper_campaign.passed
    assert acceptance.training_gate.allowed is False
    assert "paper_campaign:implementation_ready" in acceptance.remaining_blockers


def test_membership_hash_valid_helper(acceptance_no_tests) -> None:
    assert acceptance_no_tests.paper_campaign.membership_hash_valid is True


def test_action_space_section_covers_all_required_families(
    acceptance_no_tests,
) -> None:
    section = acceptance_no_tests.optimization_action_space
    assert section.passed
    assert section.catalog_action_count >= 30
    assert section.paper_lineage_actions > 0
    assert section.local_actions > 0
    assert section.missing_families == []


def test_autonomous_loop_section_proves_multi_round_delta_loop(
    acceptance_no_tests,
) -> None:
    section = acceptance_no_tests.autonomous_loop
    assert section.passed
    # The fake loop must complete multiple rounds with the full decision
    # vocabulary — not a single-module smoke.
    assert section.multi_round_completed
    assert section.rollback_observed
    assert section.pareto_axes_used
    assert "promote" in section.decisions_seen
    assert "rollback" in section.decisions_seen
    assert section.stop_policy_honest


def test_safety_probes_pass_with_stubbed_trainer(acceptance_no_tests) -> None:
    section = acceptance_no_tests.safety
    assert section.gate_fail_closed_on_missing_manifest
    assert section.gate_fail_closed_on_hash_mismatch
    assert section.gate_82_of_83_blocks
    # The 83/83 allow probe evaluated the gate only — no trainer ran.
    assert section.gate_83_of_83_allows
    assert section.full_run_consent_boundary_preserved
    assert section.passed


def test_artifacts_written_on_fail(tmp_path) -> None:
    """FAIL verdicts still write both artifacts — the Prompt-16 core rule."""

    runner = PretrainingAcceptanceRunner(
        run_tests=False,
        project_root=REPO_ROOT,
    )
    acceptance = runner.run()
    yaml_path = tmp_path / "pretraining_acceptance.yaml"
    md_path = tmp_path / "PRETRAINING_ACCEPTANCE.md"
    written_yaml, written_md = write_pretraining_acceptance_artifacts(
        acceptance,
        yaml_path=yaml_path,
        markdown_path=md_path,
    )
    assert written_yaml.is_file()
    assert written_md.is_file()
    text = written_md.read_text(encoding="utf-8")
    assert "YOLO AGENT PRE-TRAINING ACCEPTANCE" in text
    assert "Training gate:              LOCKED" in text
    assert "REAL TRAINING EXECUTED:     NO" in text
    assert "Remaining blockers:" in text


def test_render_table_pass_layout_with_synthetic_full_ready(
    tmp_path, acceptance_no_tests
) -> None:
    """A synthetic 83/83 record renders UNLOCKED without changing the manifest."""

    acceptance = acceptance_no_tests.model_copy(deep=True)
    campaign = acceptance.paper_campaign
    campaign.implementation_ready = 83
    campaign.blocked = 0
    campaign.passed = True
    campaign.checks = [
        type(check)(requirement=check.requirement, expected=check.expected, observed=check.observed, passed=True)
        for check in campaign.checks
    ]
    acceptance.safety.passed = True
    acceptance.tests.passed = True
    acceptance.training_gate.allowed = True
    acceptance.training_gate.ready = 83
    acceptance.training_gate.blocked = 0
    acceptance.training_gate.lock_reasons = []
    acceptance.remaining_blockers = []
    acceptance = acceptance.model_copy(update={"verdict": "PASS"})

    text = render_pretraining_acceptance(acceptance)
    assert "Implementation ready:       83/83" in text
    assert "Training gate:              UNLOCKED" in text
    assert "REAL TRAINING EXECUTED:     NO" in text
    assert "Remaining blockers:" not in text


def test_probe_audit_never_touches_frozen_manifest(tmp_path) -> None:
    """Probe audits are written to scratch paths; the manifest is read-only."""

    from yolo_agent.research.pretraining_acceptance import (
        _build_probe_audit,
        _build_tampered_audit,
    )

    source = REPO_ROOT / "artifacts" / "paper_83_exactness_audit.yaml"
    manifest = REPO_ROOT / "configs" / "research" / "paper_83_manifest.yaml"
    manifest_before = manifest.read_bytes()

    probe = _build_probe_audit(source, tmp_path / "probe_82.yaml", ready_target=82)
    tampered = _build_tampered_audit(source, tmp_path / "tampered.yaml")
    assert probe.is_file() and tampered.is_file()
    assert probe.parent == tmp_path
    assert manifest.read_bytes() == manifest_before


def test_integrity_counts_reflect_real_audit(acceptance_no_tests) -> None:
    counts = acceptance_no_tests.paper_campaign.integrity_counts
    assert counts.total == 83
    assert counts.ready + counts.blocked + counts.out_of_scope == counts.total
    assert counts.generic_only == 14
    assert counts.missing_runtime == 0  # ready papers all have real hooks
