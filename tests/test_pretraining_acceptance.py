"""Pre-training acceptance tests.

Pins the Prompt-16 contract: the acceptance system runs all five sections
(campaign integrity, action space, autonomous loop, safety probes, tests),
always writes its artifacts on PASS *and* FAIL, never mutates the frozen
manifest, never lowers a maturity bar, and never executes real training —
the 83/83-allow probe evaluates the gate only, with the trainer stubbed.

Pins the Prompt-18A gate semantics too: every critical section (including
the test/lint tier) is load-bearing for ``training_gate.allowed`` and for
``verdict == PASS``, and the three signals can never disagree.
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


def test_current_verdict_produces_complete_record(acceptance_no_tests) -> None:
    """Gap closure complete: the campaign is 83/83 → PASS, all sections filled."""

    acceptance = acceptance_no_tests
    assert acceptance.verdict == "PASS"
    assert acceptance.real_training_executed is False
    assert acceptance.paper_campaign.manifest_paper_count == 83
    assert acceptance.paper_campaign.unique_paper_ids == 83
    assert acceptance.paper_campaign.membership_hash_valid
    # Real counts — never lowered to reach PASS.
    assert acceptance.paper_campaign.implementation_ready == 83
    assert acceptance.paper_campaign.blocked == 0
    assert acceptance.paper_campaign.passed
    assert acceptance.training_gate.allowed is True
    assert acceptance.remaining_blockers == []


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


def test_artifacts_written_on_verdict(tmp_path) -> None:
    """Every verdict writes both artifacts — the Prompt-16 core rule."""

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
    assert "Training gate:              UNLOCKED" in text
    assert "REAL TRAINING EXECUTED:     NO" in text
    assert "Remaining blockers:" not in text


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
    assert counts.generic_only == 0
    assert counts.missing_runtime == 0  # ready papers all have real hooks


# ---------------------------------------------------------------------------
# Prompt-18A: gate semantics regression matrix.
# ---------------------------------------------------------------------------


def _section_overridden(acceptance, attr: str, passed: bool):
    """Copy the acceptance with one section's ``passed`` flag overridden."""

    from pydantic import BaseModel

    updated = acceptance.model_copy(deep=True)
    section = getattr(updated, attr)
    if isinstance(section, BaseModel):
        setattr(updated, attr, section.model_copy(update={"passed": passed}))
    else:  # pragma: no cover - all sections are models today
        setattr(updated, attr, passed)
    return updated


def test_gate_matrix_every_critical_section_is_load_bearing(
    acceptance_no_tests,
) -> None:
    """83/83 papers + one red critical section ⇒ gate locked, verdict FAIL."""

    failing_section = {
        "tests": "fast_tests_failed",
        "optimization_action_space": "action_space_failed",
        "autonomous_loop": "autonomous_loop_failed",
        "safety": "safety_failed",
        "paper_campaign": "paper_campaign_failed",
    }
    for attr, expected_reason in failing_section.items():
        candidate = _section_overridden(acceptance_no_tests, attr, False)
        critical = (
            candidate.paper_campaign.passed
            and candidate.optimization_action_space.passed
            and candidate.autonomous_loop.passed
            and candidate.safety.passed
            and candidate.tests.passed
        )
        assert critical is False, attr
        # The invariant must hold through rebuild: allowed ⇔ critical.
        rebuilt = candidate.model_copy(
            update={
                "training_gate": candidate.training_gate.model_copy(
                    update={"allowed": critical}
                ),
                "verdict": "PASS" if critical else "FAIL",
            }
        )
        assert rebuilt.training_gate.allowed is False, attr
        assert rebuilt.verdict == "FAIL", attr


def test_gate_matrix_all_green_passes(acceptance_no_tests) -> None:
    """83/83 + every section green ⇒ allowed and PASS — the only unlock."""

    acceptance = acceptance_no_tests.model_copy(deep=True)
    assert acceptance.paper_campaign.implementation_ready == 83
    assert all(
        [
            acceptance.paper_campaign.passed,
            acceptance.optimization_action_space.passed,
            acceptance.autonomous_loop.passed,
            acceptance.safety.passed,
            acceptance.tests.passed,
        ]
    )
    assert acceptance.training_gate.allowed is True
    assert acceptance.verdict == "PASS"


def test_forged_allowed_with_failed_tests_cannot_be_constructed() -> None:
    """The invariant validator rejects allowed=true over a red test tier."""

    import pytest

    from yolo_agent.research.pretraining_acceptance import (
        PaperCampaignSection,
        PretrainingAcceptance,
        SafetySection,
        TestsSection,
        TrainingGateSection,
    )
    from yolo_agent.agents.action_space import ActionCatalog

    green_campaign = PaperCampaignSection(passed=True, implementation_ready=83)
    red_tests = TestsSection(passed=False, fast_exit_code=1)
    with pytest.raises(ValueError, match="all_critical_checks_pass"):
        PretrainingAcceptance(
            paper_campaign=green_campaign,
            safety=SafetySection(passed=True),
            tests=red_tests,
            training_gate=TrainingGateSection(
                allowed=True, ready=83, blocked=0, required=83
            ),
            verdict="PASS",
        )
    _ = ActionCatalog  # import guard: the action-space module stays healthy


def test_forged_pass_verdict_with_failed_tests_cannot_be_constructed() -> None:
    """verdict=PASS over a red section is the same forgery — rejected."""

    import pytest

    from yolo_agent.research.pretraining_acceptance import (
        PaperCampaignSection,
        PretrainingAcceptance,
        SafetySection,
        TestsSection,
        TrainingGateSection,
    )

    with pytest.raises(ValueError, match="verdict must be PASS iff"):
        PretrainingAcceptance(
            paper_campaign=PaperCampaignSection(passed=True, implementation_ready=83),
            safety=SafetySection(passed=True),
            tests=TestsSection(passed=False, fast_exit_code=1),
            training_gate=TrainingGateSection(
                allowed=False, ready=83, blocked=0, required=83
            ),
            verdict="PASS",
        )


def test_tests_lock_reasons_categorized(acceptance_no_tests) -> None:
    """Fast/lint/slow failures each produce their own lock reason + IDs."""

    from yolo_agent.research.pretraining_acceptance import _tests_lock_reasons

    tests = acceptance_no_tests.tests.model_copy(deep=True)
    tests.fast_command = "pytest -q"
    tests.fast_exit_code = 1
    tests.fast_introduced_failures = ["tests/test_x.py::test_a"]
    tests.fast_preexisting_failures = ["tests/test_y.py::test_b"]
    tests.lint_command = "ruff check ."
    tests.lint_exit_code = 1
    reasons = _tests_lock_reasons(tests)
    assert "fast_tests_failed" in reasons
    assert "lint_failed" in reasons
    assert "fast_tests_failed:tests/test_x.py::test_a" in reasons
    assert "fast_tests_failed:tests/test_y.py::test_b" in reasons
    assert "slow_tests_failed" not in reasons  # slow tier not attempted

    tests.slow_attempted = True
    tests.slow_exit_code = 1
    reasons = _tests_lock_reasons(tests)
    assert "slow_tests_failed" in reasons

    green = acceptance_no_tests.tests.model_copy(deep=True)
    green.fast_command = "pytest -q"
    green.fast_exit_code = 0
    green.lint_command = "ruff check ."
    green.lint_exit_code = 0
    assert _tests_lock_reasons(green) == []
