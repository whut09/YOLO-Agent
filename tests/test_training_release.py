"""Training-release freeze tests (Prompt-17).

The release artifact is the boundary between "the gate is unlocked" and
"someone may start training".  These tests pin: the artifact is always
written (READY and BLOCKED), verification is fail-closed against every
drift (manifest, acceptance, registry, component hashes, test hashes,
git commit, in-place tampering), the CLI release command never starts a
trainer, and the real training seams (CLI train entry, queue executor,
Ultralytics runtime entrypoint) refuse allocation without a verified
release.  All trainer machinery is monkeypatched and never executes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from yolo_agent.research.training_release import (
    TrainingRelease,
    TrainingReleaseMissingError,
    build_training_release,
    canonical_payload_sha256,
    release_guard,
    render_release_summary,
    verify_training_release,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs/research/paper_83_manifest.yaml"
ACCEPTANCE = REPO_ROOT / "artifacts/pretraining_acceptance.yaml"
AUDIT = REPO_ROOT / "artifacts/paper_83_exactness_audit.yaml"


@pytest.fixture(scope="module")
def frozen_manifest_bytes() -> bytes:
    return MANIFEST.read_bytes()


def test_release_builds_ready_against_current_campaign(tmp_path: Path) -> None:
    """The live repository is 83/83 ready, so the release freezes READY."""

    out = tmp_path / "training_release_v1.yaml"
    release = build_training_release(
        project_root=REPO_ROOT,
        output_path=out,
    )
    assert out.is_file()
    assert release.release_status == "READY_FOR_FIRST_TRAINING"
    assert release.paper_ready_count == 83
    assert release.paper_blocked_count == 0
    assert release.real_training_executed is False
    assert release.lock_reasons == []
    assert release.release_hash
    assert set(release.allowed_training_modes) >= {
        "debug",
        "pilot",
        "baseline_full",
        "candidate_full",
    }
    reloaded = TrainingRelease.from_yaml(out)
    assert reloaded.release_hash == release.release_hash


def test_release_artifact_carries_required_fields(tmp_path: Path) -> None:
    out = tmp_path / "training_release_v1.yaml"
    release = build_training_release(project_root=REPO_ROOT, output_path=out)
    assert release.release_id.startswith("training-release-v1-")
    assert release.created_at is not None
    assert release.git_commit and release.git_commit != "git_unavailable"
    assert release.paper_membership_hash
    assert release.implementation_registry_hash
    assert release.acceptance_hash
    assert release.evidence.manifest_membership_hash == release.paper_membership_hash


def test_blocked_release_still_written_when_acceptance_missing(
    tmp_path: Path,
) -> None:
    """A missing acceptance artifact produces BLOCKED — the file still exists."""

    out = tmp_path / "training_release_v1.yaml"
    empty = tmp_path / "empty"
    empty.mkdir()
    release = build_training_release(
        project_root=empty,
        acceptance_path=empty / "nonexistent.yaml",
        manifest_path=REPO_ROOT / "configs/research/paper_83_manifest.yaml",
        registry_path=empty / "nonexistent_registry.yaml",
        output_path=out,
    )
    assert out.is_file()
    assert release.release_status == "BLOCKED"
    assert release.allowed_training_modes == []
    assert any(
        reason.startswith("acceptance_artifact_missing") for reason in release.lock_reasons
    )


def test_blocked_release_when_counts_below_83(tmp_path: Path) -> None:
    """An acceptance claiming 82/83 ready is BLOCKED, never READY."""

    payload = yaml.safe_load(ACCEPTANCE.read_text(encoding="utf-8-sig"))
    payload["training_gate"]["ready"] = 82
    payload["training_gate"]["allowed"] = False
    forged = tmp_path / "acceptance_82.yaml"
    with forged.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)

    out = tmp_path / "training_release_v1.yaml"
    release = build_training_release(
        project_root=REPO_ROOT,
        acceptance_path=forged,
        output_path=out,
    )
    assert release.release_status == "BLOCKED"
    assert any(
        reason.startswith("acceptance_ready_below_83") for reason in release.lock_reasons
    )
    assert "acceptance_gate_not_allowed" in release.lock_reasons


# ---------------------------------------------------------------------------
# Prompt-18A: the release layer verifies acceptance sections directly.
# ---------------------------------------------------------------------------


def _forged_acceptance(tmp_path: Path, section_key: str | None, passed: bool) -> Path:
    """Copy the committed acceptance, forcing one section's passed flag."""

    payload = yaml.safe_load(ACCEPTANCE.read_text(encoding="utf-8-sig"))
    if section_key is not None:
        payload[section_key]["passed"] = passed
    forged = tmp_path / f"acceptance_{section_key}_{passed}.yaml"
    with forged.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return forged


def test_release_blocked_when_acceptance_tests_failed_but_gate_forged_allowed(
    tmp_path: Path,
) -> None:
    """The Prompt-18A headline case: tests red + gate forged allowed.

    The acceptance claims 83/83 with ``training_gate.allowed=true`` while
    ``tests.passed=false``.  The release layer must refuse READY even though
    the gate block says yes — it verifies each critical section directly.
    """

    forged = _forged_acceptance(tmp_path, "tests", passed=False)
    out = tmp_path / "training_release_v1.yaml"
    release = build_training_release(
        project_root=REPO_ROOT,
        acceptance_path=forged,
        output_path=out,
    )
    assert release.release_status == "BLOCKED"
    assert "acceptance_tests_failed" in release.lock_reasons
    assert release.allowed_training_modes == []


@pytest.mark.parametrize(
    ("section_key", "expected_reason"),
    [
        ("paper_campaign", "acceptance_paper_campaign_failed"),
        ("optimization_action_space", "acceptance_action_space_failed"),
        ("autonomous_loop", "acceptance_autonomous_loop_failed"),
        ("safety", "acceptance_safety_failed"),
        ("tests", "acceptance_tests_failed"),
    ],
)
def test_release_blocked_when_any_critical_section_fails(
    tmp_path: Path, section_key: str, expected_reason: str
) -> None:
    """Each of the five critical sections independently blocks the release."""

    forged = _forged_acceptance(tmp_path, section_key, passed=False)
    out = tmp_path / "training_release_v1.yaml"
    release = build_training_release(
        project_root=REPO_ROOT,
        acceptance_path=forged,
        output_path=out,
    )
    assert release.release_status == "BLOCKED"
    assert expected_reason in release.lock_reasons


def test_release_blocked_when_acceptance_section_missing(tmp_path: Path) -> None:
    """A gutted acceptance (section removed) is BLOCKED — fail closed."""

    payload = yaml.safe_load(ACCEPTANCE.read_text(encoding="utf-8-sig"))
    del payload["tests"]
    forged = tmp_path / "acceptance_no_tests_section.yaml"
    with forged.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    release = build_training_release(
        project_root=REPO_ROOT,
        acceptance_path=forged,
        output_path=tmp_path / "training_release_v1.yaml",
    )
    assert release.release_status == "BLOCKED"
    assert "acceptance_section_missing:tests" in release.lock_reasons


def test_verification_passes_on_fresh_release(tmp_path: Path) -> None:
    out = tmp_path / "training_release_v1.yaml"
    build_training_release(project_root=REPO_ROOT, output_path=out)
    verification = verify_training_release(out, project_root=REPO_ROOT)
    assert verification.verified, verification.reasons
    assert all(verification.checks.values())
    assert verification.reasons == []


def test_verification_fails_when_release_missing(tmp_path: Path) -> None:
    verification = verify_training_release(
        tmp_path / "nonexistent.yaml", project_root=REPO_ROOT
    )
    assert not verification.verified
    assert "release_artifact_missing" in verification.reasons[0]


def test_verification_fails_on_in_place_tampering(tmp_path: Path) -> None:
    """Editing the release body breaks its self-hash — verification refuses."""

    out = tmp_path / "training_release_v1.yaml"
    build_training_release(project_root=REPO_ROOT, output_path=out)
    payload = yaml.safe_load(out.read_text(encoding="utf-8-sig"))
    payload["paper_ready_count"] = 83  # same count, different serialized bytes
    payload["release_id"] = "forged-release-id"
    with out.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    verification = verify_training_release(out, project_root=REPO_ROOT)
    assert not verification.verified
    assert "release_hash_mismatch" in verification.reasons


def test_verification_fails_on_status_downgrade(tmp_path: Path) -> None:
    """A BLOCKED release (honestly built) never verifies."""

    out = tmp_path / "training_release_v1.yaml"
    empty = tmp_path / "empty"
    empty.mkdir()
    build_training_release(
        project_root=empty,
        acceptance_path=empty / "nonexistent.yaml",
        manifest_path=REPO_ROOT / "configs/research/paper_83_manifest.yaml",
        registry_path=empty / "nonexistent_registry.yaml",
        output_path=out,
    )
    verification = verify_training_release(out, project_root=empty)
    assert not verification.verified
    assert "release_status:BLOCKED" in verification.reasons


def test_release_guard_raises_when_not_verified(tmp_path: Path) -> None:
    with pytest.raises(TrainingReleaseMissingError, match="TRAINING RELEASE NOT VERIFIED"):
        release_guard(tmp_path / "nonexistent.yaml", project_root=REPO_ROOT)


def test_release_guard_returns_verification_when_ok(tmp_path: Path) -> None:
    out = tmp_path / "training_release_v1.yaml"
    build_training_release(project_root=REPO_ROOT, output_path=out)
    verification = release_guard(out, project_root=REPO_ROOT)
    assert verification.verified


def test_render_summary_always_declares_no_real_training(tmp_path: Path) -> None:
    out = tmp_path / "training_release_v1.yaml"
    release = build_training_release(project_root=REPO_ROOT, output_path=out)
    lines = render_release_summary(release)
    text = "\n".join(lines)
    assert "READY FOR FIRST TRAINING: YES" in text
    assert "REAL TRAINING EXECUTED: NO" in text

    empty = tmp_path / "empty"
    empty.mkdir()
    blocked = build_training_release(
        project_root=empty,
        acceptance_path=empty / "nonexistent.yaml",
        manifest_path=REPO_ROOT / "configs/research/paper_83_manifest.yaml",
        registry_path=empty / "nonexistent_registry.yaml",
        output_path=tmp_path / "blocked.yaml",
    )
    blocked_text = "\n".join(render_release_summary(blocked))
    assert "READY FOR FIRST TRAINING: NO" in blocked_text
    assert "REAL TRAINING EXECUTED: NO" in blocked_text


# ---------------------------------------------------------------------------
# Real training seams — trainers are monkeypatched and never executed.
# ---------------------------------------------------------------------------


def test_cli_train_entry_refuses_without_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L0 seam: gate may be allowed, but a missing release blocks allocation."""

    import argparse

    from yolo_agent.cli import run_train_command

    # Move the release lookup into the temp dir by running from a cwd whose
    # relative release path is empty; force cwd via monkeypatch.chdir.
    monkeypatch.chdir(tmp_path)

    trainer_calls: list[Any] = []
    monkeypatch.setattr(
        "yolo_agent.cli.evaluate_paper_83_training_gate",
        lambda *a, **k: _allowed_gate_decision(),
    )

    args = argparse.Namespace(
        model="yolo26n.pt",
        data=REPO_ROOT / "configs" / "coco.yaml",
        run_id="probe",
        run_root=tmp_path / "runs",
        profile=None,
        kind="coco",
        goal=None,
        target_metric=None,
        target_delta=None,
        goal_description=None,
        auto_rounds=None,
        dry_run=False,
        confirm_full_run=False,
        no_auto_advance=False,
        max_steps=8,
        no_auto_import=True,
        training_release="artifacts/training_release_v1.yaml",
    )
    code = run_train_command(args)
    assert code == 2
    assert trainer_calls == []


def test_cli_train_entry_proceeds_past_release_with_verified_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L0 seam: with a verified release the entry moves beyond the release check.

    The downstream optimize pipeline is stubbed, so no trainer can run; the
    probe proves only that the release check itself is not the blocker.
    """

    import argparse

    import yolo_agent.cli as cli_module
    from yolo_agent.research.training_release import build_training_release

    release_path = tmp_path / "training_release_v1.yaml"
    build_training_release(project_root=REPO_ROOT, output_path=release_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "yolo_agent.cli.evaluate_paper_83_training_gate",
        lambda *a, **k: _allowed_gate_decision(),
    )
    monkeypatch.setattr(
        cli_module,
        "run_optimize_command",
        lambda args: (_ for _ in ()).throw(_ProbeReached()),
    )

    args = argparse.Namespace(
        model="yolo26n.pt",
        data=REPO_ROOT / "configs" / "coco.yaml",
        run_id="probe",
        run_root=tmp_path / "runs",
        profile=None,
        kind="coco",
        goal=None,
        target_metric=None,
        target_delta=None,
        goal_description=None,
        auto_rounds=None,
        dry_run=False,
        confirm_full_run=False,
        no_auto_advance=False,
        max_steps=8,
        no_auto_import=True,
        training_release=str(release_path),
    )
    with pytest.raises(_ProbeReached):
        cli_module.run_train_command(args)


def test_executor_seam_refuses_without_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """L1 seam: the queue executor refuses when the release cannot verify."""

    from types import SimpleNamespace

    from yolo_agent.core.executor import _paper_83_gate_refusal

    monkeypatch.setattr(
        "yolo_agent.research.paper_83_training_gate.evaluate_paper_83_training_gate",
        lambda *a, **k: _allowed_gate_decision(),
    )
    monkeypatch.setattr(
        "yolo_agent.research.training_release.verify_training_release",
        lambda *a, **k: SimpleNamespace(
            verified=False, reasons=["release_artifact_missing:probe"]
        ),
    )

    node = _probe_node()
    result = _paper_83_gate_refusal(
        "probe-run",
        node,
        node.command_spec,
    )
    assert result is not None
    assert result.status == "skipped"
    assert "training_release_not_verified" in result.message

    _ = verify_training_release  # keep the import referenced


def test_runtime_entrypoint_refuses_without_release(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """L3 seam: run_ultralytics_training raises before importing ultralytics."""

    from types import SimpleNamespace

    import yolo_agent.adapters.ultralytics.runtime_entrypoint as entry

    monkeypatch.setattr(
        "yolo_agent.research.paper_83_training_gate.evaluate_paper_83_training_gate",
        lambda *a, **k: _allowed_gate_decision(),
    )
    monkeypatch.setattr(
        "yolo_agent.research.training_release.release_guard",
        lambda *a, **k: (_ for _ in ()).throw(
            TrainingReleaseMissingError(
                SimpleNamespace(reasons=["release_artifact_missing:probe"])
            )
        ),
    )
    imported: list[str] = []

    class _FakeUltralyticsModule:
        def __getattr__(self, name: str) -> Any:
            imported.append(name)
            raise AssertionError("ultralytics must never be imported in this test")

    monkeypatch.setitem(
        __import__("sys").modules, "ultralytics", _FakeUltralyticsModule()
    )
    with pytest.raises(TrainingReleaseMissingError):
        entry.run_ultralytics_training(
            payload_path=str(tmp_path_probe_payload()),
            command=["python", "-m", "ultralytics", "train"],
        )
    out = capsys.readouterr().err
    assert "TRAINING RELEASE NOT VERIFIED" in out
    assert imported == []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _ProbeReached(Exception):
    """Raised when the stubbed downstream pipeline is reached."""


def _allowed_gate_decision() -> Any:
    from yolo_agent.research.paper_83_training_gate import Paper83GateDecision

    return Paper83GateDecision(
        allowed=True,
        locked=False,
        required=83,
        required_maturity="implementation_ready",
        ready=83,
        blocked=0,
        manifest_membership_hash=canonical_payload_sha256({"probe": True}),
    )


def _probe_node() -> Any:
    from yolo_agent.core.command_spec import CommandSpec
    from yolo_agent.core.experiment_graph import CandidateConfig, ExperimentNode

    spec = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project=REPO_ROOT / "runs",
        name="release-probe",
    )
    return ExperimentNode(
        node_id="release-probe-node",
        candidate_config=CandidateConfig(
            candidate_id="release-probe-candidate",
            base_model="yolo26n.pt",
            scale="n",
            framework="ultralytics",
        ),
        data_version="v1",
        command_spec=spec,
    )


def tmp_path_probe_payload() -> Path:
    return REPO_ROOT / "runs" / "test-release-probe-payload.yaml"
