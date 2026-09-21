"""Prompt-18F final-readiness command pins.

The ``papers final-readiness`` command renders the final verdict block
from committed artifacts and a live release verification.  These tests
pin both the all-green exit-0 shape and the fail-closed blocker list,
plus the acceptance invariants the Prompt-18F contract requires before
release READY is meaningful.  Nothing here starts a trainer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_agent.cli import run_papers_final_readiness_command

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = REPO_ROOT / "artifacts/pretraining_acceptance.yaml"
RELEASE = REPO_ROOT / "artifacts/training_release_v1.yaml"
PREFLIGHT = REPO_ROOT / "artifacts/paper_83_runtime_preflight.yaml"
NON_GPU = REPO_ROOT / "artifacts/non_gpu_test_acceptance.yaml"


class _Args:
    def __init__(
        self,
        *,
        acceptance: Path = ACCEPTANCE,
        release: Path = RELEASE,
        preflight: Path = PREFLIGHT,
        non_gpu: Path = NON_GPU,
        write: Path | None = None,
    ) -> None:
        self.acceptance = acceptance
        self.release = release
        self.preflight = preflight
        self.non_gpu_acceptance = non_gpu
        self.write = write


def test_final_readiness_all_green_exit_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed artifact set yields the all-green YES verdict."""

    code = run_papers_final_readiness_command(_Args())
    out = capsys.readouterr().out
    assert code == 0
    assert "SAFE TO START FIRST TRAINING: YES" in out
    assert "REAL TRAINING EXECUTED:       NO" in out
    assert "Training release:            VERIFIED" in out
    assert "Paper implementations:       83/83 PASS" in out
    assert "Paper runtime preflight:     83/83 PASS" in out
    assert "Blockers:" not in out


def test_final_readiness_fails_closed_on_missing_acceptance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing acceptance artifact forces NO with an explicit blocker."""

    code = run_papers_final_readiness_command(
        _Args(acceptance=tmp_path / "missing.yaml")
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "SAFE TO START FIRST TRAINING: NO" in out
    assert "acceptance artifact missing" in out
    assert "REAL TRAINING EXECUTED:       NO" in out


def test_final_readiness_fails_closed_on_unverified_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing release artifact forces NO with release reasons."""

    code = run_papers_final_readiness_command(_Args(release=tmp_path / "missing.yaml"))
    out = capsys.readouterr().out
    assert code == 1
    assert "SAFE TO START FIRST TRAINING: NO" in out
    assert "Training release:            FAILED" in out
    assert "release: release_artifact_missing" in out


def test_acceptance_invariants_hold_for_committed_record() -> None:
    """Prompt-18F section three: the three signals must be one signal."""

    from yolo_agent.research.pretraining_acceptance import PretrainingAcceptance

    acceptance = PretrainingAcceptance.from_yaml(ACCEPTANCE)
    assert acceptance.all_critical_checks_pass is True
    assert acceptance.training_gate.allowed is True
    assert acceptance.verdict == "PASS"


def test_final_readiness_write_persists_machine_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prompt-18F: --write persists the machine-readable final record."""

    target = tmp_path / "artifacts" / "final_training_readiness.yaml"
    code = run_papers_final_readiness_command(_Args(write=target))
    out = capsys.readouterr().out
    assert code == 0
    assert f"Final readiness artifact: {target}" in out
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "yolo_agent.final_training_readiness"
    assert payload["safe_to_start_first_training"] is True
    assert payload["real_training_executed"] is False
    assert payload["release_verified"] is True
    assert payload["checks"]["Training gate"] is True


def test_release_ready_contract_for_committed_record() -> None:
    """Prompt-18F section four: READY requires 83/0 and zero lock reasons."""

    payload = yaml.safe_load(RELEASE.read_text(encoding="utf-8-sig"))
    assert payload["release_status"] == "READY_FOR_FIRST_TRAINING"
    assert payload["paper_ready_count"] == 83
    assert payload["paper_blocked_count"] == 0
    assert payload["lock_reasons"] == []
    assert payload["real_training_executed"] is False
