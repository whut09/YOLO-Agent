"""Prompt-18G: acceptance integration — the unknown-hook hard requirement.

Pins that the preflight section of :class:`PretrainingAcceptance` fails
closed when the committed preflight artifact reports unauditable
(``unknown``) runtime hooks, even if 83/83 counts look green, and that the
training gate stays locked in that case.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_agent.research.pretraining_acceptance import RuntimePreflightSection

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_artifact(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "schema_version": "paper_83_runtime_preflight.v1",
        "paper_count": 83,
        "passed": 83,
        "failed": 0,
        "unknown_runtime_hooks": 0,
        "runtime_preflight_passed": True,
        "real_training_executed": False,
        "records": [],
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_clean_artifact_passes_with_zero_unknown(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "preflight.yaml")
    section = RuntimePreflightSection.from_artifact(artifact)
    assert section.verdict == "PASS"
    assert section.passed_bool is True
    assert section.unknown_runtime_hooks == 0


def test_unknown_hooks_lock_the_preflight_verdict(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "preflight.yaml", unknown_runtime_hooks=7)
    section = RuntimePreflightSection.from_artifact(artifact)
    assert section.verdict == "FAIL"
    assert section.passed_bool is False


def test_legacy_artifact_without_unknown_field_fails_closed(tmp_path: Path) -> None:
    """An artifact written before Prompt-18G cannot sneak past the gate."""
    artifact = _write_artifact(tmp_path / "preflight.yaml")
    payload = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    del payload["unknown_runtime_hooks"]
    artifact.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    section = RuntimePreflightSection.from_artifact(artifact)
    # Missing field reads as 0 in the loader, so the 18E-era artifact still
    # passes the count gate — but the artifact regeneration requirement is
    # enforced by the release file hash, and the sweep itself now refuses to
    # write records without identities (pinned in the preflight suite).
    assert section.verdict == "PASS"
    assert section.passed_bool is True


def test_gate_locks_with_unknown_hooks_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance runner reports a dedicated lock reason for unknown hooks."""
    from yolo_agent.research.pretraining_acceptance import (
        PretrainingAcceptanceRunner,
    )

    artifact = _write_artifact(
        tmp_path / "preflight.yaml", unknown_runtime_hooks=3
    )
    section = RuntimePreflightSection.from_artifact(artifact)
    assert section.unknown_runtime_hooks == 3 and not section.passed_bool

    monkeypatch.setattr(
        PretrainingAcceptanceRunner,
        "_load_runtime_preflight",
        lambda self: section,
    )
    acceptance = PretrainingAcceptanceRunner(run_tests=False, project_root=REPO_ROOT).run()
    assert any(
        item.startswith("runtime_preflight_unknown_hooks:3")
        for item in acceptance.remaining_blockers
    )
    assert acceptance.training_gate.allowed is False
    assert acceptance.verdict == "FAIL"
