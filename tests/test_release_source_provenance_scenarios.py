"""Prompt-18H: release source-provenance scenarios.

End-to-end pins through the real release builder and verifier:

* fresh release, no change            -> VERIFIED (with the new checks);
* unrelated README edit               -> VERIFIED (provenance is scoped);
* adapter source edit after freeze    -> FAIL adapter_source_hash_drift;
* shared loss helper edit             -> FAIL runtime_dependency_hash_drift;
* graph runtime helper edit           -> FAIL runtime_dependency_hash_drift;
* rebuild the release after the edit  -> VERIFIED again.

Every mutation touches a temporary copy restored in ``finally``; no
trainer is started anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.research.training_release import (
    build_training_release,
    verify_training_release,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

ADAPTER_SOURCE = (
    REPO_ROOT / "yolo_agent/components/adapters/losses/quality_alignment.py"
)
LOSS_HELPER = REPO_ROOT / "yolo_agent/components/auxiliary_losses.py"
GRAPH_HELPER = REPO_ROOT / "yolo_agent/components/adapters/neck/runtime.py"


def _fresh_release(tmp_path: Path) -> Path:
    out = tmp_path / "training_release_v1.yaml"
    release = build_training_release(project_root=REPO_ROOT, output_path=out)
    assert release.release_status == "READY_FOR_FIRST_TRAINING"
    assert release.evidence.adapter_source_hashes, "the freeze must pin adapter sources"
    assert release.evidence.runtime_dependency_hashes, "the freeze must pin helper modules"
    return out


def _verify(out: Path):
    return verify_training_release(out, project_root=REPO_ROOT)


def _append_line(path: Path, line: str) -> None:
    original = path.read_bytes()
    path.write_bytes(original + f"\n# {line}\n".encode("utf-8"))


@pytest.fixture()
def restore_files():
    """Snapshot files mutated by a test and restore them afterwards."""
    snapshots: dict[Path, bytes] = {}

    def _track(path: Path) -> None:
        snapshots.setdefault(path, path.read_bytes())

    yield _track
    for path, data in snapshots.items():
        path.write_bytes(data)


def test_fresh_release_verifies_with_source_checks(tmp_path: Path) -> None:
    out = _fresh_release(tmp_path)
    verification = _verify(out)
    assert verification.verified, verification.reasons
    assert verification.checks["adapter_source_hashes_current"] is True
    assert verification.checks["runtime_dependency_hashes_current"] is True


def test_unrelated_readme_edit_keeps_release_verified(
    tmp_path: Path, restore_files
) -> None:
    out = _fresh_release(tmp_path)
    readme = REPO_ROOT / "README.md"
    restore_files(readme)
    _append_line(readme, "prompt-18h unrelated docs edit")
    verification = _verify(out)
    assert verification.verified, verification.reasons


def test_adapter_source_edit_fails_verification(
    tmp_path: Path, restore_files
) -> None:
    out = _fresh_release(tmp_path)
    restore_files(ADAPTER_SOURCE)
    _append_line(ADAPTER_SOURCE, "prompt-18h post-freeze adapter edit")
    verification = _verify(out)
    assert not verification.verified
    assert any(
        reason.startswith("adapter_source_hash_drift:")
        and "quality_alignment" in reason
        for reason in verification.reasons
    ), verification.reasons
    assert verification.checks["adapter_source_hashes_current"] is False


def test_shared_loss_helper_edit_fails_verification(
    tmp_path: Path, restore_files
) -> None:
    out = _fresh_release(tmp_path)
    restore_files(LOSS_HELPER)
    _append_line(LOSS_HELPER, "prompt-18h post-freeze loss helper edit")
    verification = _verify(out)
    assert not verification.verified
    assert any(
        reason.startswith("runtime_dependency_hash_drift:")
        and "auxiliary_losses" in reason
        for reason in verification.reasons
    ), verification.reasons
    assert verification.checks["runtime_dependency_hashes_current"] is False


def test_graph_runtime_helper_edit_fails_verification(
    tmp_path: Path, restore_files
) -> None:
    out = _fresh_release(tmp_path)
    restore_files(GRAPH_HELPER)
    _append_line(GRAPH_HELPER, "prompt-18h post-freeze graph helper edit")
    verification = _verify(out)
    assert not verification.verified
    assert any(
        reason == "runtime_dependency_hash_drift:"
        "yolo_agent.components.adapters.neck.runtime"
        for reason in verification.reasons
    ), verification.reasons
    # The graph adapters whose MRO spans the edited helper also drift:
    assert any(
        reason.startswith("adapter_source_hash_drift:")
        and "neck" in reason
        for reason in verification.reasons
    ), verification.reasons


def test_rebuilding_the_release_after_an_edit_verifies_again(
    tmp_path: Path, restore_files
) -> None:
    """The honest remediation: rebuild preflight/acceptance/release."""
    out = _fresh_release(tmp_path)
    restore_files(ADAPTER_SOURCE)
    _append_line(ADAPTER_SOURCE, "prompt-18h legitimate refactor")
    assert not _verify(out).verified

    rebuilt = tmp_path / "training_release_v2.yaml"
    build_training_release(project_root=REPO_ROOT, output_path=rebuilt)
    verification = _verify(rebuilt)
    assert verification.verified, verification.reasons
    # The rebuilt freeze pins the new source state:
    assert verification.checks["adapter_source_hashes_current"] is True
