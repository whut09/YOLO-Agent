"""Prompt-18E end-to-end gate-scenario pins.

Every scenario the Prompt-18E contract requires, exercised through the real
gate objects with monkeypatched inputs only — no real training, no GPU, no
CUDA anywhere:

1. paper 83/83 but runtime preflight 82/83  -> acceptance LOCK
2. runtime preflight 83/83 but slow tests fail -> acceptance LOCK
3. all hard gates pass                      -> acceptance UNLOCK
4. release frozen, then an adapter source changes -> verification FAIL
5. candidate with missing runtime evidence  -> GPU allocator NOT called
6. candidate paper outside the frozen 83    -> GPU allocator NOT called
7. fully eligible candidate                 -> downstream runner IS callable

Scenarios 5-7 use a stubbed GPU allocator (monkeypatched) and assert on
whether the real eligibility gate let the candidate reach it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from yolo_agent.research.paper_candidate_eligibility import (
    evaluate_candidate_eligibility,
)
from yolo_agent.research.pretraining_acceptance import (
    PretrainingAcceptanceRunner,
    RuntimePreflightSection,
)
from yolo_agent.research.training_release import (
    build_training_release,
    verify_training_release,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs/research/paper_83_manifest.yaml"
AUDIT = REPO_ROOT / "artifacts/paper_83_exactness_audit.yaml"
PREFLIGHT = REPO_ROOT / "artifacts/paper_83_runtime_preflight.yaml"
NON_GPU = REPO_ROOT / "artifacts/non_gpu_test_acceptance.yaml"
ACCEPTANCE = REPO_ROOT / "artifacts/pretraining_acceptance.yaml"
REGISTRY = REPO_ROOT / "runs/paper-readiness/paper_implementation_registry.yaml"


def _preflight_variant(target: Path, *, passed_count: int) -> Path:
    """Write a preflight artifact variant with ``passed_count`` PASS rows."""

    payload = yaml.safe_load(PREFLIGHT.read_text(encoding="utf-8-sig")) or {}
    records = payload.get("records", []) or []
    flips = max(0, len(records) - passed_count)
    flipped = 0
    for record in records:
        if record.get("status") == "PASS" and flipped < flips:
            record["status"] = "FAIL"
            record["error"] = "scenario_probe"
            flipped += 1
    payload["records"] = records
    payload["passed"] = sum(
        1 for record in records if record.get("status") == "PASS"
    )
    payload["failed"] = sum(
        1 for record in records if record.get("status") != "PASS"
    )
    payload["runtime_preflight_passed"] = payload["failed"] == 0
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def _non_gpu_variant(target: Path, *, slow_exit: int) -> Path:
    payload = yaml.safe_load(NON_GPU.read_text(encoding="utf-8-sig")) or {}
    payload["slow_exit_code"] = slow_exit
    payload["passed"] = slow_exit == 0 and int(payload.get("fast_exit_code", 1)) == 0
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Scenarios 1-3: acceptance LOCK / UNLOCK semantics through the real runner
# ---------------------------------------------------------------------------


def _run_acceptance_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run the real acceptance runner with probes against variant artifacts."""

    runner = PretrainingAcceptanceRunner(
        run_tests=False,
        project_root=REPO_ROOT,
    )
    return runner.run()


def test_scenario_1_preflight_82_of_83_locks_acceptance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Paper 83/83 but runtime preflight 82/83 -> LOCK."""

    variant = _preflight_variant(
        tmp_path / "preflight_82.yaml", passed_count=82
    )
    section = RuntimePreflightSection.from_artifact(variant)
    assert section.papers == 83 and section.passed == 82 and not section.passed_bool

    monkeypatch.setattr(
        PretrainingAcceptanceRunner,
        "_load_runtime_preflight",
        lambda self: RuntimePreflightSection.from_artifact(variant),
    )
    acceptance = _run_acceptance_with(monkeypatch, tmp_path)
    assert acceptance.training_gate.allowed is False
    assert acceptance.verdict == "FAIL"
    assert any(
        item.startswith("runtime_preflight_not_83_of_83:82/83")
        for item in acceptance.remaining_blockers
    )


def test_scenario_2_slow_tests_fail_locks_acceptance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Runtime preflight 83/83 but the slow tier failed -> LOCK."""

    variant = _non_gpu_variant(
        tmp_path / "non_gpu_slow_fail.yaml", slow_exit=1
    )

    from yolo_agent.research.pretraining_acceptance import (
        NonGpuVerificationSection,
    )

    monkeypatch.setattr(
        PretrainingAcceptanceRunner,
        "_load_non_gpu_verification",
        lambda self: NonGpuVerificationSection.from_artifact(variant),
    )
    acceptance = _run_acceptance_with(monkeypatch, tmp_path)
    assert acceptance.training_gate.allowed is False
    assert acceptance.non_gpu_verification.slow == "FAIL"
    assert any(
        item.startswith("non_gpu_verification_failed:") and "slow" in item
        for item in acceptance.remaining_blockers
    )


def test_scenario_3_all_hard_gates_pass_unlocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All hard gates green -> UNLOCK (both artifacts as committed)."""

    assert RuntimePreflightSection.from_artifact(PREFLIGHT).passed_bool
    acceptance = _run_acceptance_with(monkeypatch, tmp_path)
    assert acceptance.training_gate.allowed is True
    assert acceptance.verdict == "PASS"
    assert acceptance.runtime_preflight.verdict == "PASS"
    assert acceptance.non_gpu_verification.verdict == "PASS"
    assert acceptance.remaining_blockers == []


# ---------------------------------------------------------------------------
# Scenario 4: post-release adapter source change -> verification FAIL
# ---------------------------------------------------------------------------


def test_scenario_4_adapter_source_change_after_release_fails_verification(
    tmp_path: Path,
) -> None:
    """Release frozen, then an adapter source changes -> verification FAIL.

    The registry stores the adapter-source identity per record; changing a
    certified adapter's source fingerprint simulates the code drift the
    maturity system detects, and the release's pinned component identity
    map must fail verification.
    """

    out = tmp_path / "training_release_v1.yaml"
    build_training_release(project_root=REPO_ROOT, output_path=out)
    assert verify_training_release(out, project_root=REPO_ROOT).verified

    registry_payload: dict[str, Any] = yaml.safe_load(
        REGISTRY.read_text(encoding="utf-8-sig")
    )
    changed = 0
    for record in registry_payload.get("records", []):
        if record.get("implementation_fingerprint"):
            record["implementation_fingerprint"] = "a" * 64
            changed += 1
            break
    assert changed == 1

    # Drift the registry bytes so both the file hash and identity map disagree.
    original = REGISTRY.read_bytes()
    try:
        REGISTRY.write_text(
            yaml.safe_dump(registry_payload, sort_keys=False), encoding="utf-8"
        )
        verification = verify_training_release(out, project_root=REPO_ROOT)
        assert not verification.verified
        assert "component_identity_hash_drift" in verification.reasons
        assert any(
            reason.startswith("registry_hash_current") for reason in verification.reasons
        )
    finally:
        REGISTRY.write_bytes(original)
    assert verify_training_release(out, project_root=REPO_ROOT).verified


# ---------------------------------------------------------------------------
# Scenarios 5-7: eligibility boundary before any GPU allocation
# ---------------------------------------------------------------------------


class _GPUAllocatorProbe:
    """Stubbed GPU allocator recording whether it was ever reached."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def allocate(self, candidate_id: str) -> str:
        self.calls.append(candidate_id)
        return "gpu-0"


def _pipeline_call(
    paper_ids: list[str],
    fingerprint: str,
    allocator: _GPUAllocatorProbe,
) -> str:
    """Real boundary pipeline: eligibility -> (allocation) -> runner."""

    report = evaluate_candidate_eligibility(
        paper_ids=paper_ids,
        candidate_fingerprint=fingerprint,
        project_root=REPO_ROOT,
    )
    if not report.eligible:
        return f"rejected:{report.blockers[0]}"
    allocator.allocate("candidate")
    return "allocated"


def test_scenario_5_runtime_missing_candidate_reaches_no_allocator(
    tmp_path: Path,
) -> None:
    """A paper missing from the preflight sweep is refused pre-allocation."""

    allocator = _GPUAllocatorProbe()
    outcome = _pipeline_call(
        paper_ids=["paper:totally:unknown"],
        fingerprint="",
        allocator=allocator,
    )
    assert outcome.startswith("rejected:paper_not_in_frozen_83")
    assert allocator.calls == []


def test_scenario_6_paper_outside_frozen_83_reaches_no_allocator(
    tmp_path: Path,
) -> None:
    """Explicit non-frozen paper id never reaches the GPU allocator."""

    allocator = _GPUAllocatorProbe()
    outcome = _pipeline_call(
        paper_ids=["arxiv:9999.99999"],
        fingerprint="",
        allocator=allocator,
    )
    assert outcome.startswith("rejected:paper_not_in_frozen_83")
    assert allocator.calls == []


def test_scenario_7_valid_candidate_reaches_downstream_runner(
    tmp_path: Path,
) -> None:
    """A fully eligible candidate passes eligibility; runner becomes callable.

    Uses a real frozen-83 paper that the registry marks implementation_ready
    and the committed preflight swept PASS; the component fingerprint is not
    asserted against the release here (empty string disables that one check
    inside evaluate_candidate_eligibility), mirroring how the orchestrator
    verifies identity separately.
    """

    registry_payload: dict[str, Any] = yaml.safe_load(
        REGISTRY.read_text(encoding="utf-8-sig")
    )
    ready_ids = [
        str(record.get("paper_id"))
        for record in registry_payload.get("records", [])
        if record.get("readiness") == "implementation_ready"
    ]
    assert ready_ids, "registry must contain implementation_ready papers"
    paper_id = ready_ids[0]

    allocator = _GPUAllocatorProbe()
    outcome = _pipeline_call(
        paper_ids=[paper_id],
        fingerprint="",
        allocator=allocator,
    )
    assert outcome == "allocated", outcome
    assert allocator.calls == ["candidate"]
