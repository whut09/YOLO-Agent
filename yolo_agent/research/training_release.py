"""Immutable pre-training release artifact and its verification.

Prompt-17 contract: acceptance UNLOCKING the gate is not permission to
train.  The release step freezes the exact evidence state that unlocked
the gate into a hash-bound artifact — manifest membership hash, the
acceptance record's own hash, the git commit, the implementation
registry hash, and component/runtime hashes — and every real training
entrypoint must additionally verify that artifact before allocating
anything.  Building or verifying a release never starts a trainer.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from yaml import safe_load

from yolo_agent.core.yaml_io import YAMLModelMixin

DEFAULT_ACCEPTANCE_PATH = "artifacts/pretraining_acceptance.yaml"
DEFAULT_MANIFEST_PATH = "configs/research/paper_83_manifest.yaml"
DEFAULT_REGISTRY_PATH = "runs/paper-readiness/paper_implementation_registry.yaml"
DEFAULT_RELEASE_PATH = "artifacts/training_release_v1.yaml"
DEFAULT_PREFLIGHT_PATH = "artifacts/paper_83_runtime_preflight.yaml"
DEFAULT_NON_GPU_ACCEPTANCE_PATH = "artifacts/non_gpu_test_acceptance.yaml"

RELEASE_SCHEMA_VERSION = "training_release.v1"
RELEASE_STATUS = Literal["READY_FOR_FIRST_TRAINING", "BLOCKED"]

#: Acceptance sections the release layer verifies *itself* (Prompt-18A part
#: three).  The release never trusts the acceptance's own gate verdict: even
#: if the acceptance builder has a bug and stamps ``allowed=true`` over a
#: red section, the release stays BLOCKED.  Each entry maps the payload key
#: to the lock reason emitted when that section reports ``passed: false``.
_REQUIRED_ACCEPTANCE_SECTIONS: tuple[tuple[str, str], ...] = (
    ("paper_campaign", "acceptance_paper_campaign_failed"),
    ("optimization_action_space", "acceptance_action_space_failed"),
    ("autonomous_loop", "acceptance_autonomous_loop_failed"),
    ("safety", "acceptance_safety_failed"),
    ("tests", "acceptance_tests_failed"),
)
#: Public alias: tests and callers reference the required-section contract.
REQUIRED_ACCEPTANCE_SECTIONS = _REQUIRED_ACCEPTANCE_SECTIONS

#: Prompt-18E hard-gate artifacts the release verifies *itself*, in addition
#: to the acceptance record's embedded sections.  A tampered or stale
#: acceptance cannot smuggle a red runtime sweep or a red test record past
#: the release: each artifact is loaded and re-checked here directly, and
#: each failure contributes its own lock reason.
_HARD_GATE_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    # (path, loader, lock reason)
    (DEFAULT_PREFLIGHT_PATH, "runtime_preflight", "release_runtime_preflight_not_83_of_83"),
    (DEFAULT_NON_GPU_ACCEPTANCE_PATH, "non_gpu", "release_non_gpu_verification_failed"),
)


def file_sha256(path: Path | str) -> str:
    """SHA-256 of a file's bytes; empty string when the file is missing."""

    file = Path(path)
    if not file.is_file():
        return ""
    return hashlib.sha256(file.read_bytes()).hexdigest()


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    """Stable hash of a JSON-serializable payload for cross-run comparison."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def current_git_commit(root: Path | str | None = None) -> str:
    """Current commit hash, or ``git_unavailable`` when Git cannot answer."""

    workdir = Path(root or Path(__file__).resolve().parents[2])
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "git_unavailable"
    return completed.stdout.strip() or "git_unavailable"


class ReleaseEvidence(BaseModel):
    """Every hash the release pins, so verification can recheck each one."""

    model_config = ConfigDict(extra="forbid")

    manifest_membership_hash: str
    manifest_file_sha256: str
    acceptance_file_sha256: str
    acceptance_report_hash: str
    implementation_registry_file_sha256: str
    git_commit: str
    component_runtime_hashes: dict[str, str] = Field(default_factory=dict)
    test_result_hashes: dict[str, str] = Field(default_factory=dict)


class TrainingRelease(BaseModel, YAMLModelMixin):
    """The frozen release record written to ``training_release_v1.yaml``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = RELEASE_SCHEMA_VERSION
    release_id: str
    release_status: RELEASE_STATUS
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    git_commit: str
    paper_membership_hash: str
    paper_ready_count: int
    paper_blocked_count: int
    implementation_registry_hash: str
    acceptance_hash: str
    allowed_training_modes: list[str] = Field(default_factory=list)
    evidence: ReleaseEvidence
    lock_reasons: list[str] = Field(default_factory=list)
    real_training_executed: Literal[False] = False
    release_hash: str = ""

    def compute_release_hash(self) -> str:
        """Hash the release body (excluding the hash field itself)."""

        payload = self.model_dump(mode="json", exclude={"release_hash"})
        return canonical_payload_sha256(payload)


class ReleaseVerification(BaseModel, YAMLModelMixin):
    """One verification of a release artifact against the live repository."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "training_release_verification.v1"
    verified: bool
    release_id: str
    release_status: str
    release_path: str
    checks: dict[str, bool] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TrainingReleaseMissingError(RuntimeError):
    """Raised when a training entrypoint cannot verify its release artifact."""

    def __init__(self, verification: ReleaseVerification) -> None:
        reasons = ", ".join(verification.reasons) or "unspecified"
        super().__init__(
            "TRAINING RELEASE NOT VERIFIED: real training requires an explicit, "
            f"verifiable release artifact.  Reasons: {reasons}.  "
            "Run `yolo-agent papers release` to (re)build it, then pass "
            "--training-release artifacts/training_release_v1.yaml."
        )
        self.verification = verification


def _load_acceptance(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        payload = safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"acceptance YAML must contain a mapping: {path}")
    return payload


def _commit_is_ancestor_or_equal(candidate: str, head: str, root: Path) -> bool:
    """Whether ``candidate`` == ``head`` or is reachable from it.

    Unknown/unavailable commits (e.g. ``git_unavailable``) never pass:
    the check fails closed when Git cannot answer.
    """

    if not candidate or not head or candidate == "git_unavailable" or head == "git_unavailable":
        return False
    if candidate == head:
        return True
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", candidate, head],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _collect_test_result_hashes(project_root: Path) -> dict[str, str]:
    """Hash committed audit/acceptance artifacts that record test evidence."""

    names = (
        "artifacts/paper_83_exactness_audit.yaml",
        "artifacts/pretraining_acceptance.yaml",
    )
    return {name: file_sha256(project_root / name) for name in names}


def _collect_component_runtime_hashes(
    project_root: Path,
) -> dict[str, str]:
    """Hash the route contracts whose runtime identity the release freezes."""

    route_file = project_root / "configs/components/distillation/paper_routes.yaml"
    domain_file = project_root / "configs/components/domain_adaptation/paper_routes.yaml"
    return {
        "configs/components/distillation/paper_routes.yaml": file_sha256(route_file),
        "configs/components/domain_adaptation/paper_routes.yaml": file_sha256(
            domain_file
        ),
    }


def build_training_release(
    *,
    project_root: Path | str = ".",
    acceptance_path: Path | str = DEFAULT_ACCEPTANCE_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    registry_path: Path | str = DEFAULT_REGISTRY_PATH,
    output_path: Path | str | None = None,
) -> TrainingRelease:
    """Freeze the current evidence state into the release artifact.

    READY requires the acceptance record itself to declare
    ``training_gate.allowed`` with ready == 83 and blocked == 0 *and* the
    gate re-evaluated now to agree — and (Prompt-18A) every critical
    acceptance section to report ``passed`` directly.  Anything else is
    BLOCKED.  Both verdicts are written; neither starts a trainer.
    """

    if output_path is None:
        output_path = DEFAULT_RELEASE_PATH
    root = Path(project_root).resolve()
    acceptance_file = root / acceptance_path
    manifest_file = root / manifest_path
    registry_file = root / registry_path

    lock_reasons: list[str] = []
    ready = 0
    blocked = 0
    membership_hash = ""
    acceptance_report_hash = ""

    if not acceptance_file.is_file():
        lock_reasons.append(f"acceptance_artifact_missing:{acceptance_path}")
    if not manifest_file.is_file():
        lock_reasons.append(f"manifest_missing:{manifest_path}")
    if not registry_file.is_file():
        lock_reasons.append(f"implementation_registry_missing:{registry_path}")

    acceptance_payload: dict[str, Any] = {}
    if acceptance_file.is_file():
        acceptance_payload = _load_acceptance(acceptance_file)
        gate = acceptance_payload.get("training_gate") or {}
        ready = int(gate.get("ready") or 0)
        blocked = int(gate.get("blocked") or 0)
        allowed = bool(gate.get("allowed"))
        if not allowed:
            lock_reasons.append("acceptance_gate_not_allowed")
        if ready < 83:
            lock_reasons.append(f"acceptance_ready_below_83:{ready}")
        if blocked > 0:
            lock_reasons.append(f"acceptance_blocked_present:{blocked}")
        verdict = str(acceptance_payload.get("verdict", ""))
        if verdict != "PASS":
            lock_reasons.append(f"acceptance_verdict:{verdict or 'missing'}")
        # Prompt-18A: defensive per-section verification — the release layer
        # re-checks every critical acceptance section directly and fails
        # closed on any red one, regardless of what the gate block claims.
        for section_key, section_reason in _REQUIRED_ACCEPTANCE_SECTIONS:
            section_payload = acceptance_payload.get(section_key)
            if not isinstance(section_payload, dict):
                lock_reasons.append(f"acceptance_section_missing:{section_key}")
            elif not bool(section_payload.get("passed")):
                lock_reasons.append(section_reason)
        # Cross-check the acceptance's own gate semantics: allowed=true must
        # imply verdict=PASS here, or the record is internally inconsistent.
        if allowed and verdict != "PASS":
            lock_reasons.append("acceptance_gate_verdict_inconsistent")
        acceptance_report_hash = canonical_payload_sha256(acceptance_payload)
    else:
        # Prompt-18E: even without an acceptance record, the hard-gate
        # artifacts still get their independent direct verification.
        from yolo_agent.research.pretraining_acceptance import (
            NonGpuVerificationSection,
            RuntimePreflightSection,
        )

        for artifact_path, kind, reason in _HARD_GATE_ARTIFACTS:
            artifact_file = root / artifact_path
            if not artifact_file.is_file():
                lock_reasons.append(f"{reason}:artifact_missing:{artifact_path}")
                continue
            if kind == "runtime_preflight":
                section = RuntimePreflightSection.from_artifact(artifact_file)
                if not section.passed_bool:
                    lock_reasons.append(
                        f"{reason}:{section.passed}/{section.papers}"
                    )
            else:
                section = NonGpuVerificationSection.from_artifact(artifact_file)
                if not section.passed_bool:
                    lock_reasons.append(
                        f"{reason}:fast={section.fast},slow={section.slow},ruff={section.ruff}"
                    )

    # Prompt-18E: the two hard-gate artifacts are verified directly by the
    # release layer, independent of what the acceptance record claims.
    from yolo_agent.research.pretraining_acceptance import (
        NonGpuVerificationSection,
        RuntimePreflightSection,
    )

    for artifact_path, kind, reason in _HARD_GATE_ARTIFACTS:
        artifact_file = root / artifact_path
        if not artifact_file.is_file():
            lock_reasons.append(f"{reason}:artifact_missing:{artifact_path}")
            continue
        if kind == "runtime_preflight":
            section = RuntimePreflightSection.from_artifact(artifact_file)
            if not section.passed_bool:
                lock_reasons.append(
                    f"{reason}:{section.passed}/{section.papers}"
                )
        else:
            section = NonGpuVerificationSection.from_artifact(artifact_file)
            if not section.passed_bool:
                lock_reasons.append(
                    f"{reason}:fast={section.fast},slow={section.slow},ruff={section.ruff}"
                )

    if manifest_file.is_file():
        from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest

        manifest = Paper83Manifest.from_yaml(manifest_file)
        membership_hash = manifest.campaign.membership_hash
        computed = manifest.campaign.membership_hash
        recomputed_ids = [item.paper_id for item in manifest.papers]
        from yolo_agent.research.paper_83_campaign_schemas import (
            calculate_membership_hash,
        )

        if calculate_membership_hash(recomputed_ids) != computed:
            lock_reasons.append("manifest_membership_hash_mismatch")
        if len(recomputed_ids) != 83 or len(set(recomputed_ids)) != 83:
            lock_reasons.append("manifest_membership_not_83_unique")

    # Live gate must agree with the acceptance record at release time.
    if not lock_reasons:
        from yolo_agent.research.paper_83_training_gate import (
            evaluate_paper_83_training_gate,
        )

        decision = evaluate_paper_83_training_gate(
            manifest_path=manifest_file,
            exactness_audit_path=root / "artifacts/paper_83_exactness_audit.yaml",
            implementation_registry_path=registry_file,
        )
        if not decision.allowed or decision.ready < 83 or decision.blocked > 0:
            lock_reasons.append(
                "live_gate_disagrees_with_acceptance:"
                f"ready={decision.ready},blocked={decision.blocked}"
            )

    evidence = ReleaseEvidence(
        manifest_membership_hash=membership_hash,
        manifest_file_sha256=file_sha256(manifest_file),
        acceptance_file_sha256=file_sha256(acceptance_file),
        acceptance_report_hash=acceptance_report_hash,
        implementation_registry_file_sha256=file_sha256(registry_file),
        git_commit=current_git_commit(root),
        component_runtime_hashes=_collect_component_runtime_hashes(root),
        test_result_hashes=_collect_test_result_hashes(root),
    )

    registry_hash = file_sha256(registry_file)
    status: RELEASE_STATUS = "BLOCKED" if lock_reasons else "READY_FOR_FIRST_TRAINING"
    release = TrainingRelease(
        release_id=(
            f"training-release-v1-{evidence.git_commit[:12]}-"
            f"{membership_hash[:12] if membership_hash else 'nomanifest'}"
        ),
        release_status=status,
        git_commit=evidence.git_commit,
        paper_membership_hash=membership_hash,
        paper_ready_count=ready,
        paper_blocked_count=blocked,
        implementation_registry_hash=registry_hash,
        acceptance_hash=acceptance_report_hash or file_sha256(acceptance_file),
        allowed_training_modes=(
            ["debug", "pilot", "pilot_3", "pilot_10", "baseline_full",
             "baseline_confirm", "candidate_full", "full"]
            if status == "READY_FOR_FIRST_TRAINING"
            else []
        ),
        evidence=evidence,
        lock_reasons=lock_reasons,
    )
    release.release_hash = release.compute_release_hash()
    release.to_yaml(root / output_path)
    return release


def verify_training_release(
    release_path: Path | str | None = None,
    *,
    project_root: Path | str = ".",
    expected_commit: str | None = None,
) -> ReleaseVerification:
    """Recheck every pinned hash against the live repository.  Fail-closed.

    ``release_path`` defaults to :data:`DEFAULT_RELEASE_PATH` resolved at
    *call* time (None sentinel) so the default stays overridable in tests.
    """

    if release_path is None:
        release_path = DEFAULT_RELEASE_PATH
    root = Path(project_root).resolve()
    release_file = Path(release_path)
    if not release_file.is_absolute():
        release_file = root / release_path
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    if not release_file.is_file():
        return ReleaseVerification(
            verified=False,
            release_id="",
            release_status="missing",
            release_path=str(release_file),
            checks={"release_artifact_present": False},
            reasons=[f"release_artifact_missing:{release_file}"],
        )

    try:
        release = TrainingRelease.from_yaml(release_file)
    except (OSError, ValueError) as exc:
        return ReleaseVerification(
            verified=False,
            release_id="",
            release_status="invalid",
            release_path=str(release_file),
            checks={"release_artifact_parseable": False},
            reasons=[f"release_artifact_invalid:{type(exc).__name__}: {exc}"],
        )

    # 1) Self-hash integrity: the artifact has not been edited in place.
    checks["release_hash_intact"] = (
        release.compute_release_hash() == release.release_hash
    )
    if not checks["release_hash_intact"]:
        reasons.append("release_hash_mismatch")

    # 2) Status must be READY — a BLOCKED release never unlocks training.
    checks["release_status_ready"] = release.release_status == "READY_FOR_FIRST_TRAINING"
    if not checks["release_status_ready"]:
        reasons.append(f"release_status:{release.release_status}")
        reasons.extend(release.lock_reasons)

    # 3) The recorded counts must be the full campaign.
    checks["paper_ready_83"] = release.paper_ready_count == 83
    checks["paper_blocked_0"] = release.paper_blocked_count == 0
    if release.paper_ready_count != 83:
        reasons.append(f"release_paper_ready:{release.paper_ready_count}!=83")
    if release.paper_blocked_count != 0:
        reasons.append(f"release_paper_blocked:{release.paper_blocked_count}")

    # 4) Every pinned file hash must still match the live file.
    #    Defaults resolve at call time (module globals), so a caller may
    #    repoint the acceptance/manifest/registry paths explicitly.
    def _resolve(default: str) -> Path:
        candidate = Path(default)
        return candidate if candidate.is_absolute() else root / default

    manifest_file = _resolve(DEFAULT_MANIFEST_PATH)
    acceptance_file = _resolve(DEFAULT_ACCEPTANCE_PATH)
    registry_file = _resolve(DEFAULT_REGISTRY_PATH)
    pairs = {
        "manifest_hash_current": (release.evidence.manifest_file_sha256, file_sha256(manifest_file)),
        "acceptance_hash_current": (release.evidence.acceptance_file_sha256, file_sha256(acceptance_file)),
        "registry_hash_current": (release.evidence.implementation_registry_file_sha256, file_sha256(registry_file)),
    }
    for name, (pinned, live) in pairs.items():
        checks[name] = bool(live) and pinned == live
        if not checks[name]:
            reasons.append(f"{name}: pinned={pinned[:12]} live={live[:12]}")

    # 5) Membership hash must match the live manifest.
    live_membership = ""
    if manifest_file.is_file():
        from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest

        try:
            manifest = Paper83Manifest.from_yaml(manifest_file)
            live_membership = manifest.campaign.membership_hash
        except (OSError, ValueError):
            live_membership = ""
    checks["membership_hash_current"] = bool(live_membership) and (
        live_membership == release.evidence.manifest_membership_hash
        == release.paper_membership_hash
    )
    if not checks["membership_hash_current"]:
        reasons.append("membership_hash_drift")

    # 6) Component/runtime hashes.
    live_components = _collect_component_runtime_hashes(root)
    checks["component_runtime_hashes_current"] = live_components == (
        release.evidence.component_runtime_hashes
    )
    if not checks["component_runtime_hashes_current"]:
        reasons.append("component_runtime_hash_drift")

    # 7) Test-result hashes.
    live_tests = _collect_test_result_hashes(root)
    checks["test_result_hashes_current"] = live_tests == release.evidence.test_result_hashes
    if not checks["test_result_hashes_current"]:
        reasons.append("test_result_hash_drift")

    # 8) Git commit: the pinned commit must be the current HEAD *or an
    # ancestor of it* (Prompt-18A).  A release committed at commit X can
    # never verify against HEAD == X itself — the commit that carries the
    # artifact is necessarily a child — so ancestry is the honest
    # interpretation of "frozen no later than now".  Any commit that is not
    # an ancestor (diverged/rewritten history, or the caller pinned an
    # explicit expected commit) still fails closed.
    live_commit = current_git_commit(root)
    if expected_commit is not None:
        checks["git_commit_pinned"] = release.git_commit == expected_commit
        if not checks["git_commit_pinned"]:
            reasons.append(
                f"git_commit_drift:release={release.git_commit[:12]},expected={expected_commit[:12]}"
            )
    else:
        checks["git_commit_pinned"] = _commit_is_ancestor_or_equal(
            release.git_commit, live_commit, root
        )
        if not checks["git_commit_pinned"]:
            reasons.append(
                f"git_commit_drift:release={release.git_commit[:12]},head={live_commit[:12]}"
            )

    verified = all(checks.values())
    return ReleaseVerification(
        verified=verified,
        release_id=release.release_id,
        release_status=release.release_status,
        release_path=str(release_file),
        checks=checks,
        reasons=reasons,
    )


def release_guard(
    release_path: Path | str | None = None,
    *,
    project_root: Path | str = ".",
    expected_commit: str | None = None,
) -> ReleaseVerification:
    """Verify the release and raise :class:`TrainingReleaseMissingError`."""

    if release_path is None:
        release_path = DEFAULT_RELEASE_PATH
    verification = verify_training_release(
        release_path,
        project_root=project_root,
        expected_commit=expected_commit,
    )
    if not verification.verified:
        raise TrainingReleaseMissingError(verification)
    return verification


def render_release_summary(
    release: TrainingRelease,
    verification: ReleaseVerification | None = None,
) -> list[str]:
    """Render the required Prompt-17 CLI block."""

    lines = [
        "TRAINING RELEASE",
        f"Release ID: {release.release_id}",
        f"Status: {release.release_status}",
        f"Git commit: {release.git_commit[:12]}",
        f"Paper ready: {release.paper_ready_count}/83",
        f"Paper blocked: {release.paper_blocked_count}",
        "READY FOR FIRST TRAINING: "
        + ("YES" if release.release_status == "READY_FOR_FIRST_TRAINING" else "NO"),
        "REAL TRAINING EXECUTED: NO",
    ]
    if release.lock_reasons:
        lines.append("Locked: " + ", ".join(release.lock_reasons))
    if verification is not None:
        lines.append(
            "Release verification: "
            + ("VERIFIED" if verification.verified else "FAILED")
        )
        for reason in verification.reasons:
            lines.append(f"  reason: {reason}")
    return lines


__all__ = [
    "DEFAULT_RELEASE_PATH",
    "ReleaseEvidence",
    "ReleaseVerification",
    "REQUIRED_ACCEPTANCE_SECTIONS",
    "TrainingRelease",
    "TrainingReleaseMissingError",
    "build_training_release",
    "canonical_payload_sha256",
    "current_git_commit",
    "file_sha256",
    "release_guard",
    "render_release_summary",
    "verify_training_release",
]
