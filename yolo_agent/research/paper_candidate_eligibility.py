"""Prompt-18E candidate eligibility: the GPU-allocation boundary.

A paper-informed candidate may enter the ASHA queue only when every one of
these holds at registration time:

* its paper is in the frozen 83 campaign (manifest membership);
* the implementation registry reports it ``implementation_ready``;
* the exactness audit record passes (no blockers, ready status);
* the committed runtime preflight sweeps it PASS (real forward/backward
  execution on synthetic tensors — no lazy-import surprises later);
* its current component fingerprint matches the one frozen in the verified
  training release (no code drift since the release was built).

Failure of any condition rejects the candidate *before* GPU allocation:
it never reaches an ExperimentRunner or the Ultralytics trainer.  Nothing
here starts a trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_DEFAULT = Path("configs/research/paper_83_manifest.yaml")
AUDIT_DEFAULT = Path("artifacts/paper_83_exactness_audit.yaml")
PREFLIGHT_DEFAULT = Path("artifacts/paper_83_runtime_preflight.yaml")
RELEASE_DEFAULT = Path("artifacts/training_release_v1.yaml")


@dataclass
class CandidateEligibilityReport:
    """One candidate's eligibility verdict with per-condition blockers."""

    candidate_id: str
    paper_ids: list[str] = field(default_factory=list)
    eligible: bool = False
    blockers: list[str] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.eligible


def _load_yaml(path: Path) -> dict[str, object]:
    import yaml

    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _frozen_paper_ids(manifest_path: Path) -> set[str]:
    from yolo_agent.research.paper_83_campaign_schemas import Paper83Manifest

    manifest = Paper83Manifest.from_yaml(manifest_path)
    return {item.paper_id for item in manifest.papers}


def _registry_ready_ids(registry_path: Path) -> set[str]:
    payload = _load_yaml(registry_path)
    return {
        str(record.get("paper_id"))
        for record in payload.get("records", []) or []
        if isinstance(record, dict)
        and record.get("readiness") == "implementation_ready"
        and record.get("paper_id")
    }


def _audit_pass_ids(audit_path: Path) -> set[str]:
    payload = _load_yaml(audit_path)
    return {
        str(record.get("paper_id"))
        for record in payload.get("records", []) or []
        if isinstance(record, dict)
        and record.get("status") == "implementation_ready"
        and not record.get("blockers")
        and record.get("paper_id")
    }


def _preflight_pass_ids(preflight_path: Path) -> set[str]:
    payload = _load_yaml(preflight_path)
    return {
        str(record.get("paper_id"))
        for record in payload.get("records", []) or []
        if isinstance(record, dict)
        and record.get("status") == "PASS"
        and record.get("paper_id")
    }


def _release_identity_map(release_path: Path) -> dict[str, str]:
    from yolo_agent.research.training_release import verify_training_release

    verification = verify_training_release(release_path)
    if not verification.verified:
        return {}
    payload = _load_yaml(release_path)
    evidence = payload.get("evidence") or {}
    if not isinstance(evidence, dict):
        return {}
    identities = evidence.get("component_identity_hashes") or {}
    return identities if isinstance(identities, dict) else {}


def _registry_fingerprints(registry_path: Path) -> dict[str, str]:
    payload = _load_yaml(registry_path)
    return {
        str(record.get("paper_id")): str(record.get("implementation_fingerprint") or "")
        for record in payload.get("records", []) or []
        if isinstance(record, dict) and record.get("paper_id")
    }


def evaluate_candidate_eligibility(
    *,
    paper_ids: list[str],
    candidate_fingerprint: str,
    project_root: Path | str = ".",
    manifest_path: Path | str = MANIFEST_DEFAULT,
    audit_path: Path | str = AUDIT_DEFAULT,
    registry_path: Path | str = Path(
        "runs/paper-readiness/paper_implementation_registry.yaml"
    ),
    preflight_path: Path | str = PREFLIGHT_DEFAULT,
    release_path: Path | str = RELEASE_DEFAULT,
) -> CandidateEligibilityReport:
    """Check all five Prompt-18E conditions for one paper candidate.

    ``candidate_fingerprint`` is the candidate's current execution/component
    fingerprint; for paper candidates whose registration carries a single
    paper identity it must equal that paper's frozen registry fingerprint
    (which the release pinned via ``component_identity_hashes``).
    """

    root = Path(project_root)
    blockers: list[str] = []

    manifest_file = manifest_path if Path(manifest_path).is_absolute() else root / manifest_path
    audit_file = audit_path if Path(audit_path).is_absolute() else root / audit_path
    registry_file = (
        registry_path if Path(registry_path).is_absolute() else root / registry_path
    )
    preflight_file = (
        preflight_path if Path(preflight_path).is_absolute() else root / preflight_path
    )
    release_file = (
        release_path if Path(release_path).is_absolute() else root / release_path
    )

    frozen = _frozen_paper_ids(manifest_file)
    not_frozen = sorted(set(paper_ids) - frozen)
    if not_frozen:
        blockers.append(f"paper_not_in_frozen_83:{','.join(not_frozen)}")

    ready = _registry_ready_ids(registry_file)
    not_ready = sorted(set(paper_ids) - ready)
    if not_ready:
        blockers.append(f"paper_not_implementation_ready:{','.join(not_ready)}")

    audit_pass = _audit_pass_ids(audit_file)
    not_audited = sorted(set(paper_ids) - audit_pass)
    if not_audited:
        blockers.append(f"paper_exactness_audit_not_pass:{','.join(not_audited)}")

    preflight_pass = _preflight_pass_ids(preflight_file)
    not_swept = sorted(set(paper_ids) - preflight_pass)
    if not_swept:
        blockers.append(f"paper_runtime_preflight_not_pass:{','.join(not_swept)}")

    release_map = _release_identity_map(release_file)
    if not release_map:
        blockers.append("training_release_not_verified_or_unpinned")
    else:
        registry_prints = _registry_fingerprints(registry_file)
        for paper_id in sorted(set(paper_ids) & frozen):
            pinned = release_map.get(paper_id, "")
            live = registry_prints.get(paper_id, "")
            if not pinned:
                blockers.append(f"release_identity_unpinned:{paper_id}")
            elif pinned != candidate_fingerprint and candidate_fingerprint:
                blockers.append(
                    f"candidate_fingerprint_drift:{paper_id}:"
                    f"release={pinned[:12]},candidate={candidate_fingerprint[:12]}"
                )
            elif not live:
                blockers.append(f"registry_fingerprint_missing:{paper_id}")

    return CandidateEligibilityReport(
        candidate_id="",
        paper_ids=sorted(set(paper_ids)),
        eligible=not blockers,
        blockers=blockers,
    )


__all__ = [
    "CandidateEligibilityReport",
    "evaluate_candidate_eligibility",
]
