"""Fresh base run-id allocation for the beginner training entrypoint."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


RunAllocationReason = Literal[
    "requested_id_available",
    "existing_run_has_active_work",
    "explicit_existing_run",
    "existing_run_directory",
]

_ACTIVE_QUEUE_STATUSES = {
    "queued",
    "running",
    "paused",
    "blocked_by_resource",
    "needs_resume",
    "needs_evidence",
}
_ACTIVE_ASHA_TRIAL_STATUSES = {
    "waiting",
    "running",
    "promotion_pending",
    "needs_evidence",
    "full_pending_confirmation",
}
_ACTIVE_ASHA_ASSIGNMENT_STATUSES = {"issued", "running"}


@dataclass(frozen=True)
class RunAllocation:
    """Resolved base run identifier and the reason it was selected."""

    requested_run_id: str
    allocated_run_id: str
    sequence: int
    reason: RunAllocationReason

    @property
    def changed(self) -> bool:
        """Return whether allocation changed the user-requested identifier."""
        return self.requested_run_id != self.allocated_run_id

    def metadata(self) -> dict[str, str | int | bool]:
        """Return stable run metadata for audit and status output."""
        return {
            "requested_run_id": self.requested_run_id,
            "allocated_run_id": self.allocated_run_id,
            "run_sequence": self.sequence,
            "fresh_run_reason": self.reason,
            "fresh_run_allocated": self.changed,
        }


def allocate_base_run_id(
    run_root: Path | str,
    requested_run_id: str,
    *,
    reuse_existing: bool = False,
) -> RunAllocation:
    """Allocate a fresh monotonically numbered base run when appropriate."""
    _validate_run_id(requested_run_id)
    root = Path(run_root)
    requested_dir = root / requested_run_id
    if not requested_dir.exists():
        return RunAllocation(
            requested_run_id=requested_run_id,
            allocated_run_id=requested_run_id,
            sequence=0,
            reason="requested_id_available",
        )
    if reuse_existing:
        return RunAllocation(
            requested_run_id=requested_run_id,
            allocated_run_id=requested_run_id,
            sequence=_sequence_from_run_id(requested_run_id),
            reason="explicit_existing_run",
        )
    if _run_family_has_active_work(root, requested_run_id):
        return RunAllocation(
            requested_run_id=requested_run_id,
            allocated_run_id=requested_run_id,
            sequence=_sequence_from_run_id(requested_run_id),
            reason="existing_run_has_active_work",
        )

    sequence = _next_sequence(root, requested_run_id)
    return RunAllocation(
        requested_run_id=requested_run_id,
        allocated_run_id=f"{requested_run_id}-{sequence}",
        sequence=sequence,
        reason="existing_run_directory",
    )


def _validate_run_id(run_id: str) -> None:
    """Reject identifiers that could escape or nest under the run root."""
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be one non-empty path segment")
    if "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must not contain path separators")


def _next_sequence(root: Path, requested_run_id: str) -> int:
    """Return one more than the largest existing base-run sequence."""
    pattern = re.compile(rf"^{re.escape(requested_run_id)}-(?P<sequence>\d+)$")
    sequences = [0]
    if root.is_dir():
        for path in root.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                sequences.append(int(match.group("sequence")))
    return max(sequences) + 1


def _sequence_from_run_id(run_id: str) -> int:
    """Return an explicit numeric suffix when one is present."""
    match = re.search(r"-(\d+)$", run_id)
    return int(match.group(1)) if match else 0


def _run_family_has_active_work(root: Path, run_id: str) -> bool:
    """Return whether the base run or one of its round children is resumable."""
    child_pattern = re.compile(rf"^{re.escape(run_id)}-r\d+$")
    run_dirs = [root / run_id]
    if root.is_dir():
        run_dirs.extend(path for path in root.iterdir() if path.is_dir() and child_pattern.fullmatch(path.name))
    return any(
        _queue_has_active_work(path / "execution_queue.yaml")
        or _asha_has_active_work(path / "artifacts" / "asha_state.yaml")
        for path in run_dirs
    )


def _queue_has_active_work(queue_path: Path) -> bool:
    """Inspect queue statuses without requiring a fully loadable experiment graph."""
    if not queue_path.is_file():
        return False
    try:
        payload = yaml.safe_load(queue_path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return False
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return any(
        isinstance(item, dict) and str(item.get("status", "")).strip() in _ACTIVE_QUEUE_STATUSES
        for item in items
    )


def _asha_has_active_work(state_path: Path) -> bool:
    """Return whether persisted ASHA work remains even with an empty queue."""
    if not state_path.is_file():
        return False
    try:
        payload = yaml.safe_load(state_path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return False
    if not isinstance(payload, dict):
        return False
    assignments = payload.get("assignments", [])
    if any(
        isinstance(item, dict)
        and str(item.get("status", "")).strip() in _ACTIVE_ASHA_ASSIGNMENT_STATUSES
        for item in assignments
    ):
        return True
    trials = payload.get("trials", [])
    return any(
        isinstance(item, dict)
        and str(item.get("status", "")).strip() in _ACTIVE_ASHA_TRIAL_STATUSES
        and item.get("pending_stage") is not None
        for item in trials
    )
