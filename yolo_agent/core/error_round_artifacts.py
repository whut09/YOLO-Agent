"""Prompt-18I §7: per-round error artifacts on disk.

Every round writes three machine-readable records next to the run:

    error_profile.yaml      the full DetectionErrorProfile
    error_delta.yaml        the candidate-vs-parent DetectionErrorDelta
    decision_trace.yaml     the §7 decision trace

Serialization is deterministic (sorted keys, stable aliases) so the
artifacts can be hashed, diffed, and pinned like every other run record.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.core.detection_error_delta import DetectionErrorDelta
from yolo_agent.core.detection_error_profile import DetectionErrorProfile
from yolo_agent.core.error_decision_trace import ErrorDecisionTrace

PROFILE_ARTIFACT = "error_profile.yaml"
DELTA_ARTIFACT = "error_delta.yaml"
TRACE_ARTIFACT = "decision_trace.yaml"


def _dump_yaml(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )
    return path


def write_error_profile_artifact(
    profile: DetectionErrorProfile, artifacts_dir: Path | str
) -> Path:
    return _dump_yaml(
        profile.model_dump(mode="json"), Path(artifacts_dir) / PROFILE_ARTIFACT
    )


def write_error_delta_artifact(
    delta: DetectionErrorDelta, artifacts_dir: Path | str
) -> Path:
    return _dump_yaml(
        delta.model_dump(mode="json"), Path(artifacts_dir) / DELTA_ARTIFACT
    )


def write_decision_trace_artifact(
    trace: ErrorDecisionTrace, artifacts_dir: Path | str
) -> Path:
    return _dump_yaml(
        trace.model_dump(mode="json"), Path(artifacts_dir) / TRACE_ARTIFACT
    )


def write_round_artifacts(
    artifacts_dir: Path | str,
    profile: DetectionErrorProfile,
    delta: DetectionErrorDelta | None = None,
    trace: ErrorDecisionTrace | None = None,
) -> dict[str, Path]:
    """Write all available round artifacts; return name -> path."""
    written = {"error_profile": write_error_profile_artifact(profile, artifacts_dir)}
    if delta is not None:
        written["error_delta"] = write_error_delta_artifact(delta, artifacts_dir)
    if trace is not None:
        written["decision_trace"] = write_decision_trace_artifact(trace, artifacts_dir)
    return written


def load_error_profile_artifact(path: Path | str) -> DetectionErrorProfile:
    return DetectionErrorProfile.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_error_delta_artifact(path: Path | str) -> DetectionErrorDelta:
    return DetectionErrorDelta.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_decision_trace_artifact(path: Path | str) -> ErrorDecisionTrace:
    return ErrorDecisionTrace.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "DELTA_ARTIFACT",
    "PROFILE_ARTIFACT",
    "TRACE_ARTIFACT",
    "load_decision_trace_artifact",
    "load_error_delta_artifact",
    "load_error_profile_artifact",
    "write_decision_trace_artifact",
    "write_error_delta_artifact",
    "write_error_profile_artifact",
    "write_round_artifacts",
]
