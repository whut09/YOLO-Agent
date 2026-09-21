"""Prompt-18I §9: real-run integration hook for error profiling.

One entry point connects a *real* evaluation (fixed COCO post-eval output
or any COCO GT/predictions pair) to the error-profile pipeline:

    collect_error_profile(...)

It builds the full :class:`DetectionErrorProfile` from the real GT and
prediction files, overlays the official COCO report numbers when the
post-eval report is available, writes the §7 round artifact, and returns
the profile.  No GPU is required at hook level; the caller decides which
evaluation artifacts to feed.
"""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    ErrorProfileSource,
)
from yolo_agent.core.detection_error_profile_builder import (
    build_detection_error_profile,
)
from yolo_agent.core.error_round_artifacts import (
    write_error_profile_artifact,
)


def collect_error_profile(
    *,
    gt_json: Path | str,
    predictions_json: Path | str,
    run_id: str,
    candidate_id: str,
    split: str = "val",
    official_report_json: Path | str | None = None,
    image_metadata_json: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    dataset_manifest_hash: str = "",
    protocol_hash: str = "",
    role: str = "candidate",
) -> DetectionErrorProfile:
    """Build the round profile from a real evaluation's artifacts.

    ``official_report_json`` accepts the fixed COCO post-eval report; its
    aggregate numbers override the builder's estimates exactly as in
    :func:`build_detection_error_profile`.  ``image_metadata_json`` feeds
    §5 scene slices; without metadata no slices are fabricated.
    """
    source = ErrorProfileSource(
        gt_json=Path(gt_json),
        predictions_json=Path(predictions_json),
        image_metadata_json=Path(image_metadata_json) if image_metadata_json else None,
        official_metrics_json=Path(official_report_json) if official_report_json else None,
        run_id=run_id,
        candidate_id=candidate_id,
        split=split,  # type: ignore[arg-type]
        role=role,  # type: ignore[arg-type]
        protocol_hash=protocol_hash,
        dataset_manifest_hash=dataset_manifest_hash,
    )
    profile = build_detection_error_profile(source)
    if artifacts_dir is not None:
        write_error_profile_artifact(profile, Path(artifacts_dir))
    return profile


__all__ = ["collect_error_profile"]
