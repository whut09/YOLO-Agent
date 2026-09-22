"""Resource-delta section of ``DetectionErrorDelta`` (Prompt-18J §1).

The resources section must come from real runtime measurements carried on
the profiles themselves; a missing snapshot on either side drops the
section instead of fabricating movements.
"""

from __future__ import annotations

from yolo_agent.core.detection_error_delta import (
    ResourceSnapshot,
    build_detection_error_delta,
)
from yolo_agent.core.detection_error_profile import (
    DetectionErrorProfile,
    GlobalMetrics,
)


def _profile(
    profile_id: str,
    candidate_id: str,
    *,
    map50: float,
    resources: ResourceSnapshot | None,
) -> DetectionErrorProfile:
    return DetectionErrorProfile(
        profile_id=profile_id,
        run_id="run-18j",
        candidate_id=candidate_id,
        global_=GlobalMetrics(
            map50=map50,
            map50_95=map50 * 0.6,
            precision=0.7,
            recall=0.6,
        ),
        resources=resources,
    )


def test_resource_section_records_latency_regression_from_real_snapshots() -> None:
    parent = _profile(
        "prof-parent",
        "baseline",
        map50=0.50,
        resources=ResourceSnapshot(
            latency_ms=10.0,
            params=3_000_000,
            flops_g=8.1,
            peak_memory_mb=512.0,
        ),
    )
    candidate = _profile(
        "prof-cand",
        "cand",
        map50=0.52,
        resources=ResourceSnapshot(
            latency_ms=12.4,
            params=3_150_000,
            flops_g=8.1,
            peak_memory_mb=540.0,
        ),
    )

    delta = build_detection_error_delta(candidate, parent)

    section = delta.resources
    assert section is not None
    latency = next(item for item in section.metrics if item.metric == "latency_ms")
    assert latency.candidate == 12.4
    assert latency.parent == 10.0
    assert latency.delta == 2.4
    assert latency.verdict == "regressed"
    assert latency.higher_is_better is False

    params = next(item for item in section.counts if item.metric == "params")
    assert params.candidate == 3_150_000
    assert params.delta == 150_000
    assert params.verdict == "regressed"


def test_resource_section_marks_improvements_and_unchanged() -> None:
    parent = _profile(
        "prof-parent",
        "baseline",
        map50=0.50,
        resources=ResourceSnapshot(
            latency_ms=12.0,
            params=3_000_000,
            flops_g=8.1,
            peak_memory_mb=600.0,
        ),
    )
    candidate = _profile(
        "prof-cand",
        "cand",
        map50=0.51,
        resources=ResourceSnapshot(
            latency_ms=10.0,
            params=3_000_000,
            flops_g=8.1,
            peak_memory_mb=560.0,
        ),
    )

    delta = build_detection_error_delta(candidate, parent)

    section = delta.resources
    assert section is not None
    latency = next(item for item in section.metrics if item.metric == "latency_ms")
    assert latency.verdict == "improved"
    params = next(item for item in section.counts if item.metric == "params")
    assert params.verdict == "unchanged"
    memory = next(item for item in section.metrics if item.metric == "peak_memory_mb")
    assert memory.verdict == "improved"


def test_missing_snapshot_on_one_side_drops_the_section() -> None:
    parent = _profile(
        "prof-parent",
        "baseline",
        map50=0.50,
        resources=ResourceSnapshot(latency_ms=10.0),
    )
    candidate = _profile(
        "prof-cand",
        "cand",
        map50=0.52,
        resources=None,
    )

    delta = build_detection_error_delta(candidate, parent)

    assert delta.resources is None
    assert delta.candidate_resources is None
    assert delta.parent_resources is not None


def test_partial_snapshot_only_reports_available_metrics() -> None:
    """A snapshot with only latency must not invent params/FLOPs movements."""
    parent = _profile(
        "prof-parent",
        "baseline",
        map50=0.50,
        resources=ResourceSnapshot(latency_ms=10.0),
    )
    candidate = _profile(
        "prof-cand",
        "cand",
        map50=0.52,
        resources=ResourceSnapshot(latency_ms=11.0),
    )

    delta = build_detection_error_delta(candidate, parent)

    section = delta.resources
    assert section is not None
    assert [item.metric for item in section.metrics] == ["latency_ms"]
    assert section.counts == []


def test_resources_flow_through_serialization_roundtrip(tmp_path) -> None:
    from yolo_agent.core.error_round_artifacts import (
        load_error_delta_artifact,
        load_error_profile_artifact,
        write_error_delta_artifact,
        write_error_profile_artifact,
    )

    parent = _profile(
        "prof-parent",
        "baseline",
        map50=0.50,
        resources=ResourceSnapshot(latency_ms=10.0, params=1_000),
    )
    candidate = _profile(
        "prof-cand",
        "cand",
        map50=0.53,
        resources=ResourceSnapshot(latency_ms=13.0, params=1_200),
    )
    delta = build_detection_error_delta(candidate, parent)

    profile_path = write_error_profile_artifact(candidate, tmp_path / "profile.yaml")
    delta_path = write_error_delta_artifact(delta, tmp_path / "delta.yaml")

    loaded_profile = load_error_profile_artifact(profile_path)
    loaded_delta = load_error_delta_artifact(delta_path)

    assert loaded_profile.resources is not None
    assert loaded_profile.resources.latency_ms == 13.0
    assert loaded_delta.resources is not None
    latency = next(item for item in loaded_delta.resources.metrics if item.metric == "latency_ms")
    assert latency.verdict == "regressed"
