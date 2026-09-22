"""The classification delta section (Prompt-18J §1).

Confusion-matrix movements and top-pair membership changes must appear as
lower-is-better counts: a new top confusion entering the list is a
regression, one resolving is an improvement.  Profiles without any
confusion data on either side must not produce a fabricated section.
"""

from __future__ import annotations

from yolo_agent.core.detection_error_delta import build_detection_error_delta
from yolo_agent.core.detection_error_profile import (
    ClassificationFacts,
    DetectionErrorProfile,
    GlobalMetrics,
)


def _profile(
    profile_id: str,
    candidate_id: str,
    *,
    classification: ClassificationFacts | None = None,
) -> DetectionErrorProfile:
    return DetectionErrorProfile(
        profile_id=profile_id,
        run_id="run-18j",
        candidate_id=candidate_id,
        gt_artifact="gt.json",
        dataset_manifest_hash="manifest-1",
        global_=GlobalMetrics(
            map50=0.5, map50_95=0.3, precision=0.7, recall=0.6
        ),
        classification=classification or ClassificationFacts(),
    )


def test_new_top_confusion_pair_is_a_regression() -> None:
    parent = _profile(
        "prof-parent",
        "baseline",
        classification=ClassificationFacts(
            confusion_matrix={"person->car": 3},
            top_confusion_pairs=[("person->car", 3)],
        ),
    )
    candidate = _profile(
        "prof-cand",
        "cand",
        classification=ClassificationFacts(
            confusion_matrix={"person->car": 3, "car->person": 7},
            top_confusion_pairs=[("car->person", 7), ("person->car", 3)],
        ),
    )

    delta = build_detection_error_delta(candidate, parent)

    section = delta.section("classification")
    assert section is not None
    matrix = {item.metric: item for item in section.counts}
    new_pair = matrix["confusion.car->person"]
    assert new_pair.candidate == 7
    assert new_pair.parent == 0
    assert new_pair.verdict == "regressed"
    top_new = matrix["top_pair.car->person"]
    assert top_new.delta == 1
    assert top_new.verdict == "regressed"
    unchanged = matrix["confusion.person->car"]
    assert unchanged.verdict == "unchanged"


def test_resolved_confusion_pair_is_an_improvement() -> None:
    parent = _profile(
        "prof-parent",
        "baseline",
        classification=ClassificationFacts(
            confusion_matrix={"person->car": 9},
            top_confusion_pairs=[("person->car", 9)],
        ),
    )
    candidate = _profile(
        "prof-cand",
        "cand",
        classification=ClassificationFacts(
            confusion_matrix={}, top_confusion_pairs=[]
        ),
    )

    delta = build_detection_error_delta(candidate, parent)

    section = delta.section("classification")
    assert section is not None
    counts = {item.metric: item for item in section.counts}
    resolved = counts["confusion.person->car"]
    assert resolved.delta == -9
    assert resolved.verdict == "improved"
    left_top = counts["top_pair.person->car"]
    assert left_top.delta == -1
    assert left_top.verdict == "improved"


def test_no_confusion_data_on_either_side_produces_no_section() -> None:
    parent = _profile("prof-parent", "baseline")
    candidate = _profile("prof-cand", "cand")

    delta = build_detection_error_delta(candidate, parent)

    assert delta.section("classification") is None


def test_classification_regressions_surface_in_delta_regressions() -> None:
    """The objective moved up but a new top confusion appeared: the
    regression must be visible to downstream verdict engines."""
    parent = _profile(
        "prof-parent",
        "baseline",
        classification=ClassificationFacts(
            confusion_matrix={"person->car": 2},
            top_confusion_pairs=[("person->car", 2)],
        ),
    )
    candidate = _profile(
        "prof-cand",
        "cand",
        classification=ClassificationFacts(
            confusion_matrix={"person->car": 2, "dog->cat": 11},
            top_confusion_pairs=[("dog->cat", 11)],
        ),
    )

    delta = build_detection_error_delta(candidate, parent)

    regressions = delta.regressions()
    assert any(name.startswith("classification.") for name in regressions)

    # And the verdict engine treats it as a non-objective regression.
    from yolo_agent.core.error_round_decision import decide_next_round
    from yolo_agent.core.task_spec import MetricPriority, TaskSpec

    spec = TaskSpec(
        class_names=["person"],
        primary_metric=MetricPriority(name="map50_95"),
    )
    decision = decide_next_round(delta, spec)
    # Flat objective + classification regression -> must not stop/promote.
    assert decision.decision == "refine"
