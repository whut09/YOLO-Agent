from __future__ import annotations

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.method_profiles import PaperMethodProfileBuilder
from yolo_agent.research.paper_mechanism_clusterer import PaperMechanismClusterer
from yolo_agent.research.schemas import PaperRecord


def _coverage(
    paper_id: str,
    component_ids: list[str],
    abstract: str,
):  # type: ignore[no-untyped-def]
    paper = PaperRecord(
        paper_id=paper_id,
        title="Detector method",
        year=2025,
        abstract=abstract,
        component_ids=component_ids,
    )
    return PaperMethodProfileBuilder(
        ComponentAliasResolver.from_yaml()
    ).build([paper])


def test_unique_canonical_component_exact_matches_reusable_cluster() -> None:
    coverage = _coverage(
        "sampling",
        ["small_object_sampling"],
        "Small object sampling modifies image weights in the train dataloader.",
    )

    matches, conflicts = PaperMechanismClusterer().cluster(coverage)

    assert conflicts == []
    assert len(matches) == 1
    assert matches[0].cluster_id == "sampling_class_balancing"
    assert matches[0].adapter_family == "data.sampling"
    assert matches[0].match_type == "exact_match"
    assert matches[0].confidence == "high"


def test_generic_distillation_does_not_guess_feature_or_logits_semantics() -> None:
    coverage = _coverage(
        "generic-distillation",
        ["distillation"],
        "Teacher student distillation improves the detector.",
    )

    matches, conflicts = PaperMechanismClusterer().cluster(coverage)

    assert matches[0].match_type == "unresolved"
    assert len(conflicts) == 1
    assert conflicts[0].candidate_cluster_ids == [
        "feature_distillation",
        "logits_distillation",
    ]
    assert conflicts[0].reason.startswith("ambiguous_training_semantics")


def test_explicit_feature_distillation_disambiguates_shared_adapter_component() -> None:
    coverage = _coverage(
        "feature-distillation",
        ["distillation"],
        "Feature distillation aligns intermediate teacher and student features.",
    )

    matches, conflicts = PaperMechanismClusterer().cluster(coverage)

    assert conflicts == []
    assert [item.cluster_id for item in matches] == ["feature_distillation"]
    assert matches[0].match_type == "semantic_match"
    assert matches[0].evidence
    assert all(item.source_location for item in matches[0].evidence)


def test_train_time_and_posthoc_calibration_are_not_merged() -> None:
    training = _coverage(
        "train-calibration",
        ["confidence_calibration"],
        "Confidence calibration adds an auxiliary loss during training.",
    )
    posthoc = _coverage(
        "posthoc-calibration",
        ["object_detection"],
        "Post hoc calibration applies temperature scaling in the inference policy.",
    )

    train_matches, _ = PaperMechanismClusterer().cluster(training)
    post_matches, _ = PaperMechanismClusterer().cluster(posthoc)

    assert [item.cluster_id for item in train_matches] == ["confidence_calibration"]
    assert [item.cluster_id for item in post_matches] == [
        "post_processing_calibration"
    ]
    assert train_matches[0].adapter_family != post_matches[0].adapter_family


def test_sampling_and_hard_example_mining_keep_distinct_training_semantics() -> None:
    sampling = _coverage(
        "rare-sampling",
        ["small_object_sampling"],
        "Rare class sampling changes image exposure in the train dataloader.",
    )
    mining = _coverage(
        "hard-mining",
        ["object_detection"],
        "Online hard example mining selects difficult losses during training.",
    )

    sampling_matches, _ = PaperMechanismClusterer().cluster(sampling)
    mining_matches, _ = PaperMechanismClusterer().cluster(mining)

    assert [item.cluster_id for item in sampling_matches] == [
        "sampling_class_balancing"
    ]
    assert [item.cluster_id for item in mining_matches] == ["hard_example_mining"]
    assert sampling_matches[0].training_semantic != mining_matches[0].training_semantic
