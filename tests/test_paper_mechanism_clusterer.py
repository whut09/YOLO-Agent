from __future__ import annotations

import pytest

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
    assert set(conflicts[0].candidate_cluster_ids) == {
        "attention_distillation",
        "feature_distillation",
        "localization_distillation",
        "logits_distillation",
        "masked_feature_distillation",
        "quality_aware_distillation",
        "relation_distillation",
        "teacher_ensemble_distillation",
    }
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


def test_explicit_logits_distillation_uses_output_semantics() -> None:
    coverage = _coverage(
        "logits-distillation",
        ["distillation"],
        "Logits distillation adds an auxiliary loss for teacher output distributions.",
    )

    matches, conflicts = PaperMechanismClusterer().cluster(coverage)

    assert conflicts == []
    assert [item.cluster_id for item in matches] == ["logits_distillation"]
    assert matches[0].training_semantic == (
        "teacher_student_output_distribution_alignment"
    )
    assert any(
        "logits" in str(item.value).lower()
        for item in matches[0].evidence
    )


@pytest.mark.parametrize(
    ("component_id", "phrase", "cluster_id"),
    [
        ("distillation.localization", "Localization distillation", "localization_distillation"),
        ("distillation.relation", "Relation distillation", "relation_distillation"),
        ("distillation.attention", "Attention distillation", "attention_distillation"),
        ("distillation.masked_feature", "Masked feature distillation", "masked_feature_distillation"),
        ("distillation.quality_aware", "Quality-aware distillation", "quality_aware_distillation"),
        ("distillation.teacher_ensemble", "Teacher ensemble distillation", "teacher_ensemble_distillation"),
    ],
)
def test_explicit_distillation_semantics_map_to_independent_runtimes(
    component_id: str,
    phrase: str,
    cluster_id: str,
) -> None:
    matches, conflicts = PaperMechanismClusterer().cluster(
        _coverage(component_id, [component_id], f"{phrase} adds a training loss.")
    )

    assert conflicts == []
    assert [item.cluster_id for item in matches] == [cluster_id]
    assert matches[0].match_type == "exact_match"


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


@pytest.mark.parametrize(
    ("component_id", "phrase", "cluster_id"),
    [
        (
            "assigner.task_aligned_weighting",
            "Task aligned positive weighting",
            "task_aligned_weighting",
        ),
        (
            "assigner.dynamic_topk",
            "Dynamic positive cardinality matching",
            "dynamic_topk_matching",
        ),
        (
            "assigner.quality_aware",
            "Classification IoU quality matching",
            "quality_aware_matching",
        ),
        (
            "assigner.soft_label",
            "Soft positive assignment targets",
            "soft_label_assignment",
        ),
        (
            "assigner.dual_path",
            "One to many one to one assignment adaptation",
            "dual_path_assignment",
        ),
        (
            "assigner.conflict_aware",
            "Ambiguous positive conflict rejection",
            "conflict_aware_positive_selection",
        ),
    ],
)
def test_explicit_assignment_mechanisms_map_to_independent_runtime_families(
    component_id: str,
    phrase: str,
    cluster_id: str,
) -> None:
    matches, conflicts = PaperMechanismClusterer().cluster(
        _coverage(
            component_id,
            [component_id],
            f"{phrase} changes target assignment during training.",
        )
    )

    assert conflicts == []
    assert [item.cluster_id for item in matches] == [cluster_id]
    assert matches[0].match_type == "exact_match"


def test_generic_assignment_text_does_not_guess_a_specific_runtime() -> None:
    matches, _ = PaperMechanismClusterer().cluster(
        _coverage(
            "generic-assignment",
            ["object_detection"],
            "Label assignment selects positive samples during training.",
        )
    )

    assert all(
        item.cluster_id
        not in {
            "task_aligned_weighting",
            "dynamic_topk_matching",
            "quality_aware_matching",
            "soft_label_assignment",
            "dual_path_assignment",
            "conflict_aware_positive_selection",
        }
        for item in matches
    )


def test_coupled_paper_keeps_independent_data_and_graph_clusters() -> None:
    coverage = _coverage(
        "coupled-neck-augmentation",
        ["rtmdet_large_kernel_neck"],
        "A large-kernel depthwise neck changes the model graph. Scale-aware augmentation "
        "changes the augmentation policy in training data.",
    )

    matches, conflicts = PaperMechanismClusterer().cluster(coverage)

    assert conflicts == []
    assert [item.cluster_id for item in matches] == [
        "augmentation",
        "large_kernel_neck",
    ]
    assert len({item.training_semantic for item in matches}) == 2


@pytest.mark.parametrize(
    ("component_id", "phrase", "cluster_id"),
    [
        ("neck.weighted_feature_pyramid", "Weighted feature pyramid", "weighted_feature_pyramid"),
        ("neck.bidirectional_feature_fusion", "Bidirectional feature fusion", "bidirectional_feature_fusion"),
        ("neck.gold_gather_distribute", "Gather-distribute fusion", "gather_distribute_fusion"),
        ("neck.lightweight", "Efficient depthwise neck", "lightweight_neck"),
        ("neck.rtmdet_large_kernel", "Large-kernel depthwise neck", "large_kernel_neck"),
        ("block.reparameterized_convolution", "RepConv structural re-parameterization", "reparameterized_convolution"),
        ("attention.channel", "Channel attention", "channel_attention"),
        ("attention.spatial", "Spatial attention", "spatial_attention"),
        ("neck.deformable_feature_aggregation", "Deformable feature aggregation", "deformable_feature_aggregation"),
    ],
)
def test_graph_mechanisms_cluster_to_independent_runtime_families(
    component_id: str,
    phrase: str,
    cluster_id: str,
) -> None:
    matches, conflicts = PaperMechanismClusterer().cluster(
        _coverage(component_id, [component_id], f"{phrase} changes the detector neck.")
    )

    assert conflicts == []
    assert [item.cluster_id for item in matches] == [cluster_id]
    assert matches[0].match_type == "exact_match"


def test_exact_data_mechanisms_keep_independent_runtime_identities() -> None:
    cases = [
        (
            "class-balanced",
            "class_balanced_sampling",
            "Class-balanced sampling changes the train dataloader sampling policy.",
            "class_balanced_sampling",
        ),
        (
            "repeat-factor",
            "repeat_factor_sampling",
            "Repeat-factor sampling changes image exposure in the train dataloader.",
            "repeat_factor_sampling",
        ),
        (
            "hard-negative-replay",
            "hard_negative_replay",
            "Hard-negative replay changes the train dataloader using local false positives.",
            "hard_negative_replay",
        ),
        (
            "scale-crop",
            "scale_aware_crop",
            "Scale-aware crop transforms images and boxes in the training data.",
            "scale_aware_crop",
        ),
    ]
    for paper_id, component, abstract, expected_cluster in cases:
        matches, conflicts = PaperMechanismClusterer().cluster(
            _coverage(paper_id, [component], abstract)
        )
        assert conflicts == []
        assert [item.cluster_id for item in matches] == [expected_cluster]
        assert matches[0].match_type == "exact_match"


def test_hard_negative_replay_is_not_loss_ohem() -> None:
    replay = _coverage(
        "replay",
        ["hard_negative_replay"],
        "Hard-negative replay resamples local false-positive images in the train dataloader.",
    )

    matches, conflicts = PaperMechanismClusterer().cluster(replay)

    assert conflicts == []
    assert [item.cluster_id for item in matches] == ["hard_negative_replay"]
    assert matches[0].training_semantic == "local_false_positive_image_replay"
