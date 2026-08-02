from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.research.paper_method_evidence import (
    PaperMethodEvidenceObservation,
    PaperMethodEvidenceProfile,
)
from yolo_agent.research.paper_method_evidence_extractor import (
    PaperMethodEvidenceExtractor,
)
from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.schemas import PaperRecord


def test_structured_method_evidence_is_hash_stable() -> None:
    observation = PaperMethodEvidenceObservation(
        field_name="insertion_point",
        value="trainer_loss",
        source="summary",
        source_location="summary:sentence:1",
        confidence="high",
        authorizes_method_profile=True,
    )
    profile = PaperMethodEvidenceProfile(
        paper_id="paper-1",
        insertion_points=["trainer_loss"],
        observations=[observation],
        authorizes_method_profile=True,
        authorization_reasons=["explicit_method_boundary"],
    )

    assert profile.with_hash().evidence_hash == profile.with_hash().evidence_hash


@pytest.mark.parametrize("source", ["title", "harness_hint", "category"])
def test_prior_only_sources_cannot_authorize_method_profile(source: str) -> None:
    with pytest.raises(ValidationError, match="cannot authorize"):
        PaperMethodEvidenceObservation(
            field_name="method_family",
            value="distillation",
            source=source,
            source_location=f"paper_record.{source}",
            confidence="medium",
            authorizes_method_profile=True,
        )


def test_low_confidence_evidence_cannot_authorize_method_profile() -> None:
    with pytest.raises(ValidationError, match="low-confidence"):
        PaperMethodEvidenceObservation(
            field_name="method_family",
            value="sampling",
            source="summary",
            source_location="summary:sentence:1",
            confidence="low",
            authorizes_method_profile=True,
        )


def test_aggregate_field_requires_source_observation() -> None:
    with pytest.raises(ValidationError, match="lack observations"):
        PaperMethodEvidenceProfile(
            paper_id="paper-2",
            changed_variables=["loss.correlation.weight"],
        )


def test_extracts_explicit_english_method_boundary_and_runtime_hooks() -> None:
    paper = PaperRecord(
        paper_id="sampling-paper",
        title="Small object detector",
        year=2025,
        abstract=(
            "We use small object sampling in the train dataloader. "
            "Image weights define the sampling probability."
        ),
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert profile.canonical_mechanisms == ["sampling.small_object"]
    assert profile.method_families == ["sampling"]
    assert profile.insertion_points == ["train_dataloader_sampler"]
    assert profile.changed_variables == ["data.sampling_policy"]
    assert profile.component_types == ["data"]
    assert profile.required_runtime_hooks == [
        "build_train_dataloader",
        "build_train_dataset",
    ]
    assert profile.authorizes_method_profile is True


def test_extracts_hard_example_mining_without_merging_sampling() -> None:
    paper = PaperRecord(
        paper_id="ohem-paper",
        title="Detector training",
        year=2025,
        abstract="Online hard example mining selects difficult losses during training.",
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert profile.method_families == ["hard_example_mining"]
    assert profile.insertion_points == ["trainer_loss"]
    assert profile.changed_variables == ["loss.hard_example_ratio"]
    assert "sampling" not in profile.method_families
    assert profile.authorizes_method_profile is True


@pytest.mark.parametrize(
    ("phrase", "mechanism", "changed_variable"),
    [
        (
            "small-object weighted sampling",
            "sampling.small_object_weighted",
            "data.small_object_weighted_sampling",
        ),
        (
            "class-balanced sampling",
            "sampling.class_balanced",
            "data.class_balanced_sampling",
        ),
        (
            "repeat-factor sampling",
            "sampling.repeat_factor",
            "data.repeat_factor_sampling",
        ),
        (
            "hard-negative replay",
            "sampling.hard_negative_replay",
            "data.hard_negative_replay",
        ),
        (
            "false-negative class boost",
            "sampling.false_negative_class_boost",
            "data.false_negative_class_boost",
        ),
        (
            "rare-class copy-paste",
            "augmentation.copy_paste_rare_classes",
            "data.copy_paste_rare_classes",
        ),
        (
            "scale-aware crop",
            "augmentation.scale_aware_crop",
            "data.scale_aware_crop",
        ),
        (
            "object-centric crop",
            "augmentation.object_centric_crop",
            "data.object_centric_crop",
        ),
        (
            "multi-image sampling schedule",
            "augmentation.multi_image_sampling_schedule",
            "data.multi_image_sampling_schedule",
        ),
    ],
)
def test_extracts_precise_data_mechanism_boundary(
    phrase: str,
    mechanism: str,
    changed_variable: str,
) -> None:
    paper = PaperRecord(
        paper_id=mechanism,
        title="Data pipeline detector",
        year=2025,
        abstract=f"We apply {phrase} in the training data loader as a sampling policy.",
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert mechanism in profile.canonical_mechanisms
    assert profile.changed_variables == [changed_variable]
    assert profile.component_types == ["data"]
    assert profile.authorizes_method_profile is True


def test_extracts_logits_distillation_boundary() -> None:
    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(PaperRecord(
        paper_id="logits-distillation",
        title="Detector distillation",
        year=2025,
        abstract=(
            "Logits distillation adds an auxiliary loss for teacher output "
            "distributions."
        ),
    ))

    assert profile.insertion_points == ["logits_distillation", "trainer_loss"]
    assert profile.changed_variables == [
        "distillation.logits.weight",
        "loss.auxiliary.weight",
    ]
    assert profile.authorizes_method_profile is True


def test_title_only_mechanism_remains_low_confidence_prior() -> None:
    paper = PaperRecord(
        paper_id="title-only",
        title="Teacher Student Distillation for Detection",
        year=2025,
        abstract="Object detection study.",
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert "distillation.yolo26_teacher_student" in profile.canonical_mechanisms
    assert profile.authorizes_method_profile is False
    mechanism = next(
        item for item in profile.observations
        if item.field_name == "canonical_mechanism"
    )
    assert mechanism.source == "title"
    assert mechanism.confidence == "low"
    assert mechanism.authorizes_method_profile is False


def test_extracts_mixed_chinese_english_method_evidence() -> None:
    paper = PaperRecord(
        paper_id="zh-sampling",
        title="小目标检测方法",
        year=2025,
        abstract=(
            "我们在训练数据加载器中使用 small object sampling，"
            "图像权重控制采样概率，推理结构不变。"
        ),
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert profile.canonical_mechanisms == ["sampling.small_object"]
    assert profile.insertion_points == ["train_dataloader_sampler"]
    assert profile.changed_variables == ["data.sampling_policy"]
    assert profile.training_only is None
    assert profile.inference_changed is False
    assert profile.authorizes_method_profile is True


def test_markdown_table_and_formula_context_extract_semantics_not_benchmark() -> None:
    paper = PaperRecord(
        paper_id="table-loss",
        title="Quality detector",
        year=2025,
        abstract=(
            "| Component | Insertion | Variable |\n"
            "|---|---|---|\n"
            "| correlation loss | auxiliary loss | auxiliary loss weight |\n"
            "The training objective is L = L_native + lambda L_corr; AP=42.1."
        ),
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert profile.canonical_mechanisms == ["loss.quality.correlation"]
    assert profile.insertion_points == ["trainer_loss"]
    assert profile.changed_variables == ["loss.auxiliary.weight"]
    assert profile.component_types == ["loss"]
    dumped = profile.model_dump(mode="json")
    assert "42.1" not in str(dumped)


@pytest.mark.parametrize(
    ("text", "mechanism"),
    [
        ("OTA label assignment modifies the target assigner.", "assigner.optimal_transport"),
        ("DSLA 标签分配修改目标分配器。", "assigner.dynamic_smooth_label"),
        ("SAHI uses sliced inference and a slice overlap policy.", "inference.sahi_slicing"),
    ],
)
def test_abbreviations_and_synonyms_are_source_grounded(
    text: str,
    mechanism: str,
) -> None:
    paper = PaperRecord(
        paper_id=mechanism,
        title="Detector",
        year=2025,
        abstract=text,
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert mechanism in profile.canonical_mechanisms
    assert all(item.source_location for item in profile.observations)


def test_title_echo_catalog_summary_remains_prior_only() -> None:
    paper = PaperRecord(
        paper_id="cvpr-template",
        title="Scale-Aware Automatic Augmentation for Object Detection",
        year=2021,
        abstract=(
            "CVPR 2021 paper on Scale-Aware Automatic Augmentation "
            "for Object Detection."
        ),
        component_ids=["object_detection"],
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert "augmentation" in profile.method_families
    assert profile.authorizes_method_profile is False
    assert all(item.authorizes_method_profile is False for item in profile.observations)


def test_explicit_noncanonical_method_boundary_creates_authorizing_evidence() -> None:
    paper = PaperRecord(
        paper_id="augmentation-summary",
        title="Detection augmentation",
        year=2025,
        abstract=(
            "The method applies scale-aware augmentation to training data "
            "and changes the augmentation policy."
        ),
        component_ids=["object_detection"],
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert profile.canonical_mechanisms == []
    assert profile.method_families == ["augmentation"]
    assert profile.changed_variables == ["data.augmentation_policy"]
    assert profile.authorizes_method_profile is True


def test_malformed_text_is_non_blocking_and_deterministic() -> None:
    paper = PaperRecord(
        paper_id="garbled",
        title="Detector",
        year=2025,
        abstract="\ufffd\ufffd\x00 malformed \ufffd text small_object_sampling",
    )
    extractor = PaperMethodEvidenceExtractor(ComponentAliasResolver.from_yaml())

    first = extractor.extract(paper)
    second = extractor.extract(paper)

    assert first == second
    assert first.evidence_hash == second.evidence_hash
    assert first.authorizes_method_profile is False


def test_extractor_does_not_create_benchmark_or_local_evidence_fields() -> None:
    paper = PaperRecord(
        paper_id="paper-claim-only",
        title="Quality detector",
        year=2025,
        abstract=(
            "Correlation loss is an auxiliary loss with AP=42.1 and +2.0 mAP."
        ),
    )

    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)
    dumped = profile.model_dump(mode="json")

    assert "benchmark" not in dumped
    assert "reported_delta" not in dumped
    assert "local_evidence" not in dumped
    assert all(item.evidence_level == "paper_prior" for item in profile.observations)


def test_extractor_never_calls_network(monkeypatch) -> None:
    def fail_network(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr("urllib.request.urlopen", fail_network)
    paper = PaperRecord(
        paper_id="offline",
        title="Small object detector",
        year=2025,
        abstract=(
            "Small object sampling modifies image weights in the train dataloader."
        ),
        official_code_url="https://github.com/owner/project",
    )

    result = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(paper)

    assert result.authorizes_method_profile is True


@pytest.mark.parametrize(
    ("text", "expected_hooks"),
    [
        (
            "Scale-aware augmentation changes the augmentation policy in training data.",
            ["build_train_dataloader", "build_train_dataset"],
        ),
        (
            "Feature distillation adds a feature distillation objective.",
            ["build_criterion", "compute_loss"],
        ),
        (
            "Model quantization changes the post-training model quantization policy.",
            ["checkpoint_load", "checkpoint_save"],
        ),
    ],
)
def test_generic_method_boundaries_derive_runtime_hooks(
    text: str,
    expected_hooks: list[str],
) -> None:
    profile = PaperMethodEvidenceExtractor(
        ComponentAliasResolver.from_yaml()
    ).extract(PaperRecord(
        paper_id=text[:12],
        title="Detector",
        year=2025,
        abstract=text,
    ))

    assert profile.required_runtime_hooks == expected_hooks
