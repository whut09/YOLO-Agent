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
