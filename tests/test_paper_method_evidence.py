from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.research.paper_method_evidence import (
    PaperMethodEvidenceObservation,
    PaperMethodEvidenceProfile,
)


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
