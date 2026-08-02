from __future__ import annotations

import pytest
from pydantic import ValidationError

from yolo_agent.research.mechanism_clusters import (
    ClusterEvidence,
    MechanismClusterConfig,
    MechanismClusterDefinition,
    PaperMechanismClusterMatch,
)


def _cluster(cluster_id: str, semantic: str) -> MechanismClusterDefinition:
    return MechanismClusterDefinition(
        cluster_id=cluster_id,
        display_name=cluster_id,
        training_semantic=semantic,
        adapter_family="loss.shared",
        method_families=[cluster_id],
        insertion_points=["trainer_loss"],
        required_runtime_hooks=["compute_loss"],
    )


def test_cluster_config_rejects_duplicate_training_semantics() -> None:
    with pytest.raises(ValidationError, match="duplicate adapter-family"):
        MechanismClusterConfig(clusters=[
            _cluster("quality-a", "auxiliary_quality_target"),
            _cluster("quality-b", "auxiliary_quality_target"),
        ])


def test_semantic_match_requires_evidence_and_confidence() -> None:
    with pytest.raises(ValidationError, match="requires source evidence"):
        PaperMechanismClusterMatch(
            paper_id="paper",
            profile_id="profile",
            cluster_id="quality_alignment",
            adapter_family="loss.quality_alignment",
            training_semantic="auxiliary_quality_target",
            match_type="semantic_match",
            confidence="medium",
            confidence_score=0.7,
            match_reason="semantic alias",
        )

    match = PaperMechanismClusterMatch(
        paper_id="paper",
        profile_id="profile",
        cluster_id="quality_alignment",
        adapter_family="loss.quality_alignment",
        training_semantic="auxiliary_quality_target",
        match_type="semantic_match",
        confidence="high",
        confidence_score=0.9,
        evidence=[ClusterEvidence(
            field_name="method_family",
            value="quality_alignment",
            source="summary",
            source_location="summary:paragraph:1",
            confidence="high",
        )],
        match_reason="explicit method family and trainer loss insertion",
    )

    assert match.evidence[0].source_location == "summary:paragraph:1"
