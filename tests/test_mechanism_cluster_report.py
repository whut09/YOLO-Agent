from __future__ import annotations

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.mechanism_cluster_report import (
    build_mechanism_cluster_report,
)
from yolo_agent.research.method_profiles import PaperMethodProfileBuilder
from yolo_agent.research.paper_mechanism_clusterer import PaperMechanismClusterer
from yolo_agent.research.schemas import PaperRecord


def _paper(
    paper_id: str,
    component: str,
    abstract: str,
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=paper_id,
        year=2025,
        abstract=abstract,
        component_ids=[component],
    )


def test_cluster_report_aggregates_parameters_limitations_and_sources() -> None:
    papers = [
        _paper(
            "sampling-a",
            "small_object_sampling",
            "Small object sampling uses image weights in the train dataloader.",
        ),
        _paper(
            "sampling-b",
            "small_object_sampling",
            "Rare class sampling changes the sampling policy in training data.",
        ),
    ]
    coverage = PaperMethodProfileBuilder(
        ComponentAliasResolver.from_yaml()
    ).build(papers)
    clusterer = PaperMechanismClusterer()
    matches, conflicts = clusterer.cluster(coverage)

    report = build_mechanism_cluster_report(
        coverage,
        config=clusterer.config,
        matches=matches,
        conflicts=conflicts,
    )

    summary = next(
        item for item in report.clusters
        if item.cluster_id == "sampling_class_balancing"
    )
    assert summary.paper_ids == ["sampling-a", "sampling-b"]
    assert summary.paper_count == 2
    assert summary.parameter_differences["changed_variables"] == [
        "data.sampling_policy"
    ]
    assert summary.source_locations
    assert report.report_hash


def test_implementation_queue_prioritizes_adapter_family_paper_coverage() -> None:
    papers = [
        _paper(
            "augmentation-a",
            "object_detection",
            "Scale-aware augmentation changes the augmentation policy in training data.",
        ),
        _paper(
            "augmentation-b",
            "object_detection",
            "Synthetic data augmentation changes training data augmentation policy.",
        ),
        _paper(
            "reparam-one",
            "object_detection",
            "Structural reparameterization changes the backbone convolution.",
        ),
    ]
    coverage = PaperMethodProfileBuilder(
        ComponentAliasResolver.from_yaml()
    ).build(papers)
    clusterer = PaperMechanismClusterer()
    matches, conflicts = clusterer.cluster(coverage)
    report = build_mechanism_cluster_report(
        coverage,
        config=clusterer.config,
        matches=matches,
        conflicts=conflicts,
    )

    adapter_required = [
        item
        for item in report.implementation_opportunities
        if item.implementation_status == "adapter_required"
    ]
    assert adapter_required[0].cluster_id == "augmentation"
    assert adapter_required[0].paper_count == 2
    assert adapter_required[0].paper_ids == ["augmentation-a", "augmentation-b"]


def test_runtime_ready_cluster_is_not_ranked_as_new_adapter_work() -> None:
    coverage = PaperMethodProfileBuilder(
        ComponentAliasResolver.from_yaml()
    ).build([_paper(
        "sampling",
        "small_object_sampling",
        "Small object sampling uses image weights in the train dataloader.",
    )])
    clusterer = PaperMechanismClusterer()
    matches, conflicts = clusterer.cluster(coverage)
    report = build_mechanism_cluster_report(
        coverage,
        config=clusterer.config,
        matches=matches,
        conflicts=conflicts,
    )

    sampling = next(
        item for item in report.implementation_opportunities
        if item.cluster_id == "sampling_class_balancing"
    )
    assert sampling.implementation_status in {
        "adapter_available",
        "runtime_ready",
    }
    assert "one_adapter_family_can_cover_multiple_source_papers" not in sampling.reasons
