"""Offline tests for four-denominator executable paper coverage."""

from __future__ import annotations

from yolo_agent.research.component_aliases import ComponentAliasResolver
from yolo_agent.research.executable_coverage import ExecutablePaperCoverageAuditor
from yolo_agent.research.maturity_snapshot import (
    EffectiveComponentMaturityManifest,
    FrozenComponentMaturity,
)
from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
)
from yolo_agent.research.method_profiles import PaperMethodProfileBuilder
from yolo_agent.research.schemas import PaperRecord


def _paper(
    paper_id: str,
    component_ids: list[str],
    *,
    applicability: str = "direct_adapter_candidate",
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=paper_id,
        year=2025,
        component_ids=component_ids,
        applicability=applicability,  # type: ignore[arg-type]
    )


def _report(papers: list[PaperRecord]):  # type: ignore[no-untyped-def]
    resolver = ComponentAliasResolver.from_yaml()
    return resolver, PaperMethodProfileBuilder(resolver).build(papers)


def _runtime_maturity(component_id: str) -> EffectiveComponentMaturityManifest:
    return EffectiveComponentMaturityManifest(
        entries=[
            FrozenComponentMaturity(
                component_id=component_id,
                adapter_hash="a" * 64,
                code_commit="commit",
                ultralytics_version="8.4.test",
                protocol_hash="protocol",
                effective_maturity="smoke_passed",
                runtime_execution_ready=True,
            )
        ]
    )


def test_four_denominators_exclude_separate_detector_family() -> None:
    resolver, method = _report(
        [
            _paper("sampling-a", ["small_object_sampling"]),
            _paper("sampling-b", ["small_object_sampling"]),
            _paper(
                "detr",
                ["open_vocabulary_detection"],
                applicability="separate_detector_family",
            ),
            _paper("unknown", ["unmapped_method"]),
        ]
    )
    baseline = ExecutablePaperCoverageAuditor(
        contracts=resolver.contracts,
        maturity=_runtime_maturity("sampling.small_object"),
    ).build(
        method,
        source_method_coverage_hash="m" * 64,
        source_taxonomy_hash="t" * 64,
    )

    assert baseline.denominators["all_papers"].paper_count == 4
    assert baseline.denominators["yolo26_compatible_papers"].paper_ids == [
        "sampling-a",
        "sampling-b",
    ]
    assert baseline.denominators["adaptable_component_papers"].paper_ids == [
        "sampling-a",
        "sampling-b",
    ]
    assert baseline.denominators["exact_reproduction_candidates"].paper_ids == []
    assert baseline.runtime_ready_paper_count == 2
    assert baseline.adapter_to_papers["sampling.small_object"] == [
        "sampling-a",
        "sampling-b",
    ]
    assert baseline.runtime_adapter_to_papers["sampling.small_object"] == [
        "sampling-a",
        "sampling-b",
    ]

    entries = {item.paper_id: item for item in baseline.entries}
    assert entries["detr"].compatibility_class == "separate_detector_family"
    assert entries["detr"].adaptation_scope == "whole_detector"
    assert entries["detr"].exclusion_reason
    assert entries["unknown"].compatibility_class == "insufficient_information"


def test_adapter_class_without_runtime_artifact_is_not_runtime_ready() -> None:
    resolver, method = _report([_paper("sampling", ["small_object_sampling"])])

    baseline = ExecutablePaperCoverageAuditor(
        contracts=resolver.contracts,
    ).build(
        method,
        source_method_coverage_hash="m" * 64,
        source_taxonomy_hash="t" * 64,
    )

    entry = baseline.entries[0]
    assert entry.reusable_adapter_candidates == ["sampling.small_object"]
    assert entry.runtime_ready_adapters == []
    assert entry.compatibility_class == "yolo26_adapter_available"
    assert baseline.reusable_adapter_paper_count == 1
    assert baseline.runtime_ready_paper_count == 0


def test_one_paper_can_reuse_multiple_adapters_and_hooks() -> None:
    resolver, method = _report(
        [_paper("coupled", ["small_object_sampling", "p2_head"])]
    )
    maturity = EffectiveComponentMaturityManifest(
        entries=[
            *_runtime_maturity("sampling.small_object").entries,
            *_runtime_maturity("head.p2_small_object").entries,
        ]
    )

    entry = ExecutablePaperCoverageAuditor(
        contracts=resolver.contracts,
        maturity=maturity,
    ).build(
        method,
        source_method_coverage_hash="m" * 64,
        source_taxonomy_hash="t" * 64,
    ).entries[0]

    assert entry.adaptation_scope == "coupled_components"
    assert entry.canonical_mechanisms == [
        "head.p2_small_object",
        "sampling.small_object",
    ]
    assert entry.reusable_adapter_candidates == entry.canonical_mechanisms
    assert entry.runtime_ready_adapters == entry.canonical_mechanisms
    assert entry.required_runtime_hooks == [
        "build_model",
        "build_train_dataloader",
    ]


def test_report_hash_and_denominator_membership_are_stable() -> None:
    resolver, method = _report([_paper("sampling", ["small_object_sampling"])])
    auditor = ExecutablePaperCoverageAuditor(contracts=resolver.contracts)

    first = auditor.build(
        method,
        source_method_coverage_hash="m" * 64,
        source_taxonomy_hash="t" * 64,
    )
    second = auditor.build(
        method,
        source_method_coverage_hash="m" * 64,
        source_taxonomy_hash="t" * 64,
    )

    assert first.report_hash == second.report_hash
    assert first.denominators["all_papers"].paper_ids == ["sampling"]


def test_report_rejects_reverse_index_drift() -> None:
    resolver, method = _report([_paper("sampling", ["small_object_sampling"])])
    baseline = ExecutablePaperCoverageAuditor(contracts=resolver.contracts).build(
        method,
        source_method_coverage_hash="m" * 64,
        source_taxonomy_hash="t" * 64,
    )
    payload = baseline.model_dump(mode="json")
    payload["adapter_to_papers"] = {"sampling.small_object": ["wrong-paper"]}
    payload["report_hash"] = ""

    try:
        ExecutablePaperCoverageBaseline.model_validate(payload)
    except ValueError as exc:
        assert "adapter_to_papers does not match" in str(exc)
    else:  # pragma: no cover - report drift must fail closed
        raise AssertionError("drifted reverse index must be rejected")
