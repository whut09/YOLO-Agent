from __future__ import annotations

from datetime import datetime, timezone

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import ComponentMaturityArtifact
from yolo_agent.research.coverage_acceptance import PaperCoverageAcceptanceBuilder
from yolo_agent.research.executable_coverage_schemas import (
    ExecutablePaperCoverageBaseline,
    PaperCoverageDenominator,
    PaperExecutableCoverageEntry,
)
from yolo_agent.research.method_profiles import (
    CanonicalMechanismCoverage,
    CompatibleMechanismCoverage,
    PaperEvidenceInventory,
    PaperImplementationDecision,
    PaperMechanismMapping,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)


def _profile(paper_id: str, mechanism_id: str) -> PaperMethodProfile:
    return PaperMethodProfile(
        profile_id=f"profile:{paper_id}",
        paper_id=paper_id,
        canonical_component_ids=[mechanism_id],
        source_locations=[f"catalog:{paper_id}"],
        evidence_inventory=PaperEvidenceInventory(
            source_locations=[f"catalog:{paper_id}"]
        ),
    )


def _decision(
    paper_id: str,
    mechanism_id: str,
    adapter_id: str | None,
) -> PaperImplementationDecision:
    mapping = PaperMechanismMapping(
        paper_id=paper_id,
        profile_id=f"profile:{paper_id}",
        source_term=mechanism_id,
        source="summary",
        source_location=f"catalog:{paper_id}",
        canonical_component_id=mechanism_id,
        alias_match_type="exact",
        yolo26_compatibility="compatible",
        implementation_status="adapter_implemented" if adapter_id else "metadata_only",
        reusable_adapter_id=adapter_id,
    )
    kwargs = {
        "paper_id": paper_id,
        "profile_id": f"profile:{paper_id}",
        "canonical_component_ids": [mechanism_id],
        "mechanism_mappings": [mapping],
        "source_locations": [f"catalog:{paper_id}"],
    }
    if adapter_id:
        return PaperImplementationDecision(
            decision="reuse_existing_adapter",
            reusable_adapter_ids=[adapter_id],
            **kwargs,
        ).with_hash()
    return PaperImplementationDecision(
        decision="new_method_profile",
        reasons=["adapter is not implemented"],
        **kwargs,
    ).with_hash()


def _contract(
    component_id: str,
    maturity: str,
) -> ComponentContract:
    artifacts = []
    if maturity == "smoke_passed":
        artifacts = [
            ComponentMaturityArtifact(
                component_id=component_id,
                artifact_type="smoke_report",
                artifact_path=f"artifacts/{component_id}.yaml",
                artifact_sha256="a" * 64,
                protocol_hash="c" * 64,
                target_maturity="smoke_passed",
                status="passed",
                producer="test",
            )
        ]
    return ComponentContract(
        component_id=component_id,
        display_name=component_id,
        category="data_pipeline",
        maturity=maturity,
        maturity_artifacts=artifacts,
    )


def _fixtures() -> tuple[PaperMethodCoverageReport, ExecutablePaperCoverageBaseline]:
    paper_specs = [
        ("p1", "mechanism.a", "adapter.a", "yolo26_runtime_ready"),
        ("p2", "mechanism.a", "adapter.a", "yolo26_adapter_available"),
        ("p3", "mechanism.b", None, "yolo26_adapter_required"),
        ("p4", "mechanism.c", None, "separate_detector_family"),
        ("p5", "mechanism.d", None, "insufficient_information"),
    ]
    profiles = [_profile(paper_id, mechanism_id) for paper_id, mechanism_id, _, _ in paper_specs]
    decisions = [
        _decision(paper_id, mechanism_id, adapter_id)
        for paper_id, mechanism_id, adapter_id, _ in paper_specs
    ]
    method = PaperMethodCoverageReport(
        paper_count=5,
        profile_count=5,
        decision_counts={
            "reuse_existing_adapter": 2,
            "new_method_profile": 3,
        },
        profiles=profiles,
        decisions=decisions,
        compatible_mechanism_coverage=CompatibleMechanismCoverage(
            referenced_mechanism_count=4,
            compatible_mechanism_count=2,
            potentially_adaptable_mechanism_count=2,
            reusable_adapter_mechanism_count=1,
            runtime_ready_mechanism_count=1,
            compatible_adapter_coverage_ratio=0.5,
            runtime_ready_coverage_ratio=0.5,
            mechanisms=[
                CanonicalMechanismCoverage(
                    canonical_component_id="mechanism.a",
                    paper_ids=["p1", "p2"],
                    reference_count=2,
                    yolo26_compatibility="compatible",
                    reusable_adapter=True,
                ),
                CanonicalMechanismCoverage(
                    canonical_component_id="mechanism.b",
                    paper_ids=["p3"],
                    reference_count=1,
                    yolo26_compatibility="adapter_required",
                ),
            ],
        ),
    )
    entries = [
        PaperExecutableCoverageEntry(
            paper_id=paper_id,
            profile_id=f"profile:{paper_id}",
            decision=decision.decision,
            compatibility_class=compatibility,
            adaptation_scope=(
                "single_component"
                if compatibility not in {"separate_detector_family", "insufficient_information"}
                else ("whole_detector" if compatibility == "separate_detector_family" else "none")
            ),
            canonical_mechanisms=[mechanism_id],
            reusable_adapter_candidates=[adapter_id] if adapter_id else [],
            source_locations=[f"catalog:{paper_id}"],
        )
        for (paper_id, mechanism_id, adapter_id, compatibility), decision in zip(
            paper_specs, decisions, strict=True
        )
    ]
    all_ids = ["p1", "p2", "p3", "p4", "p5"]
    compatible_ids = ["p1", "p2", "p3"]
    baseline = ExecutablePaperCoverageBaseline(
        source_method_coverage_hash="d" * 64,
        source_taxonomy_hash="e" * 64,
        generated_at=datetime.now(timezone.utc),
        denominators={
            "all_papers": PaperCoverageDenominator(
                name="all_papers", definition="all", paper_count=5, paper_ids=all_ids
            ),
            "yolo26_compatible_papers": PaperCoverageDenominator(
                name="yolo26_compatible_papers",
                definition="compatible",
                paper_count=3,
                paper_ids=compatible_ids,
            ),
            "adaptable_component_papers": PaperCoverageDenominator(
                name="adaptable_component_papers",
                definition="adaptable",
                paper_count=3,
                paper_ids=compatible_ids,
            ),
            "exact_reproduction_candidates": PaperCoverageDenominator(
                name="exact_reproduction_candidates",
                definition="exact",
                paper_count=0,
                paper_ids=[],
            ),
        },
        compatibility_counts={
            "yolo26_runtime_ready": 1,
            "yolo26_adapter_available": 1,
            "yolo26_adapter_required": 1,
            "separate_detector_family": 1,
            "insufficient_information": 1,
        },
        reusable_adapter_paper_count=2,
        adapter_to_papers={"adapter.a": ["p1", "p2"]},
        mechanism_to_papers={
            "mechanism.a": ["p1", "p2"],
            "mechanism.b": ["p3"],
            "mechanism.c": ["p4"],
            "mechanism.d": ["p5"],
        },
        entries=entries,
    )
    return method, baseline


def test_acceptance_uses_compatible_denominators_and_traces_artifacts() -> None:
    method, baseline = _fixtures()
    report = PaperCoverageAcceptanceBuilder(
        effective_contracts={"adapter.a": _contract("adapter.a", "smoke_passed")}
    ).build(method, baseline, source_method_coverage_hash="f" * 64)

    papers = report.metrics["compatible_papers_certified_adapter"]
    mechanisms = report.metrics["compatible_mechanism_smoke_passed"]
    assert papers.denominator_ids == ["p1", "p2", "p3"]
    assert papers.numerator_ids == ["p1", "p2"]
    assert mechanisms.denominator_ids == ["mechanism.a", "mechanism.b"]
    assert mechanisms.numerator_ids == ["mechanism.a"]
    assert report.adapter_traces[0].artifacts[0].artifact_sha256 == "a" * 64
    assert report.separate_detector_family_paper_ids == ["p4"]
    assert report.insufficient_information_paper_ids == ["p5"]
    assert report.exact_reproduction_paper_ids == []


def test_acceptance_reports_ceil_gaps_and_highest_yield_mechanisms() -> None:
    method, baseline = _fixtures()
    report = PaperCoverageAcceptanceBuilder(effective_contracts={}).build(
        method, baseline, source_method_coverage_hash="f" * 64
    )

    gap = next(
        item
        for item in report.gaps
        if item.metric_id == "compatible_papers_certified_adapter"
    )
    assert gap.additional_required == 3
    assert gap.missing_ids == ["p1", "p2", "p3"]
    assert report.next_mechanisms[0].mechanism_id == "mechanism.a"
    assert report.next_mechanisms[0].covered_paper_count == 2
    assert report.status == "failed"
