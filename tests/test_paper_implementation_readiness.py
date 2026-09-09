from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from yolo_agent.components.contracts import load_contracts
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMechanismMapping,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)
from yolo_agent.research.paper_83_campaign_schemas import (
    Paper83Manifest,
    Paper83Paper,
)
from yolo_agent.research.paper_implementation_readiness import (
    PaperImplementationReadinessEvaluator,
)
from yolo_agent.research.paper_implementation_registry import (
    PaperImplementationRegistryBuilder,
)
from yolo_agent.research.paper_implementation_schemas import (
    PaperImplementationRegistry,
    PaperImplementationSpec,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "research" / "paper_83_manifest.yaml"
METHOD_COVERAGE_PATH = ROOT / "research" / "production" / "paper_method_coverage.yaml"
CONTRACTS_PATH = ROOT / "research" / "production" / "component_contracts.yaml"


def _paper(
    paper_id: str = "paper-a",
    *,
    membership_hash: str = "a" * 64,
    components: list[str] | None = None,
    adapters: list[str] | None = None,
) -> Paper83Paper:
    return Paper83Paper(
        paper_id=paper_id,
        title=f"Title {paper_id}",
        year=2024,
        source="fixture",
        method_profile_id=f"profile:{paper_id}",
        frozen_certified_adapter_ids=["loss.quality.correlation"],
        current_component_ids=components or [],
        current_adapter_ids=adapters or [],
        current_disposition="implementation_request",
    )


def _profile(
    paper_id: str = "paper-a",
    *,
    paper_components: list[str] | None = None,
    canonical_components: list[str] | None = None,
    parameters: dict[str, object] | None = None,
    protocol: dict[str, object] | None = None,
) -> PaperMethodProfile:
    return PaperMethodProfile(
        profile_id=f"profile:{paper_id}",
        paper_id=paper_id,
        paper_component_ids=paper_components or [],
        canonical_component_ids=canonical_components or [],
        paper_parameters=parameters or {},
        protocol_constraints=protocol or {},
        source_locations=["fixture:paper-method"],
    )


def _decision(
    paper_id: str = "paper-a",
    *,
    mechanism_mappings: list[PaperMechanismMapping] | None = None,
) -> PaperImplementationDecision:
    return PaperImplementationDecision(
        paper_id=paper_id,
        profile_id=f"profile:{paper_id}",
        decision="reuse_existing_adapter",
        canonical_component_ids=["loss.quality.correlation"],
        reusable_adapter_ids=["loss.quality.correlation"],
        reasons=["fixture"],
        mechanism_mappings=mechanism_mappings or [],
    )


def _authorized_mapping(paper_id: str, source_term: str) -> PaperMechanismMapping:
    return PaperMechanismMapping(
        paper_id=paper_id,
        profile_id=f"profile:{paper_id}",
        source_term=source_term,
        source="summary",
        source_location="fixture:summary",
        canonical_component_id="loss.quality.correlation",
        alias_match_type="exact_match",
        yolo26_compatibility="compatible",
        implementation_status="smoke_passed",
        reusable_adapter_id="loss.quality.correlation",
        reusable_adapter_ids=["loss.quality.correlation"],
        adapter_verified=True,
        runtime_execution_ready=True,
        confidence="high",
        authorizes_method_profile=True,
    )


def _ready_spec(
    paper_id: str,
    *,
    membership_hash: str = "a" * 64,
    evidence_class: str = "paper_specific",
) -> PaperImplementationSpec:
    fingerprint = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()
    return PaperImplementationSpec(
        paper_id=paper_id,
        manifest_membership_hash=membership_hash,
        method_profile_id=f"profile:{paper_id}",
        title=f"Title {paper_id}",
        year=2024,
        implementation_domain="yolo26:component_adaptation",
        mechanism_summary="paper-specific mechanism",
        paper_specific_mechanisms=["paper-specific mechanism"],
        shared_primitives=["shared.primitive"],
        component_ids=["shared.primitive"],
        adapter_ids=["shared.primitive"],
        runtime_insertion_points=["trainer_loss"],
        runtime_hooks=["compute_loss"],
        paper_specific_config={"weight": 0.2},
        paper_evidence_refs=["fixture:paper"],
        unit_test_refs=["tests/test_fixture.py"],
        smoke_test_refs=["tests/test_fixture.py"],
        compatibility_test_refs=["tests/test_fixture.py"],
        implementation_fingerprint=fingerprint,
        readiness="implementation_ready",
        implementation_evidence_class=evidence_class,  # type: ignore[arg-type]
        runtime_implementation_verified=True,
        unit_tests_passed=True,
        non_mock_smoke_passed=True,
        compatibility_validation_passed=True,
    )


def test_generic_adapter_is_not_paper_implementation_ready() -> None:
    evaluator = PaperImplementationReadinessEvaluator(
        manifest_membership_hash="a" * 64,
        contracts={},
        tests_root=None,
    )

    spec = evaluator.evaluate(
        paper=_paper(),
        profile=_profile(
            paper_components=["knowledge_distillation"],
            canonical_components=["distillation.yolo26_teacher_student"],
        ),
        decision=_decision(),
    )

    assert spec.implementation_evidence_class == "generic_only"
    assert spec.readiness != "implementation_ready"
    assert "paper_specific_mechanism_missing" in " ".join(spec.blockers)


def test_mock_smoke_cannot_be_paper_implementation_ready() -> None:
    with pytest.raises(ValueError, match="paper-specific non-mock evidence"):
        _ready_spec("paper-mock", evidence_class="mock_only")


def test_shared_primitive_can_support_two_independent_paper_specs() -> None:
    first = _ready_spec("paper-a")
    second = _ready_spec("paper-b")
    registry = PaperImplementationRegistry(
        manifest_path="fixture/paper-83.yaml",
        manifest_membership_hash="a" * 64,
        paper_count=2,
        records=[first, second],
    ).with_hash()

    assert registry.implementation_ready_count == 2
    assert [item.paper_id for item in registry.records] == ["paper-a", "paper-b"]
    assert first.implementation_fingerprint != second.implementation_fingerprint


def test_two_paper_readiness_records_are_evaluated_independently() -> None:
    evaluator = PaperImplementationReadinessEvaluator(
        manifest_membership_hash="a" * 64,
        contracts={},
        tests_root=None,
    )
    generic = evaluator.evaluate(
        paper=_paper("paper-generic"),
        profile=_profile(
            "paper-generic",
            paper_components=["domain_adaptation"],
            canonical_components=["domain_adaptation.general"],
        ),
        decision=_decision("paper-generic"),
    )
    specific = evaluator.evaluate(
        paper=_paper("paper-specific"),
        profile=_profile(
            "paper-specific",
            paper_components=["correlation_loss"],
            canonical_components=["loss.quality.correlation"],
            parameters={"changed_variables": ["loss.correlation.weight"]},
            protocol={
                "insertion_points": ["trainer_loss"],
                "required_runtime_hooks": ["compute_loss"],
            },
        ),
        decision=_decision(
            "paper-specific",
            mechanism_mappings=[
                _authorized_mapping("paper-specific", "correlation_loss")
            ],
        ),
    )

    assert generic.paper_id != specific.paper_id
    assert generic.implementation_evidence_class == "generic_only"
    assert specific.implementation_evidence_class == "paper_specific"
    assert generic.readiness != "implementation_ready"


def test_missing_paper_config_blocks_specific_mechanism() -> None:
    evaluator = PaperImplementationReadinessEvaluator(
        manifest_membership_hash="a" * 64,
        contracts={},
        tests_root=None,
    )

    spec = evaluator.evaluate(
        paper=_paper("paper-no-config"),
        profile=_profile("paper-no-config", paper_components=["new_loss"]),
        decision=_decision(
            "paper-no-config",
            mechanism_mappings=[
                _authorized_mapping("paper-no-config", "new_loss")
            ],
        ),
    )

    assert spec.implementation_evidence_class == "paper_specific"
    assert "missing_paper_config" in spec.blockers
    assert spec.readiness != "implementation_ready"


def test_unresolved_mapping_cannot_authorize_a_paper_mechanism() -> None:
    evaluator = PaperImplementationReadinessEvaluator(
        manifest_membership_hash="a" * 64,
        contracts={},
        tests_root=None,
    )

    mapping = _authorized_mapping("paper-unresolved", "new_loss").model_copy(
        update={"authorizes_method_profile": False}
    )
    spec = evaluator.evaluate(
        paper=_paper("paper-unresolved"),
        profile=_profile(
            "paper-unresolved",
            paper_components=["new_loss"],
            canonical_components=["loss.quality.correlation"],
        ),
        decision=_decision("paper-unresolved", mechanism_mappings=[mapping]),
    )

    assert spec.paper_specific_mechanisms == []
    assert spec.implementation_evidence_class == "metadata_only"
    assert any(
        blocker.endswith("paper_specific_mechanism_missing")
        for blocker in spec.blockers
    )


def test_generic_canonical_mapping_cannot_become_paper_specific() -> None:
    evaluator = PaperImplementationReadinessEvaluator(
        manifest_membership_hash="a" * 64,
        contracts={},
        tests_root=None,
    )
    mapping = _authorized_mapping(
        "paper-generic-canonical", "day_night_adaptation"
    ).model_copy(update={
        "canonical_component_id": "domain_adaptation.general",
        "reusable_adapter_id": "domain_adaptation.general",
        "reusable_adapter_ids": ["domain_adaptation.general"],
    })

    spec = evaluator.evaluate(
        paper=_paper("paper-generic-canonical"),
        profile=_profile(
            "paper-generic-canonical",
            paper_components=["day_night_adaptation"],
            canonical_components=["domain_adaptation.general"],
        ),
        decision=PaperImplementationDecision(
            paper_id="paper-generic-canonical",
            profile_id="profile:paper-generic-canonical",
            decision="reuse_existing_adapter",
            canonical_component_ids=["domain_adaptation.general"],
            reusable_adapter_ids=["domain_adaptation.general"],
            reasons=["fixture"],
            mechanism_mappings=[mapping],
        ),
    )

    assert spec.paper_specific_mechanisms == []
    assert spec.implementation_evidence_class == "generic_only"
    assert spec.readiness != "implementation_ready"


def test_non_mock_production_smoke_evidence_is_visible_to_paper_audit() -> None:
    contracts = {
        item.component_id: item
        for item in load_contracts(CONTRACTS_PATH)
        if item.component_id == "loss.quality.correlation"
    }
    contract = contracts["loss.quality.correlation"]
    assert contract.can_execute

    manifest = Paper83Manifest.from_yaml(MANIFEST_PATH)
    paper = next(item for item in manifest.papers if item.paper_id == "arxiv:2301.01019")
    coverage = PaperMethodCoverageReport.from_yaml(METHOD_COVERAGE_PATH)
    profile = next(item for item in coverage.profiles if item.paper_id == paper.paper_id)
    decision = next(item for item in coverage.decisions if item.paper_id == paper.paper_id)
    evaluator = PaperImplementationReadinessEvaluator(
        manifest_membership_hash=manifest.campaign.membership_hash,
        contracts=contracts,
        tests_root=ROOT / "tests",
    )
    spec = evaluator.evaluate(paper=paper, profile=profile, decision=decision)

    assert spec.non_mock_smoke_passed
    assert not any("mock_only" in blocker for blocker in spec.blockers)


def test_pilot_reproduction_is_not_required_for_implementation_ready() -> None:
    spec = _ready_spec("paper-before-pilot")

    assert spec.is_implementation_ready
    assert not spec.pilot_reproduced


def test_production_builder_uses_exact_frozen_membership() -> None:
    manifest = Paper83Manifest.from_yaml(MANIFEST_PATH)
    registry = PaperImplementationRegistryBuilder(
        manifest_path=MANIFEST_PATH,
        method_coverage_path=METHOD_COVERAGE_PATH,
        inventory_path=ROOT / "runs" / "coverage-audit" / "paper_execution_inventory.yaml",
        coverage_path=ROOT / "research" / "production" / "coverage_baseline.yaml",
        contracts_path=CONTRACTS_PATH,
        tests_root=ROOT / "tests",
    ).build()

    assert registry.paper_count == 83
    assert {item.paper_id for item in registry.records} == {
        item.paper_id for item in manifest.papers
    }
    assert len({item.paper_id for item in registry.records}) == 83
    records = {item.paper_id: item for item in registry.records}
    for paper_id in ("arxiv:2303.13853", "arxiv:2603.18541"):
        assert records[paper_id].paper_specific_mechanisms == []
        assert records[paper_id].implementation_evidence_class == "generic_only"
        assert not records[paper_id].is_implementation_ready
    assert "distillation.feature" not in records[
        "cvf:cvpr2021:Dai_General_Instance_Distillation_for_Object_Detection"
    ].shared_primitives


def test_frozen_membership_rejects_a_changed_id(tmp_path: Path) -> None:
    payload = Paper83Manifest.from_yaml(MANIFEST_PATH).model_dump(mode="json")
    payload["papers"][0]["paper_id"] = "paper-not-in-frozen-campaign"
    changed = tmp_path / "changed-manifest.yaml"
    import yaml

    changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="membership_hash|sorted and unique"):
        PaperImplementationRegistryBuilder(manifest_path=changed).build()
